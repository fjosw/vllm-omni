# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal real-time-style transcription WebSocket server for Kyutai STT.

Protocol:
    Client -> Server:
        {"type": "session.config", "model": "...", "sample_rate": 24000}
        {"type": "audio.chunk", "data": "<base64 PCM16 mono>"}   # repeatable
        {"type": "audio.done"}                                    # finish

    Server -> Client:
        {"type": "session.ready"}
        {"type": "transcript.delta", "text": "..."}              # streamed
        {"type": "transcript.done", "text": "<full>"}
        {"type": "error", "message": "..."}

The current implementation buffers all chunks until ``audio.done`` and
runs the model once. The transcript is streamed back token-by-token via
``AsyncOmni.generate()``.

True mid-flight encode (transcript deltas while audio is still arriving)
requires plumbing per-request audio chunks through vllm-omni's
``add_streaming_update_async`` channel and growing the per-position
audio bias inside ``kyutai_preprocess`` accordingly. See the README in
this directory for the roadmap.

Run:
    python examples/online_serving/kyutai_stt/streaming_server.py \\
        --model /path/to/stt-1b-en_fr-trfs --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import math
import uuid
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from scipy.signal import resample_poly
from vllm import SamplingParams
from vllm.logger import init_logger

from vllm_omni.entrypoints.async_omni import AsyncOmni

logger = init_logger(__name__)


def _decode_pcm16_le(b64: str) -> np.ndarray:
    """Decode a base64 little-endian int16 PCM blob to float32 in [-1, 1]."""
    raw = base64.b64decode(b64)
    pcm = np.frombuffer(raw, dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def _resample_to(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample float32 mono audio with scipy (rational ratio polyphase)."""
    if src_sr == dst_sr:
        return audio
    g = math.gcd(src_sr, dst_sr)
    return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)


def _build_app(omni: AsyncOmni, default_model: str, target_sr: int = 24000) -> FastAPI:
    app = FastAPI()

    @app.websocket("/v1/audio/transcriptions/stream")
    async def transcription_endpoint(websocket: WebSocket) -> None:  # noqa: D401
        await websocket.accept()
        request_id = f"stt-{uuid.uuid4().hex}"
        try:
            cfg_msg = await websocket.receive_json()
            if cfg_msg.get("type") != "session.config":
                await websocket.send_json({"type": "error", "message": "Expected session.config first"})
                await websocket.close()
                return
            sample_rate = int(cfg_msg.get("sample_rate", target_sr))
            await websocket.send_json({"type": "session.ready", "request_id": request_id})

            audio_chunks: list[np.ndarray] = []
            while True:
                msg = await websocket.receive_json()
                msg_type = msg.get("type")
                if msg_type == "audio.chunk":
                    audio_chunks.append(_decode_pcm16_le(msg["data"]))
                elif msg_type == "audio.done":
                    break
                else:
                    await websocket.send_json({"type": "error", "message": f"Unknown type: {msg_type}"})

            if not audio_chunks:
                await websocket.send_json({"type": "error", "message": "No audio received"})
                await websocket.close()
                return

            audio = np.concatenate(audio_chunks)
            if sample_rate != target_sr:
                audio = _resample_to(audio, sample_rate, target_sr)

            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=374,
                seed=42,
                detokenize=True,
            )
            prompt = {"prompt": "", "multi_modal_data": {"audio": [audio]}}

            full_text = ""
            async for stage_output in omni.generate(
                prompt,
                request_id=request_id,
                sampling_params_list=[sampling_params],
            ):
                if stage_output.final_output_type != "text":
                    continue
                request_output = stage_output.request_output
                if not request_output.outputs:
                    continue
                text = request_output.outputs[0].text or ""
                if text and text != full_text:
                    delta = text[len(full_text) :]
                    full_text = text
                    if delta:
                        await websocket.send_json({"type": "transcript.delta", "text": delta})

            await websocket.send_json({"type": "transcript.done", "text": full_text})
        except WebSocketDisconnect:
            logger.info("Client disconnected (request_id=%s)", request_id)
        except Exception as exc:
            logger.exception("Error handling session %s", request_id)
            try:
                await websocket.send_json({"type": "error", "message": str(exc)})
            except Exception:
                pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass
            try:
                await omni.abort(request_id)  # type: ignore[attr-defined]
            except Exception:
                pass

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": default_model}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/kyutai_speech_to_text.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    omni = AsyncOmni(model=args.model, deploy_config=args.deploy_config)
    app = _build_app(omni, default_model=args.model)
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
