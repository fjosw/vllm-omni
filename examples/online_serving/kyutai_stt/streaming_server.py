# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-time-style transcription WebSocket server for Kyutai STT.

Protocol:
    Client -> Server:
        {"type": "session.config", "sample_rate": 24000}
        {"type": "audio.chunk", "data": "<base64 PCM16 mono>"}     # repeat
        {"type": "audio.done"}                                       # finish

    Server -> Client:
        {"type": "session.ready", "request_id": "..."}
        {"type": "transcript.delta", "text": "..."}                  # streamed
        {"type": "transcript.done",  "text": "<full>"}
        {"type": "error", "message": "..."}

Behaviour: as audio chunks arrive the server periodically submits a
fresh request to ``AsyncOmni`` containing the cumulative audio buffer
so far, aborts the previous request, and forwards each new transcript
prefix to the client as a delta. First-token latency is bounded below
by ``--commit-interval-ms`` rather than the total audio duration.

Run:
    python examples/online_serving/kyutai_stt/streaming_server.py \\
        --model /path/to/stt-1b-en_fr-trfs --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import math
import uuid
from typing import Any

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from scipy.signal import resample_poly
from vllm import SamplingParams
from vllm.logger import init_logger

from vllm_omni.entrypoints.async_omni import AsyncOmni

logger = init_logger(__name__)


def _decode_pcm16_le(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    pcm = np.frombuffer(raw, dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def _resample_to(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    g = math.gcd(src_sr, dst_sr)
    return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)


class _SessionState:
    """Per-WebSocket-session bookkeeping for re-submission streaming."""

    def __init__(self, target_sr: int) -> None:
        self.target_sr = target_sr
        self.input_sr: int | None = None
        self.audio_chunks: list[np.ndarray] = []  # @ input_sr (will resample on submit)
        self.audio_done = asyncio.Event()
        self.have_audio = asyncio.Event()
        self.closed = False
        # Transcript already shown to the client; only newer prefixes get
        # forwarded as deltas (the LM may shorten an in-progress hypothesis
        # between submissions, in which case we hold the committed text).
        self.committed_text: str = ""

    def add_chunk(self, audio: np.ndarray) -> None:
        self.audio_chunks.append(audio)
        self.have_audio.set()

    def total_input_samples(self) -> int:
        return sum(len(c) for c in self.audio_chunks)

    def cumulative_audio(self) -> np.ndarray:
        if not self.audio_chunks:
            return np.zeros(0, dtype=np.float32)
        if len(self.audio_chunks) == 1:
            audio = self.audio_chunks[0]
        else:
            audio = np.concatenate(self.audio_chunks)
        return _resample_to(audio, int(self.input_sr or self.target_sr), self.target_sr)


def _build_app(
    omni: AsyncOmni,
    *,
    target_sr: int = 24000,
    commit_interval_ms: int = 800,
    max_tokens: int = 374,
) -> FastAPI:
    app = FastAPI()

    @app.websocket("/v1/audio/transcriptions/stream")
    async def transcription_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        session = _SessionState(target_sr=target_sr)

        try:
            cfg_msg = await websocket.receive_json()
            if cfg_msg.get("type") != "session.config":
                await websocket.send_json({"type": "error", "message": "Expected session.config first"})
                await websocket.close()
                return
            session.input_sr = int(cfg_msg.get("sample_rate", target_sr))
            session_id = f"stt-{uuid.uuid4().hex[:8]}"
            await websocket.send_json({"type": "session.ready", "request_id": session_id})

            async def reader() -> None:
                try:
                    while True:
                        msg = await websocket.receive_json()
                        msg_type = msg.get("type")
                        if msg_type == "audio.chunk":
                            session.add_chunk(_decode_pcm16_le(msg["data"]))
                        elif msg_type == "audio.done":
                            session.audio_done.set()
                            session.have_audio.set()
                            return
                        else:
                            await websocket.send_json({"type": "error", "message": f"Unknown type: {msg_type}"})
                except WebSocketDisconnect:
                    session.closed = True
                    session.audio_done.set()
                    session.have_audio.set()

            reader_task = asyncio.create_task(reader())

            commit_interval_s = commit_interval_ms / 1000.0
            cycle = 0
            last_submitted_samples = 0
            try:
                while True:
                    # Wait either for new audio or for done.
                    try:
                        await asyncio.wait_for(session.have_audio.wait(), timeout=commit_interval_s)
                    except asyncio.TimeoutError:
                        pass
                    session.have_audio.clear()

                    if session.closed:
                        break
                    n_input = session.total_input_samples()
                    is_done = session.audio_done.is_set()
                    if n_input == 0:
                        if is_done:
                            break
                        continue
                    # Don't re-submit if no new audio has arrived since last cycle
                    if n_input == last_submitted_samples and not is_done:
                        continue

                    last_submitted_samples = n_input
                    cycle += 1
                    request_id = f"{session_id}-{cycle}"
                    audio = session.cumulative_audio()
                    sampling_params = SamplingParams(
                        temperature=0.0,
                        max_tokens=max_tokens,
                        seed=42,
                        detokenize=True,
                    )
                    prompt = {"prompt": "", "multi_modal_data": {"audio": [audio]}}

                    cycle_text = ""
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
                        text = (request_output.outputs[0].text or "").rstrip()
                        if text and text != cycle_text:
                            cycle_text = text
                        # Mid-cycle: if more audio arrived, abort and start a fresh cycle.
                        if session.have_audio.is_set() and not is_done and session.total_input_samples() > n_input:
                            try:
                                await omni.abort(request_id)
                            except Exception:
                                pass
                            break

                    # Mid-stream the LM's hypothesis can change (e.g. punctuation
                    # before a comma flips when more audio arrives), so we send the
                    # full current transcript as an "update" rather than a strict
                    # additive delta. We also compute and forward a stable-prefix
                    # delta against the previously-committed transcript for clients
                    # that want delta semantics.
                    if cycle_text:
                        await websocket.send_json({"type": "transcript.update", "text": cycle_text})
                        # Find the longest common prefix with the previous transcript
                        # so we can also emit a delta event.
                        prev = session.committed_text
                        common = 0
                        for i in range(min(len(prev), len(cycle_text))):
                            if prev[i] != cycle_text[i]:
                                break
                            common += 1
                        if common < len(prev):
                            await websocket.send_json({"type": "transcript.revise", "drop_chars": len(prev) - common})
                        delta = cycle_text[common:]
                        if delta:
                            await websocket.send_json({"type": "transcript.delta", "text": delta})
                        session.committed_text = cycle_text

                    if is_done and last_submitted_samples == n_input:
                        break

                await websocket.send_json({"type": "transcript.done", "text": session.committed_text})
            finally:
                reader_task.cancel()
                try:
                    await reader_task
                except (asyncio.CancelledError, Exception):
                    pass
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as exc:
            logger.exception("Error handling session")
            try:
                await websocket.send_json({"type": "error", "message": str(exc)})
            except Exception:
                pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    @app.websocket("/v1/realtime")
    async def realtime_endpoint(websocket: WebSocket) -> None:
        """OpenAI Realtime API mapping (transcription-only subset).

        Implements the events client SDKs need for live transcription:
            client -> server:
                session.update
                input_audio_buffer.append
                input_audio_buffer.commit
            server -> client:
                session.created
                conversation.item.input_audio_transcription.delta
                conversation.item.input_audio_transcription.completed
                error
        """
        await websocket.accept()
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        item_id = f"item_{uuid.uuid4().hex[:12]}"
        await websocket.send_json(
            {
                "type": "session.created",
                "session": {
                    "id": session_id,
                    "model": "kyutai-stt",
                    "modalities": ["text"],
                    "input_audio_format": "pcm16",
                },
            }
        )

        session = _SessionState(target_sr=target_sr)
        session.input_sr = target_sr
        commit_received = asyncio.Event()
        committed_text = ""

        async def reader() -> None:
            nonlocal item_id
            try:
                while True:
                    msg = await websocket.receive_json()
                    msg_type = msg.get("type", "")
                    if msg_type == "session.update":
                        fmt = msg.get("session", {}).get("input_audio_format")
                        if fmt and fmt != "pcm16":
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "error": {
                                        "type": "invalid_request_error",
                                        "message": f"Unsupported input_audio_format: {fmt}",
                                    },
                                }
                            )
                    elif msg_type == "input_audio_buffer.append":
                        session.add_chunk(_decode_pcm16_le(msg.get("audio", "")))
                    elif msg_type == "input_audio_buffer.commit":
                        commit_received.set()
                        session.audio_done.set()
                        session.have_audio.set()
                        return
                    elif msg_type == "input_audio_buffer.clear":
                        session.audio_chunks.clear()
                    # Other event types are silently ignored: clients may send
                    # response.create, conversation.item.create, etc. that
                    # don't apply to a pure-transcription endpoint.
            except WebSocketDisconnect:
                session.closed = True
                session.audio_done.set()
                session.have_audio.set()
                commit_received.set()

        reader_task = asyncio.create_task(reader())

        commit_interval_s = commit_interval_ms / 1000.0
        cycle = 0
        last_submitted_samples = 0
        try:
            while True:
                try:
                    await asyncio.wait_for(session.have_audio.wait(), timeout=commit_interval_s)
                except asyncio.TimeoutError:
                    pass
                session.have_audio.clear()
                if session.closed:
                    break
                n_input = session.total_input_samples()
                is_done = session.audio_done.is_set()
                if n_input == 0:
                    if is_done:
                        break
                    continue
                if n_input == last_submitted_samples and not is_done:
                    continue

                last_submitted_samples = n_input
                cycle += 1
                request_id = f"{session_id}-{cycle}"
                audio = session.cumulative_audio()
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=max_tokens,
                    seed=42,
                    detokenize=True,
                )
                prompt = {"prompt": "", "multi_modal_data": {"audio": [audio]}}

                cycle_text = ""
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
                    cycle_text = (request_output.outputs[0].text or "").rstrip()
                    if session.have_audio.is_set() and not is_done and session.total_input_samples() > n_input:
                        try:
                            await omni.abort(request_id)
                        except Exception:
                            pass
                        break

                if cycle_text:
                    common = 0
                    for i in range(min(len(committed_text), len(cycle_text))):
                        if committed_text[i] != cycle_text[i]:
                            break
                        common += 1
                    delta = cycle_text[common:]
                    if delta:
                        await websocket.send_json(
                            {
                                "type": "conversation.item.input_audio_transcription.delta",
                                "item_id": item_id,
                                "content_index": 0,
                                "delta": delta,
                            }
                        )
                    committed_text = cycle_text

                if is_done and last_submitted_samples == n_input:
                    break

            await websocket.send_json(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": item_id,
                    "content_index": 0,
                    "transcript": committed_text,
                }
            )
        finally:
            reader_task.cancel()
            try:
                await reader_task
            except (asyncio.CancelledError, Exception):
                pass
            try:
                await websocket.close()
            except Exception:
                pass

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),
        model: str = Form(""),
        language: str | None = Form(None),
        response_format: str = Form("json"),
    ):
        """OpenAI-compatible Whisper transcription endpoint.

        POST a wav/flac/mp3/etc. file as ``multipart/form-data`` and receive
        either a JSON ``{"text": "..."}`` body (``response_format=json``,
        default) or the bare transcript (``response_format=text``).
        """
        del model  # We expose a single model per server.
        del language  # Kyutai is en/fr only and detects automatically.
        if response_format not in ("json", "text"):
            raise HTTPException(status_code=400, detail=f"Unsupported response_format: {response_format}")

        raw = await file.read()
        try:
            audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not decode audio: {exc}") from exc
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = _resample_to(audio.astype(np.float32), int(sr), target_sr)

        request_id = f"stt-rest-{uuid.uuid4().hex[:8]}"
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            seed=42,
            detokenize=True,
        )
        prompt = {"prompt": "", "multi_modal_data": {"audio": [audio]}}

        text = ""
        async for stage_output in omni.generate(
            prompt,
            request_id=request_id,
            sampling_params_list=[sampling_params],
        ):
            if stage_output.final_output_type != "text":
                continue
            request_output = stage_output.request_output
            if request_output.outputs:
                text = request_output.outputs[0].text or ""

        text = text.strip()
        if response_format == "text":
            return PlainTextResponse(text)
        return JSONResponse({"text": text})

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok"}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/kyutai_speech_to_text.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--commit-interval-ms",
        type=int,
        default=800,
        help="How often to submit a fresh request with the cumulative audio buffer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    omni = AsyncOmni(model=args.model, deploy_config=args.deploy_config)
    app = _build_app(omni, commit_interval_ms=args.commit_interval_ms)
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
