# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-time-style transcription HTTP+WebSocket server for Kyutai STT.

Three endpoints share a single ``AsyncOmni`` engine and a common
re-submission session loop (see :func:`_run_session`):

* ``POST /v1/audio/transcriptions``           — OpenAI Whisper-shaped REST.
* ``WS   /v1/audio/transcriptions/stream``    — custom protocol with
  ``transcript.update`` snapshots + ``transcript.delta`` / ``.revise`` events.
* ``WS   /v1/realtime``                        — OpenAI Realtime subset
  (``input_audio_buffer.append``/``.commit`` →
  ``conversation.item.input_audio_transcription.delta``/``.completed``).

Behaviour: as audio chunks arrive the server periodically (every
``--commit-interval-ms``) submits a fresh request to ``AsyncOmni`` with
the cumulative audio buffer, aborts the previous in-flight request when
fresher audio is available, and forwards transcript prefixes back to
the client. First-token latency is bounded by the commit interval
rather than the total audio duration.

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
from collections.abc import Awaitable, Callable
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

# Mimi runs at 12.5 Hz, so each audio frame is 1920 samples at 24 kHz.
_MIMI_FRAME_SIZE = 1920


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
    """Per-WebSocket-session bookkeeping for the re-submission loop."""

    def __init__(self, target_sr: int) -> None:
        self.target_sr = target_sr
        self.input_sr: int = target_sr
        self.audio_chunks: list[np.ndarray] = []
        self.audio_done = asyncio.Event()
        self.have_audio = asyncio.Event()
        self.closed = False
        # Last transcript snapshot we emitted to the client; the LM may revise
        # an in-progress hypothesis between cycles, so a delta-based protocol
        # needs to compare against this on each update.
        self.committed_text: str = ""

    def add_chunk(self, audio: np.ndarray) -> None:
        self.audio_chunks.append(audio)
        self.have_audio.set()

    def total_input_samples(self) -> int:
        return sum(len(c) for c in self.audio_chunks)

    def cumulative_audio(self) -> np.ndarray:
        if not self.audio_chunks:
            return np.zeros(0, dtype=np.float32)
        audio = self.audio_chunks[0] if len(self.audio_chunks) == 1 else np.concatenate(self.audio_chunks)
        return _resample_to(audio, self.input_sr, self.target_sr)


def _max_tokens_for_audio(audio: np.ndarray, *, fe_pad_frames: int, safety_frames: int) -> int:
    """Cap decode length to the audio horizon.

    Kyutai emits one text token per audio frame. ``_kyutai_audio_bias_full``
    has shape ``(1 + n_frames, hidden)`` so decode fills up to position
    ``n_frames``; the FE adds ``audio_silence_prefix_seconds + audio_delay_seconds
    + 1`` seconds of trailing silence to flush the model's output-delay
    pipeline. Decoding past the horizon adds zero bias and wanders into
    unconditioned text (typically trailing periods).
    """
    n_frames = max(1, len(audio) // _MIMI_FRAME_SIZE)
    return max(1, n_frames + fe_pad_frames + safety_frames - 1)


async def _run_session(
    omni: AsyncOmni,
    session: _SessionState,
    *,
    session_id: str,
    commit_interval_s: float,
    max_tokens_cap: int,
    fe_pad_frames: int,
    safety_frames: int,
    on_update: Callable[[str], Awaitable[None]],
) -> str:
    """Re-submission loop: submit fresh requests with the cumulative audio
    every commit interval, abort when fresher audio arrives, and call
    ``on_update`` with each new committed transcript snapshot. Returns the
    final transcript when the session completes."""
    cycle = 0
    last_submitted_samples = 0

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
            max_tokens=min(
                max_tokens_cap,
                _max_tokens_for_audio(audio, fe_pad_frames=fe_pad_frames, safety_frames=safety_frames),
            ),
            seed=42,
            detokenize=True,
        )
        prompt = {"prompt": "", "multi_modal_data": {"audio": [audio]}}

        cycle_text = ""
        async for stage_output in omni.generate(prompt, request_id=request_id, sampling_params_list=[sampling_params]):
            if stage_output.final_output_type != "text":
                continue
            request_output = stage_output.request_output
            if not request_output.outputs:
                continue
            cycle_text = (request_output.outputs[0].text or "").rstrip()
            # Abort mid-cycle if fresher audio arrived; the next cycle will
            # re-submit with the larger buffer.
            if session.have_audio.is_set() and not is_done and session.total_input_samples() > n_input:
                try:
                    await omni.abort(request_id)
                except Exception:
                    pass
                break

        if cycle_text and cycle_text != session.committed_text:
            await on_update(cycle_text)
            session.committed_text = cycle_text

        if is_done and last_submitted_samples == n_input:
            break

    return session.committed_text


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _build_app(
    omni: AsyncOmni,
    *,
    target_sr: int = 24000,
    commit_interval_ms: int = 800,
    max_tokens_cap: int = 4096,
    fe_pad_frames: int = 19,
    safety_frames: int = 6,
) -> FastAPI:
    app = FastAPI()
    commit_interval_s = commit_interval_ms / 1000.0

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

            async def on_update(text: str) -> None:
                # Mid-stream the LM's hypothesis can change (e.g. punctuation
                # flips when more audio arrives), so we send the full current
                # transcript as an "update" plus a stable-prefix delta+revise
                # pair for clients that prefer additive semantics.
                await websocket.send_json({"type": "transcript.update", "text": text})
                common = _common_prefix_len(session.committed_text, text)
                if common < len(session.committed_text):
                    await websocket.send_json(
                        {"type": "transcript.revise", "drop_chars": len(session.committed_text) - common}
                    )
                delta = text[common:]
                if delta:
                    await websocket.send_json({"type": "transcript.delta", "text": delta})

            reader_task = asyncio.create_task(reader())
            try:
                final = await _run_session(
                    omni,
                    session,
                    session_id=session_id,
                    commit_interval_s=commit_interval_s,
                    max_tokens_cap=max_tokens_cap,
                    fe_pad_frames=fe_pad_frames,
                    safety_frames=safety_frames,
                    on_update=on_update,
                )
                await websocket.send_json({"type": "transcript.done", "text": final})
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
        """OpenAI Realtime API mapping (transcription-only subset)."""
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

        async def reader() -> None:
            try:
                while True:
                    msg = await websocket.receive_json()
                    msg_type = msg.get("type", "")
                    if msg_type == "input_audio_buffer.append":
                        session.add_chunk(_decode_pcm16_le(msg.get("audio", "")))
                    elif msg_type == "input_audio_buffer.commit":
                        session.audio_done.set()
                        session.have_audio.set()
                        return
                    elif msg_type == "input_audio_buffer.clear":
                        session.audio_chunks.clear()
                    # Other event types (response.create, conversation.item.create,
                    # session.update with non-pcm16 format, etc.) are silently
                    # ignored so existing client SDKs that send them don't break
                    # the session.
            except WebSocketDisconnect:
                session.closed = True
                session.audio_done.set()
                session.have_audio.set()

        async def on_update(text: str) -> None:
            common = _common_prefix_len(session.committed_text, text)
            delta = text[common:]
            if delta:
                await websocket.send_json(
                    {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "item_id": item_id,
                        "content_index": 0,
                        "delta": delta,
                    }
                )

        reader_task = asyncio.create_task(reader())
        try:
            final = await _run_session(
                omni,
                session,
                session_id=session_id,
                commit_interval_s=commit_interval_s,
                max_tokens_cap=max_tokens_cap,
                fe_pad_frames=fe_pad_frames,
                safety_frames=safety_frames,
                on_update=on_update,
            )
            await websocket.send_json(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": item_id,
                    "content_index": 0,
                    "transcript": final,
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
        """OpenAI Whisper-compatible transcription endpoint.

        POST a wav/flac/mp3/etc. file as ``multipart/form-data`` and receive
        either ``{"text": "..."}`` (``response_format=json``, default) or the
        bare transcript (``response_format=text``).
        """
        del model, language  # We expose a single model and detect language automatically.
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
            max_tokens=min(
                max_tokens_cap,
                _max_tokens_for_audio(audio, fe_pad_frames=fe_pad_frames, safety_frames=safety_frames),
            ),
            seed=42,
            detokenize=True,
        )
        prompt = {"prompt": "", "multi_modal_data": {"audio": [audio]}}

        text = ""
        async for stage_output in omni.generate(prompt, request_id=request_id, sampling_params_list=[sampling_params]):
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
        help="How often to re-submit a fresh request with the cumulative audio buffer.",
    )
    parser.add_argument(
        "--fe-pad-frames",
        type=int,
        default=19,
        help=(
            "Trailing-silence padding the feature extractor adds, in Mimi frames. "
            "Default 19 matches the 1B en/fr checkpoint (audio_delay_seconds=0.5 + 1 s)."
        ),
    )
    parser.add_argument(
        "--horizon-safety-frames",
        type=int,
        default=6,
        help="Extra decode-horizon margin (~0.5 s) to flush tokens still in the model's pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    omni = AsyncOmni(model=args.model, deploy_config=args.deploy_config)
    app = _build_app(
        omni,
        commit_interval_ms=args.commit_interval_ms,
        fe_pad_frames=args.fe_pad_frames,
        safety_frames=args.horizon_safety_frames,
    )
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    asyncio.run(uvicorn.Server(config).serve())


if __name__ == "__main__":
    main()
