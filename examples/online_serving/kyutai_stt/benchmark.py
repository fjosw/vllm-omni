# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the streaming-transcription server.

Measures three numbers per scenario:
* TTFT - time-to-first-transcript-token from the server's first delta.
* RTF  - real-time factor: wall-clock / audio-duration. <1 = faster than real time.
* total wall - end-to-end time from first chunk sent to transcript.done.

Scenarios:
* offline-rest: POST the wav to /v1/audio/transcriptions (Whisper API).
* ws-fast:      WebSocket, audio chunks pushed as fast as possible.
* ws-realtime:  WebSocket, chunks paced at 1x real time.
* ws-concurrent: N clients simultaneously, real-time pacing.

Run:
    python examples/online_serving/kyutai_stt/benchmark.py \\
        --base-url http://127.0.0.1:8765 \\
        --audio /path/to/audio.wav --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import sys
import time

import httpx
import numpy as np
import websockets
from vllm.multimodal.media.audio import load_audio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8765")
    p.add_argument("--audio", required=True)
    p.add_argument("--chunk-ms", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--reps", type=int, default=2, help="Repetitions per scenario for warm cache.")
    return p.parse_args()


async def _ws_session(
    *,
    ws_url: str,
    pcm16: bytes,
    chunk_bytes: int,
    chunk_ms: int,
    realtime: bool,
    sample_rate: int,
) -> tuple[float | None, float, str]:
    """Returns (ttft_s, total_s, transcript)."""
    t0 = time.time()
    ttft: float | None = None
    full_text = ""
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.config", "sample_rate": sample_rate}))
        ready = json.loads(await ws.recv())
        if ready.get("type") != "session.ready":
            raise RuntimeError(f"Unexpected first event: {ready}")

        async def _send_audio() -> None:
            for off in range(0, len(pcm16), chunk_bytes):
                chunk = pcm16[off : off + chunk_bytes]
                await ws.send(
                    json.dumps(
                        {
                            "type": "audio.chunk",
                            "data": base64.b64encode(chunk).decode("ascii"),
                        }
                    )
                )
                if realtime:
                    await asyncio.sleep(chunk_ms / 1000.0)
            await ws.send(json.dumps({"type": "audio.done"}))

        async def _read_replies() -> str:
            nonlocal ttft, full_text
            while True:
                try:
                    msg = json.loads(await ws.recv())
                except websockets.ConnectionClosedOK:
                    break
                t = msg.get("type")
                if t == "transcript.update":
                    if ttft is None and msg.get("text"):
                        ttft = time.time() - t0
                    full_text = msg.get("text", "")
                elif t == "transcript.done":
                    return msg.get("text", full_text)
                elif t == "error":
                    raise RuntimeError(f"server error: {msg.get('message')}")
            return full_text

        send_task = asyncio.create_task(_send_audio())
        full = await _read_replies()
        await send_task
    return ttft, time.time() - t0, full


async def _rest_session(*, base_url: str, audio_bytes: bytes, file_name: str) -> tuple[float, str]:
    t0 = time.time()
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            f"{base_url}/v1/audio/transcriptions",
            files={"file": (file_name, audio_bytes, "audio/wav")},
            data={"model": "kyutai"},
        )
        resp.raise_for_status()
        text = resp.json().get("text", "")
    return time.time() - t0, text


async def main() -> None:
    args = parse_args()
    sample_rate = 24000
    audio, _ = load_audio(args.audio, sr=sample_rate, mono=True)
    audio_dur = len(audio) / sample_rate
    pcm16 = np.clip(audio * 32768.0, -32768, 32767).astype("<i2").tobytes()
    chunk_bytes = ((args.chunk_ms * sample_rate) // 1000) * 2

    with open(args.audio, "rb") as f:
        wav_bytes = f.read()

    ws_url = args.base_url.replace("http", "ws") + "/v1/audio/transcriptions/stream"

    print(f"audio: {audio_dur:.2f}s @ {sample_rate} Hz, {len(pcm16)} bytes pcm16")
    print()

    # --- REST (offline) ---
    print("== Whisper-style POST /v1/audio/transcriptions ==")
    durations = []
    transcript = ""
    for _ in range(args.reps):
        d, transcript = await _rest_session(
            base_url=args.base_url, audio_bytes=wav_bytes, file_name=args.audio.split("/")[-1]
        )
        durations.append(d)
    median_d = statistics.median(durations)
    print(f"  wall median:  {median_d:.2f}s   (RTF {median_d / audio_dur:.3f})")
    print(f"  transcript:   {transcript!r}")
    print()

    # --- WebSocket fast ---
    print("== WebSocket, fast upload (no pacing) ==")
    ttfts: list[float] = []
    walls: list[float] = []
    transcript = ""
    for _ in range(args.reps):
        ttft, wall, transcript = await _ws_session(
            ws_url=ws_url,
            pcm16=pcm16,
            chunk_bytes=chunk_bytes,
            chunk_ms=args.chunk_ms,
            realtime=False,
            sample_rate=sample_rate,
        )
        if ttft is not None:
            ttfts.append(ttft)
        walls.append(wall)
    median_w = statistics.median(walls)
    print(f"  ttft median:  {statistics.median(ttfts):.2f}s")
    print(f"  wall median:  {median_w:.2f}s   (RTF {median_w / audio_dur:.3f})")
    print(f"  transcript:   {transcript!r}")
    print()

    # --- WebSocket realtime ---
    print("== WebSocket, real-time pacing (1x audio rate) ==")
    ttfts = []
    walls = []
    transcript = ""
    for _ in range(args.reps):
        ttft, wall, transcript = await _ws_session(
            ws_url=ws_url,
            pcm16=pcm16,
            chunk_bytes=chunk_bytes,
            chunk_ms=args.chunk_ms,
            realtime=True,
            sample_rate=sample_rate,
        )
        if ttft is not None:
            ttfts.append(ttft)
        walls.append(wall)
    median_w = statistics.median(walls)
    print(f"  ttft median:  {statistics.median(ttfts):.2f}s   (audio={audio_dur:.2f}s; ideal=0)")
    print(f"  wall median:  {median_w:.2f}s   (RTF {median_w / audio_dur:.3f})")
    print(f"  transcript:   {transcript!r}")
    print()

    # --- Concurrent ---
    if args.concurrency > 1:
        print(f"== Concurrent x{args.concurrency} (Whisper POST in parallel) ==")
        t0 = time.time()
        tasks = [
            _rest_session(base_url=args.base_url, audio_bytes=wav_bytes, file_name=args.audio.split("/")[-1])
            for _ in range(args.concurrency)
        ]
        results = await asyncio.gather(*tasks)
        wall = time.time() - t0
        per_request_walls = [d for d, _ in results]
        median_per = statistics.median(per_request_walls)
        print(f"  total wall:        {wall:.2f}s")
        print(f"  per-request wall:  median {median_per:.2f}s (sequential would be {sum(per_request_walls):.2f}s)")
        print(
            f"  scaling efficiency: {sum(per_request_walls) / wall:.2f}x ({100 * sum(per_request_walls) / wall / args.concurrency:.0f}% of perfect)"
        )
        print(f"  all transcripts identical: {len({t for _, t in results}) == 1}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
