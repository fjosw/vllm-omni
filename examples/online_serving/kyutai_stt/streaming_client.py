# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference client for the streaming-transcription WebSocket server.

Speaks the custom protocol used by ``WS /v1/audio/transcriptions/stream``:
``transcript.update`` snapshots, ``transcript.revise`` rollbacks, and
``transcript.delta`` additive suffixes.

Run from anywhere with no vllm install required:

    uv run --no-project examples/online_serving/kyutai_stt/streaming_client.py \\
        --url ws://localhost:8765/v1/audio/transcriptions/stream \\
        --audio path/to/audio.wav --realtime
"""

# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "websockets", "soundfile", "scipy"]
# ///

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import sys
import time

import numpy as np
import soundfile as sf
import websockets
from scipy.signal import resample_poly


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--chunk-ms", type=int, default=200)
    p.add_argument(
        "--realtime",
        action="store_true",
        help="Pace chunk sends at real time (default: send as fast as possible).",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print every partial-transcript update on its own timestamped line.",
    )
    return p.parse_args()


def _load_audio_pcm16(path: str, target_sr: int = 24000) -> tuple[bytes, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        g = math.gcd(int(sr), target_sr)
        audio = resample_poly(audio, target_sr // g, int(sr) // g).astype(np.float32)
    pcm16 = np.clip(audio * 32768, -32768, 32767).astype("<i2").tobytes()
    return pcm16, len(audio)


async def main() -> None:
    args = parse_args()
    sample_rate = 24000
    pcm16, n_samples = _load_audio_pcm16(args.audio, sample_rate)
    chunk_bytes = (args.chunk_ms * sample_rate // 1000) * 2
    audio_dur = n_samples / sample_rate

    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.config", "sample_rate": sample_rate}))
        ready = json.loads(await ws.recv())
        if ready.get("type") != "session.ready":
            print(f"unexpected first event: {ready}", file=sys.stderr)
            return

        t0 = time.time()
        n_partials = 0

        async def send_audio() -> None:
            for off in range(0, len(pcm16), chunk_bytes):
                await ws.send(
                    json.dumps(
                        {
                            "type": "audio.chunk",
                            "data": base64.b64encode(pcm16[off : off + chunk_bytes]).decode(),
                        }
                    )
                )
                if args.realtime:
                    await asyncio.sleep(args.chunk_ms / 1000)
            await ws.send(json.dumps({"type": "audio.done"}))

        async def read_replies() -> str:
            nonlocal n_partials
            while True:
                try:
                    msg = json.loads(await ws.recv())
                except websockets.ConnectionClosedOK:
                    return ""
                t = msg.get("type")
                if t == "transcript.update":
                    n_partials += 1
                    text = msg.get("text", "")
                    if args.verbose:
                        sys.stdout.write(f"[partial @{time.time() - t0:6.2f}s] {text}\n")
                    else:
                        sys.stdout.write("\r\033[K" + text)
                    sys.stdout.flush()
                elif t == "transcript.done":
                    if not args.verbose:
                        sys.stdout.write("\n")
                    return msg.get("text", "")
                elif t == "error":
                    print(f"\n[server error] {msg.get('message')}", file=sys.stderr)
                    return ""

        send_task = asyncio.create_task(send_audio())
        full = await read_replies()
        await send_task
        elapsed = time.time() - t0
        if args.verbose:
            print("---")
        print(f"final: {full!r}")
        print(f"audio: {audio_dur:.2f}s, wall: {elapsed:.2f}s, RTF: {elapsed / audio_dur:.3f}, partials: {n_partials}")


if __name__ == "__main__":
    asyncio.run(main())
