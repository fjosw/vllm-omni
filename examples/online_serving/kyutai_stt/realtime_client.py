# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI-Realtime-style client for the streaming-transcription server.

Speaks the OpenAI Realtime event names (``session.update``,
``input_audio_buffer.append``,
``conversation.item.input_audio_transcription.delta``, etc.). Useful as
a smoke test for the ``/v1/realtime`` endpoint and for exercising the
shape that production OpenAI Realtime SDKs expect.

Run from anywhere with no vllm install required:

    uv run --no-project examples/online_serving/kyutai_stt/realtime_client.py \\
        --url ws://localhost:8765/v1/realtime \\
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
    p.add_argument("--url", default="ws://localhost:8765/v1/realtime")
    p.add_argument("--audio", required=True)
    p.add_argument("--chunk-ms", type=int, default=200)
    p.add_argument("--realtime", action="store_true")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    sample_rate = 24000
    audio, sr = sf.read(args.audio, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        g = math.gcd(int(sr), sample_rate)
        audio = resample_poly(audio, sample_rate // g, int(sr) // g).astype(np.float32)
    pcm16 = np.clip(audio * 32768, -32768, 32767).astype("<i2").tobytes()
    chunk_bytes = (args.chunk_ms * sample_rate // 1000) * 2

    async with websockets.connect(args.url, max_size=None) as ws:
        first = json.loads(await ws.recv())
        if first.get("type") != "session.created":
            print(f"unexpected first event: {first}", file=sys.stderr)
            return

        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "input_audio_format": "pcm16",
                        "input_audio_transcription": {"model": "whisper-1"},
                    },
                }
            )
        )

        async def send_audio() -> None:
            for off in range(0, len(pcm16), chunk_bytes):
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(pcm16[off : off + chunk_bytes]).decode(),
                        }
                    )
                )
                if args.realtime:
                    await asyncio.sleep(args.chunk_ms / 1000)
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        async def read_replies() -> str:
            full = ""
            while True:
                try:
                    msg = json.loads(await ws.recv())
                except websockets.ConnectionClosedOK:
                    break
                t = msg.get("type")
                if t == "conversation.item.input_audio_transcription.delta":
                    delta = msg.get("delta", "")
                    full += delta
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                elif t == "conversation.item.input_audio_transcription.completed":
                    sys.stdout.write("\n")
                    return msg.get("transcript", full)
                elif t == "error":
                    print(f"\n[server error] {msg.get('error')}", file=sys.stderr)
                    return full
            return full

        t0 = time.time()
        send_task = asyncio.create_task(send_audio())
        full = await read_replies()
        await send_task
        elapsed = time.time() - t0
        audio_dur = len(audio) / sample_rate
        print(f"\nfinal: {full!r}")
        print(f"audio: {audio_dur:.2f}s, wall: {elapsed:.2f}s, RTF: {elapsed / audio_dur:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
