# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference client for the streaming-transcription WebSocket server.

Streams a wav file in fixed-size chunks at real-time pace, prints the
transcript deltas as they arrive, and reports total wall time.

Run:
    python examples/online_serving/kyutai_stt/streaming_client.py \\
        --url ws://localhost:8765/v1/audio/transcriptions/stream \\
        --audio /path/to/audio.wav
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time

import numpy as np
import websockets
from vllm.multimodal.media.audio import load_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://localhost:8765/v1/audio/transcriptions/stream")
    parser.add_argument("--audio", required=True, help="Path to a wav file to stream.")
    parser.add_argument(
        "--chunk-ms", type=int, default=80, help="Audio chunk duration in ms (default 80 = 1 Mimi frame)."
    )
    parser.add_argument(
        "--realtime", action="store_true", help="Pace chunk sends at real-time (default: send as fast as possible)."
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    sample_rate = 24000
    audio, _ = load_audio(args.audio, sr=sample_rate, mono=True)
    pcm16 = np.clip(audio * 32768.0, -32768, 32767).astype("<i2").tobytes()
    chunk_samples = (args.chunk_ms * sample_rate) // 1000
    chunk_bytes = chunk_samples * 2  # 2 bytes per int16

    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.config", "sample_rate": sample_rate}))
        ready = json.loads(await ws.recv())
        if ready.get("type") != "session.ready":
            print(f"Unexpected first message: {ready}", file=sys.stderr)
            return

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
                if args.realtime:
                    await asyncio.sleep(args.chunk_ms / 1000.0)
            await ws.send(json.dumps({"type": "audio.done"}))

        async def _read_replies() -> str:
            full = ""
            # Render with carriage return so the visible line is always the
            # latest hypothesis (no growing scrollback when the LM revises).
            while True:
                try:
                    msg = json.loads(await ws.recv())
                except websockets.ConnectionClosedOK:
                    break
                t = msg.get("type")
                if t == "transcript.update":
                    full = msg.get("text", "")
                    sys.stdout.write("\r\033[K" + full)
                    sys.stdout.flush()
                elif t == "transcript.delta":
                    # Already rendered via transcript.update; ignore.
                    pass
                elif t == "transcript.revise":
                    # Already rendered via transcript.update; ignore.
                    pass
                elif t == "transcript.done":
                    sys.stdout.write("\n")
                    return msg.get("text", full)
                elif t == "error":
                    print(f"\n[server error] {msg.get('message')}", file=sys.stderr)
                    return full
            return full

        t0 = time.time()
        send_task = asyncio.create_task(_send_audio())
        full = await _read_replies()
        await send_task
        elapsed = time.time() - t0
        audio_dur = len(audio) / sample_rate
        print(f"\nfinal: {full!r}")
        print(f"audio: {audio_dur:.2f}s, wall: {elapsed:.2f}s, RTF: {elapsed / audio_dur:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
