# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI-Realtime-style client for the streaming-transcription server.

Speaks the OpenAI Realtime event names (``session.update``,
``input_audio_buffer.append``,
``conversation.item.input_audio_transcription.delta``, etc.) and is the
correct shape for the OpenAI Python SDK's ``realtime`` client. Use this
to validate the ``/v1/realtime`` endpoint.

Run:
    python examples/online_serving/kyutai_stt/realtime_client.py \\
        --url ws://localhost:8765/v1/realtime \\
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
    parser.add_argument("--url", default="ws://localhost:8765/v1/realtime")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--realtime", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    sample_rate = 24000
    audio, _ = load_audio(args.audio, sr=sample_rate, mono=True)
    pcm16 = np.clip(audio * 32768.0, -32768, 32767).astype("<i2").tobytes()
    chunk_bytes = ((args.chunk_ms * sample_rate) // 1000) * 2

    async with websockets.connect(args.url, max_size=None) as ws:
        # The OpenAI Realtime spec sends session.created automatically once the
        # WebSocket is open; consume it before doing anything else.
        first = json.loads(await ws.recv())
        if first.get("type") != "session.created":
            print(f"Unexpected first event: {first}", file=sys.stderr)
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

        async def _send_audio() -> None:
            for off in range(0, len(pcm16), chunk_bytes):
                chunk = pcm16[off : off + chunk_bytes]
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                        }
                    )
                )
                if args.realtime:
                    await asyncio.sleep(args.chunk_ms / 1000.0)
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        async def _read_replies() -> str:
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
        send_task = asyncio.create_task(_send_audio())
        full = await _read_replies()
        await send_task
        elapsed = time.time() - t0
        audio_dur = len(audio) / sample_rate
        print(f"\nfinal: {full!r}")
        print(f"audio: {audio_dur:.2f}s, wall: {elapsed:.2f}s, RTF: {elapsed / audio_dur:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
