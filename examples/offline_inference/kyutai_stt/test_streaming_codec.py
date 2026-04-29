# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify Mimi streaming-encode equivalence.

Encodes the same waveform in (a) one shot and (b) chunks of ``frame_size``
samples, and checks that the resulting per-frame audio bias is bitwise
identical. Streaming equivalence is a prerequisite for the real-time
transcription API: any divergence here propagates directly to the LM.

Usage:
    python examples/offline_inference/kyutai_stt/test_streaming_codec.py \\
        --model /path/to/stt-1b-en_fr-trfs \\
        --audio-path /path/to/audio.wav
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoConfig, MimiModel
from vllm.multimodal.media.audio import load_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--target-sr", type=int, default=24000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=10,
        help="Number of audio frames (1 frame = frame_size samples) per streaming chunk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    codec = MimiModel.from_pretrained(args.model, subfolder="codec", dtype=torch.float32).to(args.device)
    codec.eval()
    frame_size = int(cfg.codec_config.frame_size)
    chunk_samples = args.chunk_frames * frame_size

    audio, _ = load_audio(args.audio_path, sr=args.target_sr, mono=True)
    iv = torch.from_numpy(audio).to(args.device, dtype=torch.float32).view(1, 1, -1)
    n_frames = iv.shape[-1] // frame_size
    iv = iv[..., : n_frames * frame_size]

    with torch.no_grad():
        codes_oneshot = codec.encode(iv).audio_codes  # (1, num_codebooks, T)

    streamed: list[torch.Tensor] = []
    state: dict[str, object] = {"padding_cache": None, "encoder_past": None}
    with torch.no_grad():
        for start in range(0, iv.shape[-1], chunk_samples):
            chunk = iv[..., start : start + chunk_samples]
            out = codec.encode(
                chunk,
                padding_cache=state["padding_cache"],
                encoder_past_key_values=state["encoder_past"],
                use_streaming=True,
                return_dict=True,
            )
            state["padding_cache"] = out.padding_cache
            state["encoder_past"] = out.encoder_past_key_values
            if out.audio_codes.shape[-1] > 0:
                streamed.append(out.audio_codes)

    codes_streamed = torch.cat(streamed, dim=-1)

    print(f"audio: {iv.shape[-1]} samples ({iv.shape[-1] / args.target_sr:.2f}s)")
    print(f"frame_size: {frame_size}, chunk_samples: {chunk_samples}, chunk_frames: {args.chunk_frames}")
    print(f"one-shot codes shape:  {tuple(codes_oneshot.shape)}")
    print(f"streamed codes shape:  {tuple(codes_streamed.shape)}")

    if codes_oneshot.shape != codes_streamed.shape:
        raise SystemExit(f"FAIL: shape mismatch {codes_oneshot.shape} vs {codes_streamed.shape}")

    diff = (codes_oneshot != codes_streamed).sum().item()
    total = codes_oneshot.numel()
    if diff == 0:
        print(f"PASS: streamed codes match one-shot exactly ({total} elements)")
    else:
        print(f"FAIL: {diff}/{total} codes differ ({100 * diff / total:.2f}%)")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
