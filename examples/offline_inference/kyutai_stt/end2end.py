# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline transcription with Kyutai Speech-to-Text via vllm-omni.

Usage:
    python examples/offline_inference/kyutai_stt/end2end.py \\
        --model kyutai/stt-1b-en_fr-trfs \\
        --audio-path /path/to/audio.wav
"""

from __future__ import annotations

import argparse

from vllm import SamplingParams
from vllm.multimodal.media.audio import load_audio

from vllm_omni.entrypoints.omni import Omni


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="kyutai/stt-1b-en_fr-trfs")
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/kyutai_speech_to_text.yaml")
    parser.add_argument("--target-sr", type=int, default=24000)
    parser.add_argument("--max-tokens", type=int, default=374)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio, _ = load_audio(args.audio_path, sr=args.target_sr, mono=True)

    omni = Omni(model=args.model, deploy_config=args.deploy_config)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=42,
        detokenize=True,
    )
    outputs = omni.generate(
        [{"prompt": "", "multi_modal_data": {"audio": [audio]}}],
        [sampling_params],
    )
    for stage_outputs in outputs:
        if stage_outputs.final_output_type == "text":
            request_output = stage_outputs.request_output
            print(f"[{request_output.request_id}] {request_output.outputs[0].text!r}")


if __name__ == "__main__":
    main()
