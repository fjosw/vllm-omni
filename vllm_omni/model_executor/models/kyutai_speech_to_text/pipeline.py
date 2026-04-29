# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kyutai Speech-to-Text pipeline: a single autoregressive LLM stage."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

KYUTAI_SPEECH_TO_TEXT_PIPELINE = PipelineConfig(
    model_type="kyutai_speech_to_text",
    model_arch="KyutaiSpeechToTextForConditionalGeneration",
    hf_architectures=("KyutaiSpeechToTextForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="stt",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
