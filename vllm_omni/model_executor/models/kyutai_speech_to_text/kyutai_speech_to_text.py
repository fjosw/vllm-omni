# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kyutai Speech-to-Text (kyutai/stt-1b-en_fr, kyutai/stt-2.6b-en).

LLaMA-style decoder over a multi-stream input: at every step the model sees
a text token plus N audio codebook tokens (Mimi codec, 12.5 Hz, 32 codebooks)
embedded through a single shared table and summed into one additive bias.

The audio bias is applied at every prefill + decode step via vllm-omni's
:class:`CustomProcessMixin` ``preprocess`` hook (vLLM mainline's multimodal
merge only fires at prompt placeholder positions and is gone during decode).
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import nn
from transformers import BatchFeature, MimiModel
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.inputs import TokensPrompt
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
)
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin

if TYPE_CHECKING:
    from vllm.config import ModelConfig, SpeechToTextConfig
    from vllm.config.multimodal import BaseDummyOptions
    from vllm.config.speech_to_text import SpeechToTextParams
    from vllm.inputs import MultiModalDataDict, PromptType


class KyutaiSpeechToTextEmbeddings(VocabParallelEmbedding):
    """Embedding table covering text vocab + per-codebook ranges + a pad row.

    :meth:`embed_audio_only` shifts codebook ids into their per-codebook rows
    and returns the sum across the codebook dim — the additive audio bias.
    """

    def __init__(
        self,
        config,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        text_vocab = int(getattr(config, "text_vocab_size", config.vocab_size))
        super().__init__(
            num_embeddings=text_vocab + config.num_codebooks * config.codebook_vocab_size + 1,
            embedding_dim=config.hidden_size,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.num_codebooks = config.num_codebooks
        self.audio_pad_token_id = config.audio_pad_token_id

        codebook_offsets = torch.arange(config.num_codebooks) * config.codebook_vocab_size + text_vocab
        self.register_buffer("codebook_offsets", codebook_offsets, persistent=False)

    def embed_audio_only(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """``(..., num_codebooks)`` codes -> ``(..., hidden_size)`` summed bias."""
        if audio_codes.shape[-1] != self.num_codebooks:
            raise ValueError(f"Expected last dim {self.num_codebooks}; got {tuple(audio_codes.shape)}")
        shifted = torch.where(
            audio_codes == self.audio_pad_token_id,
            audio_codes,
            audio_codes + self.codebook_offsets,
        )
        return super().forward(shifted).sum(dim=-2)


def _bridge_kyutai_config(config) -> None:
    """Synthesize the LLaMA-shaped fields the HF Kyutai config doesn't expose."""
    if not hasattr(config, "intermediate_size") or config.intermediate_size is None:
        ffn_dim = getattr(config, "ffn_dim", None)
        if ffn_dim is not None:
            if ffn_dim % 2 != 0:
                raise ValueError(f"ffn_dim={ffn_dim} must be even (gate and up halves)")
            config.intermediate_size = ffn_dim // 2

    if getattr(config, "sliding_window", None) is not None and not getattr(config, "layer_types", None):
        config.layer_types = ["sliding_attention"] * config.num_hidden_layers

    if not getattr(config, "rope_parameters", None):
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is not None:
            config.rope_parameters = {"rope_type": "default", "rope_theta": float(rope_theta)}


def _resolve_codec_rates(config) -> tuple[int, float]:
    codec = getattr(config, "codec_config", None)
    sample_rate = getattr(codec, "sampling_rate", None) or 24_000
    frame_size = getattr(codec, "frame_size", None)
    if frame_size:
        frame_rate = sample_rate / frame_size
    else:
        frame_rate = getattr(codec, "frame_rate", None) or 12.5
    return int(sample_rate), float(frame_rate)


class KyutaiSpeechToTextModel(LlamaModel):
    """LlamaModel with the multi-stream audio-aware embedding table.

    Bridges the HF Kyutai config to the LLaMA-shaped fields vLLM's path
    expects (``intermediate_size``, ``layer_types``, ``rope_parameters``)
    and swaps the standard ``VocabParallelEmbedding`` for one that also
    knows how to look up audio-codebook ranges.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _bridge_kyutai_config(vllm_config.model_config.hf_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if not isinstance(self.embed_tokens, PPMissingLayer):
            self.embed_tokens = KyutaiSpeechToTextEmbeddings(
                vllm_config.model_config.hf_config,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "embed_tokens"),
            )


class KyutaiSttProcessingInfo(BaseProcessingInfo):
    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(**kwargs)

    def get_feature_extractor(self, **kwargs: object):
        return self.get_hf_processor(**kwargs).feature_extractor

    def get_data_parser(self) -> MultiModalDataParser:
        fe = self.get_feature_extractor()
        return MultiModalDataParser(target_sr=fe.sampling_rate, target_channels=1)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int] | None = None,
    ) -> Mapping[str, int]:
        if not mm_counts or mm_counts.get("audio", 0) <= 0:
            return {}
        cfg = self.ctx.model_config.hf_config
        max_frames = max(1, int(cfg.max_position_embeddings) - 1)
        return {"audio": max_frames}


class KyutaiSttDummyInputsBuilder(BaseDummyInputsBuilder[KyutaiSttProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, "BaseDummyOptions"],
    ) -> "MultiModalDataDict":
        n_audios = mm_counts.get("audio", 0)
        if n_audios <= 0:
            return {}
        fe = self.info.get_feature_extractor()
        return {
            "audio": self._get_dummy_audios(
                length=30 * fe.sampling_rate,
                num_audios=n_audios,
                overrides=mm_options.get("audio"),
            )
        }


class KyutaiSttMultiModalProcessor(BaseMultiModalProcessor[KyutaiSttProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        audios = mm_data.get("audios") or mm_data.get("audio") or []
        tok = self.info.get_tokenizer()
        cfg = self.info.ctx.model_config.hf_config
        prompt_ids = [int(cfg.bos_token_id)] + (tok.encode(prompt) if prompt else [])

        if not audios:
            return BatchFeature({"input_ids": [prompt_ids]}, tensor_type="pt")

        fe = self.info.get_feature_extractor()
        out = fe(list(audios), sampling_rate=fe.sampling_rate, return_tensors="pt", padding=True)
        return BatchFeature(
            {"input_values": out["input_values"], "input_ids": [prompt_ids]},
            tensor_type="pt",
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {"input_values": MultiModalFieldConfig.batched("audio")}

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        # vLLM only attaches ``mm_features`` to the request when at least one
        # placeholder slot is registered for the modality, otherwise the audio
        # kwargs never reach the preprocess hook. Reserve one in-vocab pad
        # token per audio item; the standard merge stays a no-op (see
        # ``embed_multimodal``) and ``kyutai_preprocess`` adds the per-frame
        # bias on top of every position.
        pad_id = int(self.info.ctx.model_config.hf_config.pad_token_id)
        placeholder = [pad_id]
        return [
            PromptInsertion(
                modality="audio",
                target=PromptIndexTargets.end(),
                insertion=lambda item_idx: placeholder,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    KyutaiSttMultiModalProcessor,
    info=KyutaiSttProcessingInfo,
    dummy_inputs=KyutaiSttDummyInputsBuilder,
)
class KyutaiSpeechToTextForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
    CustomProcessMixin,
):
    requires_raw_input_tokens: ClassVar[bool] = True

    packed_modules_mapping: ClassVar[Mapping[str, list[str]]] = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    supported_languages: ClassVar[Mapping[str, str]] = {
        "en": "English",
        "fr": "French",
    }

    hf_to_vllm_mapper: ClassVar[WeightsMapper] = WeightsMapper(
        orig_to_new_substr={
            ".q_proj.linear.weight": ".q_proj.weight",
            ".k_proj.linear.weight": ".k_proj.weight",
            ".v_proj.linear.weight": ".v_proj.weight",
            ".o_proj.linear.weight": ".o_proj.weight",
            ".mlp.fc2.weight": ".mlp.down_proj.weight",
            "embed_tokens.embed_tokens.weight": "embed_tokens.weight",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        self.model = KyutaiSpeechToTextModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.codec_model = MimiModel(config.codec_config)

        text_vocab = int(getattr(config, "text_vocab_size", config.vocab_size))
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                text_vocab,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if getattr(config, "tie_word_embeddings", False):
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
            self.logits_processor = LogitsProcessor(text_vocab)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        self.has_preprocess = True
        self.set_custom_preprocess(self.kyutai_preprocess)

    def get_language_model(self) -> nn.Module:
        return self.model

    def embed_multimodal(self, **kwargs: object) -> list[torch.Tensor]:
        """Return one zero-tensor per audio item.

        ``_get_prompt_updates`` reserves a single pad token per item so vLLM
        attaches ``mm_features`` to the request; the actual audio bias is
        injected by :meth:`kyutai_preprocess`, so the standard merge is a no-op.
        """
        input_values = kwargs.get("input_values")
        if input_values is None:
            return []
        param = next(self.parameters())
        return [
            torch.zeros(1, self.config.hidden_size, device=param.device, dtype=param.dtype)
            for _ in range(len(input_values))
        ]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Kyutai's HF checkpoint stores the gate+up MLP halves as a single
        # merged ``mlp.fc1.weight``; pre-split it into ``gate_proj`` /
        # ``up_proj`` so upstream's stacked-params mapping packs them into
        # ``gate_up_proj`` like any other Llama checkpoint.
        codec_w: dict[str, torch.Tensor] = {}
        backbone: list[tuple[str, torch.Tensor]] = []
        for name, w in weights:
            if name.startswith("codec_model."):
                codec_w[name[len("codec_model.") :]] = w
            elif name.endswith(".mlp.fc1.weight"):
                base = name[: -len(".fc1.weight")]
                gate, up = w.chunk(2, dim=0)
                backbone.append((f"{base}.gate_proj.weight", gate))
                backbone.append((f"{base}.up_proj.weight", up))
            else:
                backbone.append((name, w))

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if getattr(self.config, "tie_word_embeddings", False) else None),
            ignore_unexpected_prefixes=["codec_model."],
        )
        loaded = loader.load_weights(backbone, mapper=self.hf_to_vllm_mapper)

        if codec_w:
            self.codec_model.load_state_dict(codec_w, strict=False)
            for n in self.codec_model.state_dict().keys():
                loaded.add(f"codec_model.{n}")
        return loaded

    @torch.no_grad()
    def _compute_audio_bias_full(
        self,
        input_values: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Run the Mimi codec on the padded waveform and pre-compute the
        additive bias for every generation position.

        Output shape ``(num_audio_positions, hidden_size)``: index 0 holds
        the audio-BOS bias; indices 1..N hold the per-frame biases.
        """
        # Match the codec's parameter dtype (its conv biases reject
        # mixed-dtype inputs); the LM-side add is done in the codec dtype
        # to preserve precision and the runner casts back to its bf16
        # buffer at write-time.
        codec_dtype = next(self.codec_model.parameters()).dtype
        iv = input_values.to(device=device, dtype=codec_dtype)
        if iv.dim() == 2:
            iv = iv.unsqueeze(0)
        codes = self.codec_model.encode(iv).audio_codes  # (1, num_codebooks, T)
        codes = codes[0].transpose(0, 1).contiguous()  # (T, num_codebooks)

        embed = self.model.embed_tokens
        per_frame = embed.embed_audio_only(codes)
        bos_codes = torch.full(
            (1, embed.num_codebooks),
            int(self.config.audio_bos_token_id),
            dtype=codes.dtype,
            device=codes.device,
        )
        return torch.cat([embed.embed_audio_only(bos_codes), per_frame], dim=0)

    def kyutai_preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info: object,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Add the per-position audio bias to ``input_embeds`` for the
        currently scheduled tokens of one request.

        The absolute next-position counter (``_kyutai_next_pos``) and the
        cached bias (``_kyutai_audio_bias_full``) are persisted across
        prefill + decode steps in the model_intermediate_buffer.
        """
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        update: dict[str, object] = {}
        bias_full = info.get("_kyutai_audio_bias_full")
        if bias_full is None:
            input_values = self._extract_input_values(info)
            if input_values is None:
                return input_ids, input_embeds, update
            bias_full = self._compute_audio_bias_full(input_values, device=input_embeds.device)
            update["_kyutai_audio_bias_full"] = bias_full
        else:
            # The buffer mechanism moves Tensor values to CPU between steps;
            # restore the bias to the embed device before the GPU add.
            bias_full = bias_full.to(device=input_embeds.device, non_blocking=True)

        next_pos = int(info.get("_kyutai_next_pos", 0))
        span = input_embeds.shape[0]
        end = next_pos + span
        max_pos = bias_full.shape[0]
        if end > max_pos:
            slice_end = min(end, max_pos)
            slice_len = slice_end - next_pos
            if slice_len > 0:
                input_embeds = input_embeds.clone()
                input_embeds[:slice_len] = input_embeds[:slice_len] + bias_full[next_pos:slice_end]
        else:
            input_embeds = input_embeds + bias_full[next_pos:end]
        update["_kyutai_next_pos"] = end

        return input_ids, input_embeds, update

    @staticmethod
    def _extract_input_values(info: Mapping[str, object]) -> torch.Tensor | None:
        """Pull the padded waveform out of the per-request info bundle."""
        iv = info.get("input_values")
        if iv is not None:
            return iv if isinstance(iv, torch.Tensor) else torch.as_tensor(iv)
        for feat in info.get("mm_features") or ():
            item = getattr(feat, "data", None)
            if item is not None and "input_values" in item:
                return item.get_data()["input_values"]
        return None

    @classmethod
    def get_speech_to_text_config(cls, model_config: "ModelConfig", task_type: str) -> "SpeechToTextConfig":
        from vllm.config import SpeechToTextConfig

        sr, _ = _resolve_codec_rates(model_config.hf_config)
        return SpeechToTextConfig(sample_rate=sr, max_audio_clip_s=None, min_energy_split_window_size=None)

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: "SpeechToTextConfig",
        model_config: "ModelConfig",
    ) -> int | None:
        _, fr = _resolve_codec_rates(model_config.hf_config)
        return max(1, int(round(audio_duration_s * fr)))

    @classmethod
    def get_generation_prompt(cls, stt_params: "SpeechToTextParams") -> "PromptType":
        cfg = stt_params.model_config.hf_config
        return TokensPrompt(
            prompt_token_ids=[int(cfg.bos_token_id)],
            multi_modal_data={"audio": [(stt_params.audio, int(stt_params.stt_config.sample_rate))]},
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        return None


__all__ = [
    "KyutaiSpeechToTextEmbeddings",
    "KyutaiSpeechToTextModel",
    "KyutaiSpeechToTextForConditionalGeneration",
]
