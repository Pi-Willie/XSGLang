from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from transformers import PretrainedConfig
else:
    PretrainedConfig = Any


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: List[str]
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: int | None = None

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0 or "moe" in self.model_type

    @staticmethod
    def _normalize_hidden_act(config: PretrainedConfig) -> str:
        hidden_activation = getattr(config, "hidden_activation", None)
        hidden_act = hidden_activation or getattr(config, "hidden_act", "silu")
        return str(hidden_act).lower()

    @staticmethod
    def _normalize_architectures(config: PretrainedConfig) -> List[str]:
        architectures = list(getattr(config, "architectures", []) or [])
        if architectures:
            return architectures

        model_type = str(getattr(config, "model_type", "llama"))
        fallback = {
            "llama": "LlamaForCausalLM",
            "qwen2": "Qwen2ForCausalLM",
            "qwen3": "Qwen3ForCausalLM",
            "qwen3_moe": "Qwen3MoeForCausalLM",
            "mistral": "MistralForCausalLM",
            "ministral": "MinistralForCausalLM",
            "gemma": "GemmaForCausalLM",
        }
        return [fallback.get(model_type, "LlamaForCausalLM")]

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        num_qo_heads = int(config.num_attention_heads)
        num_kv_heads = int(getattr(config, "num_key_value_heads", num_qo_heads))
        head_dim = int(getattr(config, "head_dim", config.hidden_size // num_qo_heads))
        tie_word_embeddings = bool(getattr(config, "tie_word_embeddings", False))
        model_type = str(getattr(config, "model_type", "llama"))
        num_experts = int(getattr(config, "num_local_experts", getattr(config, "num_experts", 0)))
        num_experts_per_tok = int(getattr(config, "num_experts_per_tok", 0))
        moe_intermediate_size = int(getattr(config, "moe_intermediate_size", 0))
        norm_topk_prob = bool(getattr(config, "norm_topk_prob", False))
        architectures = cls._normalize_architectures(config)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is None:
            rope_scaling = getattr(config, "rope_parameters", None)

        return cls(
            num_layers=int(config.num_hidden_layers),
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=int(config.hidden_size),
            vocab_size=int(config.vocab_size),
            intermediate_size=int(config.intermediate_size),
            hidden_act=cls._normalize_hidden_act(config),
            rms_norm_eps=float(getattr(config, "rms_norm_eps", 1e-5)),
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=int(getattr(config, "max_position_embeddings", 8192)),
                base=float(getattr(config, "rope_theta", 10000.0)),
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
            attention_bias=bool(getattr(config, "attention_bias", False)),
            attention_dropout=float(getattr(config, "attention_dropout", 0.0)),
            sliding_window=getattr(config, "sliding_window", None),
        )
