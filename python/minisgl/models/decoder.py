from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .architectures import DecoderArchitectureSpec
from .base import BaseLLMModel
from .hook_utils import dispatch_layer_entries, dispatch_special_point, prepare_hook_runtime
from .utils import GatedMLP, MoEMLP, RopeAttn

if TYPE_CHECKING:
    from minisgl.hooks import HookContext, HookDispatchEntry

    from .config import ModelConfig


class DecoderLayer(BaseOP):
    def __init__(
        self,
        config: "ModelConfig",
        layer_id: int,
        *,
        architecture: DecoderArchitectureSpec,
    ) -> None:
        self.self_attn = RopeAttn(
            config,
            layer_id,
            has_attn_bias=architecture.attention_bias,
            has_qk_norm=architecture.qk_norm,
        )
        if architecture.is_moe:
            self.mlp = MoEMLP(config)
        else:
            self.mlp = GatedMLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        intra_entries: Tuple["HookDispatchEntry", ...] = (),
        contexts: List["HookContext | None"] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        if contexts is not None and len(intra_entries) > 0:
            x = dispatch_layer_entries(x, self._layer_id, intra_entries, contexts)
        x = self.mlp.forward(x)
        return x, residual


class DecoderModel(BaseOP):
    def __init__(self, config: "ModelConfig", architecture: DecoderArchitectureSpec) -> None:
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [
                DecoderLayer(
                    config,
                    layer_id,
                    architecture=architecture,
                )
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.architecture = architecture

    @property
    def num_layers(self) -> int:
        return len(self.layers.op_list)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]

    def forward_with_hooks(self, input_ids: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        layer_dispatch, intra_dispatch, contexts = prepare_hook_runtime(
            batch,
            num_layers=self.num_layers,
        )

        x = self.embed_tokens.forward(input_ids)
        x = dispatch_special_point(x, batch, contexts, point="post_embedding")

        residual: torch.Tensor | None = None
        for layer_idx, layer in enumerate(self.layers.op_list):
            intra_entries = intra_dispatch[layer_idx] if intra_dispatch else ()
            if len(intra_entries) > 0:
                x, residual = layer.forward(
                    x,
                    residual,
                    intra_entries=intra_entries,
                    contexts=contexts,
                )
            else:
                x, residual = layer.forward(x, residual)

            entries = layer_dispatch[layer_idx] if layer_dispatch else ()
            if len(entries) > 0:
                x = dispatch_layer_entries(x, layer_idx, entries, contexts, residual=residual)

        x = self.norm.forward(x, residual)[0]
        return dispatch_special_point(x, batch, contexts, point="pre_lm_head")


class DecoderForCausalLM(BaseLLMModel):
    def __init__(self, config: "ModelConfig", architecture: DecoderArchitectureSpec) -> None:
        self.model = DecoderModel(config, architecture)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        self.architecture = architecture
        super().__init__()

    @property
    def num_layers(self) -> int:
        return self.model.num_layers

    def forward(self):
        hidden_states = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(hidden_states)
        return self._build_forward_output(hidden_states=hidden_states, logits=logits)

    def forward_with_hooks(self):
        hidden_states = self.model.forward_with_hooks(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(hidden_states)
        return self._build_forward_output(hidden_states=hidden_states, logits=logits)


__all__ = ["DecoderForCausalLM", "DecoderLayer", "DecoderModel"]
