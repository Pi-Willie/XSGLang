from __future__ import annotations

from .architectures import get_decoder_architecture
from .decoder import DecoderForCausalLM


class Qwen3MoeForCausalLM(DecoderForCausalLM):
    def __init__(self, config):
        super().__init__(config, get_decoder_architecture("Qwen3MoeForCausalLM"))


__all__ = ["Qwen3MoeForCausalLM"]
