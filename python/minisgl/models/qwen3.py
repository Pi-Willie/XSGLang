from __future__ import annotations

from .architectures import get_decoder_architecture
from .decoder import DecoderForCausalLM


class Qwen3ForCausalLM(DecoderForCausalLM):
    def __init__(self, config):
        super().__init__(config, get_decoder_architecture("Qwen3ForCausalLM"))


__all__ = ["Qwen3ForCausalLM"]
