from __future__ import annotations

from .architectures import get_decoder_architecture
from .decoder import DecoderForCausalLM


class LlamaForCausalLM(DecoderForCausalLM):
    def __init__(self, config):
        super().__init__(config, get_decoder_architecture("LlamaForCausalLM"))


__all__ = ["LlamaForCausalLM"]
