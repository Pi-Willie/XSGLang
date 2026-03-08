from __future__ import annotations

from .architectures import get_decoder_architecture
from .decoder import DecoderForCausalLM


class GemmaForCausalLM(DecoderForCausalLM):
    def __init__(self, config):
        super().__init__(config, get_decoder_architecture("GemmaForCausalLM"))


__all__ = ["GemmaForCausalLM"]
