from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from minisgl.core import DEFAULT_ENGINE_CAP_MASK
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config, resolve_model_paths

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 256
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    distributed_init_addr: str | None = None
    lora_path: str | None = None
    output_head_path: str | None = None
    engine_cap_mask: int = DEFAULT_ENGINE_CAP_MASK

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.resolved_model_path)

    @cached_property
    def resolved_paths(self):
        return resolve_model_paths(self.model_path, self.lora_path)

    @property
    def resolved_model_path(self) -> str:
        return self.resolved_paths.model_path

    @property
    def resolved_lora_path(self) -> str | None:
        return self.resolved_paths.lora_path

    @property
    def tokenizer_path(self) -> str:
        return self.resolved_paths.tokenizer_path

    @cached_property
    def model_config(self) -> ModelConfig:
        from minisgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        if self.distributed_init_addr is not None:
            return self.distributed_init_addr
        return "tcp://127.0.0.1:2333"
