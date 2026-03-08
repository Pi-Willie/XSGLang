from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

import torch

if TYPE_CHECKING:
    from minisgl.hooks import HookSpec, LogitProcessor
    from minisgl.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    hook_spec: HookSpec | None = None
    logit_processor: LogitProcessor | None = None
    hook_config: Dict[str, Any] | None = None
    hook_preset_name: str | None = None
    adapter_id: str | None = None
    requested_outputs: tuple[str, ...] = ()
    capture_output_history: bool = False
    chunked_req: ChunkedReq | None = None

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
