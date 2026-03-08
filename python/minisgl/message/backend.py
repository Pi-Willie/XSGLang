from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

import torch
from minisgl.core import SamplingParams

if TYPE_CHECKING:
    from minisgl.hooks import HookSpec, LogitProcessor

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams
    hook_spec: HookSpec | None = None
    logit_processor: LogitProcessor | None = None
    hook_config: Dict[str, Any] | None = None
    hook_preset_name: str | None = None
    adapter_id: str | None = None
    requested_outputs: tuple[str, ...] = ()
    capture_output_history: bool = False


@dataclass
class AdapterControlMsg(BaseBackendMsg):
    uid: int
    action: str
    adapter_path: str | None = None
    force: bool = False


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    uid: int
