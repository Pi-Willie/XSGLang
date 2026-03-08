from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg:
    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    uid: int
    next_token: int
    finished: bool


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid: int
    text: str | List[Dict[str, str]]
    sampling_params: SamplingParams
    hook_config: Dict[str, Any] | None = None
    hook_preset_name: str | None = None
    adapter_id: str | None = None
    requested_outputs: tuple[str, ...] = ()
    capture_output_history: bool = False


@dataclass
class AdapterControlMsg(BaseTokenizerMsg):
    uid: int
    action: str
    adapter_path: str | None = None
    force: bool = False


@dataclass
class AdapterResultMsg(BaseTokenizerMsg):
    uid: int
    ok: bool
    active_adapter: str | None
    message: str
    active_request_count: int = 0


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int
