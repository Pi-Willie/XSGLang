from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, DefaultDict, Dict, List, Tuple

import torch
from minisgl.utils import empty_like_cpu

HookCallback = Callable[[torch.Tensor, "HookContext"], torch.Tensor]
LogitProcessor = Callable[[torch.Tensor], torch.Tensor]


@dataclass(slots=True)
class HookContext:
    layer_idx: int
    request_id: int
    slice_start: int
    slice_end: int
    step: int
    token_ids_so_far: List[int]
    user_state: Dict[str, Any]
    continuation_id: int | None = None
    state_id: str | None = None
    session_id: int | None = None
    residual_slice: torch.Tensor | None = None


@dataclass
class HookSpec:
    layer_hooks: Dict[int, Tuple[HookCallback, ...]] = field(default_factory=dict)
    intra_layer_hooks: Dict[int, Tuple[HookCallback, ...]] = field(default_factory=dict)
    post_embedding_hooks: Tuple[HookCallback, ...] = ()
    pre_lm_head_hooks: Tuple[HookCallback, ...] = ()
    has_writes: bool = False
    tier: int = 1

    def __post_init__(self) -> None:
        self.layer_hooks = {k: tuple(v) for k, v in self.layer_hooks.items() if len(v) > 0}
        self.intra_layer_hooks = {
            k: tuple(v) for k, v in self.intra_layer_hooks.items() if len(v) > 0
        }
        self.post_embedding_hooks = tuple(self.post_embedding_hooks)
        self.pre_lm_head_hooks = tuple(self.pre_lm_head_hooks)

    @property
    def has_any_hook(self) -> bool:
        return bool(
            self.layer_hooks
            or self.intra_layer_hooks
            or self.post_embedding_hooks
            or self.pre_lm_head_hooks
        )


@dataclass(frozen=True, slots=True)
class HookDispatchEntry:
    req_index: int
    start: int
    end: int
    callbacks: Tuple[HookCallback, ...]


@dataclass(frozen=True, slots=True)
class HookSpecialEntry:
    req_index: int
    callbacks: Tuple[HookCallback, ...]


class ActivationCaptureResult:
    """Stores captured activations and supports deferred host materialization."""

    def __init__(self, to_cpu: bool = False):
        self.to_cpu = to_cpu
        self._data: DefaultDict[int, DefaultDict[int, List[torch.Tensor]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def add(self, request_id: int, layer_idx: int, tensor: torch.Tensor) -> None:
        captured = tensor.clone()
        if self.to_cpu:
            cpu_copy = empty_like_cpu(captured, pin_memory=True)
            cpu_copy.copy_(captured, non_blocking=True)
            self._data[request_id][layer_idx].append(cpu_copy)
            return
        self._data[request_id][layer_idx].append(captured)

    def materialize(self, to_cpu: bool | None = None) -> Dict[int, Dict[int, List[torch.Tensor]]]:
        want_cpu = self.to_cpu if to_cpu is None else to_cpu
        if want_cpu and torch.cuda.is_available():
            torch.cuda.synchronize()

        out: Dict[int, Dict[int, List[torch.Tensor]]] = {}
        for req_id, layer_map in self._data.items():
            out[req_id] = {}
            for layer_idx, tensors in layer_map.items():
                if want_cpu and not self.to_cpu:
                    out[req_id][layer_idx] = [t.detach().cpu() for t in tensors]
                else:
                    out[req_id][layer_idx] = list(tensors)
        return out


def normalize_layer_hooks(
    hooks: Dict[int, HookCallback] | Dict[int, List[HookCallback]] | None,
) -> Dict[int, Tuple[HookCallback, ...]]:
    if hooks is None:
        return {}

    normalized: Dict[int, Tuple[HookCallback, ...]] = {}
    for layer_idx, cbs in hooks.items():
        if isinstance(cbs, list):
            callbacks = tuple(cbs)
        else:
            callbacks = (cbs,)
        if callbacks:
            normalized[int(layer_idx)] = callbacks
    return normalized


def build_dispatch_table(
    reqs: List[Any],
    req_slices: List[Tuple[int, int]],
    num_layers: int,
    *,
    use_intra_layer: bool,
) -> List[Tuple[HookDispatchEntry, ...]]:
    table: List[List[HookDispatchEntry]] = [[] for _ in range(num_layers)]
    for req_idx, req in enumerate(reqs):
        spec: HookSpec | None = getattr(req, "hook_spec", None)
        if spec is None:
            continue
        hook_map = spec.intra_layer_hooks if use_intra_layer else spec.layer_hooks
        if not hook_map:
            continue
        start, end = req_slices[req_idx]
        for layer_idx, callbacks in hook_map.items():
            if 0 <= layer_idx < num_layers and callbacks:
                table[layer_idx].append(
                    HookDispatchEntry(
                        req_index=req_idx,
                        start=start,
                        end=end,
                        callbacks=callbacks,
                    )
                )
    return [tuple(entries) for entries in table]


def build_special_dispatch_table(
    reqs: List[Any],
    *,
    point: str,
) -> Tuple[HookSpecialEntry, ...]:
    if point == "post_embedding":
        attr = "post_embedding_hooks"
    elif point == "pre_lm_head":
        attr = "pre_lm_head_hooks"
    else:
        raise ValueError(f"Unknown hook point: {point}")

    entries: List[HookSpecialEntry] = []
    for req_idx, req in enumerate(reqs):
        spec: HookSpec | None = getattr(req, "hook_spec", None)
        if spec is None:
            continue
        callbacks = getattr(spec, attr)
        if callbacks:
            entries.append(HookSpecialEntry(req_index=req_idx, callbacks=callbacks))
    return tuple(entries)
