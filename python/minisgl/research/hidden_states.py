from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Literal, Sequence, Tuple

import torch
from minisgl.core import HookProgram
from minisgl.hooks import HookCallback, HookSpec, normalize_layer_hooks
from minisgl.utils import empty_like_cpu


CaptureMode = Literal["last_token", "full_segment"]


@dataclass(frozen=True)
class HiddenStateRequest:
    layers: Sequence[int] | Literal["all"]
    capture: CaptureMode = "last_token"
    to_cpu: bool = True
    state_key: str | None = None


@dataclass(frozen=True)
class HiddenStateTraceItem:
    step: int
    token_count: int
    last_token_id: int | None
    tensor: torch.Tensor


class HiddenStateTrace:
    """Stores hidden-state traces keyed by continuation id and layer.

    The storage layout is intentionally simple and researcher-friendly:
    `trace.materialize()[continuation_id][layer_idx]` returns a list of per-step samples.
    """

    def __init__(self, to_cpu: bool = True):
        self.to_cpu = to_cpu
        self._data: DefaultDict[int, DefaultDict[int, List[HiddenStateTraceItem]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def add(
        self,
        continuation_id: int,
        layer_idx: int,
        *,
        step: int,
        token_ids: Sequence[int],
        tensor: torch.Tensor,
    ) -> None:
        captured = tensor.clone()
        if self.to_cpu:
            host = empty_like_cpu(captured, pin_memory=True)
            host.copy_(captured, non_blocking=True)
            captured = host
        token_list = list(token_ids)
        self._data[continuation_id][layer_idx].append(
            HiddenStateTraceItem(
                step=int(step),
                token_count=len(token_list),
                last_token_id=None if len(token_list) == 0 else int(token_list[-1]),
                tensor=captured,
            )
        )

    def materialize(self, to_cpu: bool | None = None) -> Dict[int, Dict[int, List[HiddenStateTraceItem]]]:
        want_cpu = self.to_cpu if to_cpu is None else to_cpu
        if want_cpu and torch.cuda.is_available():
            torch.cuda.synchronize()

        out: Dict[int, Dict[int, List[HiddenStateTraceItem]]] = {}
        for continuation_id, layer_map in self._data.items():
            out[continuation_id] = {}
            for layer_idx, entries in layer_map.items():
                if want_cpu and not self.to_cpu:
                    out[continuation_id][layer_idx] = [
                        HiddenStateTraceItem(
                            step=item.step,
                            token_count=item.token_count,
                            last_token_id=item.last_token_id,
                            tensor=item.tensor.detach().cpu(),
                        )
                        for item in entries
                    ]
                else:
                    out[continuation_id][layer_idx] = list(entries)
        return out

    def latest(self, continuation_id: int, layer_idx: int) -> HiddenStateTraceItem:
        return self._data[continuation_id][layer_idx][-1]

    def stack(self, continuation_id: int, layer_idx: int) -> torch.Tensor:
        entries = self._data[continuation_id][layer_idx]
        if len(entries) == 0:
            raise KeyError(
                f"No hidden-state trace found for continuation={continuation_id}, layer={layer_idx}"
            )
        return torch.stack([entry.tensor for entry in entries], dim=0)


def _resolve_layers(
    layers: Sequence[int] | Literal["all"],
    *,
    num_layers: int | None,
) -> Tuple[int, ...]:
    if layers == "all":
        if num_layers is None:
            raise ValueError("num_layers is required when layers='all'")
        return tuple(range(num_layers))
    resolved = tuple(dict.fromkeys(int(layer) for layer in layers))
    if len(resolved) == 0:
        raise ValueError("At least one layer must be requested for hidden-state tracing")
    return resolved


def _select_capture_tensor(hidden_states: torch.Tensor, capture: CaptureMode) -> torch.Tensor:
    if capture == "last_token":
        return hidden_states[-1]
    if capture == "full_segment":
        return hidden_states
    raise ValueError(f"Unsupported hidden-state capture mode: {capture}")


def hidden_state_trace_hook(
    *,
    capture: CaptureMode = "last_token",
    to_cpu: bool = True,
    trace: HiddenStateTrace | None = None,
    state_key: str | None = None,
) -> HookCallback:
    """Build a read-only hook for capturing hidden states.

    The hook can write to an external `HiddenStateTrace`, to the per-request
    `hook_user_state`, or both.
    """

    def _hook(hidden_states: torch.Tensor, ctx) -> torch.Tensor:
        selected = _select_capture_tensor(hidden_states, capture)
        continuation_id = ctx.continuation_id if ctx.continuation_id is not None else int(ctx.request_id)
        token_ids = list(ctx.token_ids_so_far)
        if trace is not None:
            trace.add(
                continuation_id,
                ctx.layer_idx,
                step=ctx.step,
                token_ids=token_ids,
                tensor=selected,
            )
        if state_key is not None:
            captured = selected.clone()
            if to_cpu:
                host = empty_like_cpu(captured, pin_memory=True)
                host.copy_(captured, non_blocking=True)
                captured = host
            layer_map = ctx.user_state.setdefault(state_key, {})
            layer_map.setdefault(ctx.layer_idx, []).append(
                {
                    "step": int(ctx.step),
                    "token_count": len(token_ids),
                    "last_token_id": None if len(token_ids) == 0 else int(token_ids[-1]),
                    "tensor": captured,
                }
            )
        return hidden_states

    return _hook


def build_hidden_state_trace_program(
    layers: Sequence[int] | Literal["all"],
    *,
    num_layers: int | None = None,
    capture: CaptureMode = "last_token",
    to_cpu: bool = True,
    state_key: str | None = None,
    label: str | None = None,
) -> tuple[HookProgram, HiddenStateTrace]:
    resolved_layers = _resolve_layers(layers, num_layers=num_layers)
    trace = HiddenStateTrace(to_cpu=to_cpu)
    callback = hidden_state_trace_hook(
        capture=capture,
        to_cpu=to_cpu,
        trace=trace,
        state_key=state_key,
    )
    hook_spec = HookSpec(
        layer_hooks=normalize_layer_hooks({layer_idx: callback for layer_idx in resolved_layers}),
        has_writes=False,
        tier=1,
    )
    return HookProgram(hook_spec=hook_spec, label=label), trace


def stack_hidden_state_trace(
    trace: HiddenStateTrace,
    continuation_id: int,
    layer_idx: int,
) -> torch.Tensor:
    return trace.stack(continuation_id, layer_idx)


__all__ = [
    "CaptureMode",
    "HiddenStateRequest",
    "HiddenStateTrace",
    "HiddenStateTraceItem",
    "build_hidden_state_trace_program",
    "hidden_state_trace_hook",
    "stack_hidden_state_trace",
]
