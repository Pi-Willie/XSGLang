from __future__ import annotations

from typing import List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.hooks import (
    HookContext,
    HookDispatchEntry,
    HookSpecialEntry,
    build_dispatch_table,
    build_special_dispatch_table,
)


def _ensure_dispatch_tables(
    batch: Batch,
    num_layers: int,
) -> Tuple[List[Tuple[HookDispatchEntry, ...]], List[Tuple[HookDispatchEntry, ...]]]:
    cached_layers = getattr(batch, "_hook_num_layers", None)
    if cached_layers != num_layers:
        batch._hook_num_layers = num_layers
        batch._hook_layer_dispatch = build_dispatch_table(
            batch.reqs,
            batch.req_slices,
            num_layers,
            use_intra_layer=False,
        )
        batch._hook_intra_layer_dispatch = build_dispatch_table(
            batch.reqs,
            batch.req_slices,
            num_layers,
            use_intra_layer=True,
        )
    return batch._hook_layer_dispatch, batch._hook_intra_layer_dispatch


def _ensure_special_dispatch_tables(
    batch: Batch,
) -> Tuple[Tuple[HookSpecialEntry, ...], Tuple[HookSpecialEntry, ...]]:
    post_dispatch = getattr(batch, "_hook_post_embedding_dispatch", None)
    pre_dispatch = getattr(batch, "_hook_pre_lm_head_dispatch", None)
    if post_dispatch is None or pre_dispatch is None:
        batch._hook_post_embedding_dispatch = build_special_dispatch_table(
            batch.reqs,
            point="post_embedding",
        )
        batch._hook_pre_lm_head_dispatch = build_special_dispatch_table(
            batch.reqs,
            point="pre_lm_head",
        )
    return batch._hook_post_embedding_dispatch, batch._hook_pre_lm_head_dispatch


def _build_or_update_context(req: Req, start: int, end: int) -> HookContext:
    ctx = req.hook_context
    if ctx is None:
        ctx = HookContext(
            layer_idx=-1,
            request_id=req.uid,
            slice_start=start,
            slice_end=end,
            step=req.generation_step,
            token_ids_so_far=req.generated_token_ids,
            user_state=req.hook_user_state,
            continuation_id=req.continuation_id,
            state_id=req.state_id,
            session_id=req.session_id,
        )
        req.hook_context = ctx
        return ctx

    ctx.layer_idx = -1
    ctx.slice_start = start
    ctx.slice_end = end
    ctx.step = req.generation_step
    ctx.token_ids_so_far = req.generated_token_ids
    ctx.user_state = req.hook_user_state
    ctx.continuation_id = req.continuation_id
    ctx.state_id = req.state_id
    ctx.session_id = req.session_id
    ctx.residual_slice = None
    return ctx


def _build_contexts(batch: Batch) -> List[HookContext | None]:
    contexts: List[HookContext | None] = []
    for req_idx, req in enumerate(batch.reqs):
        spec = req.hook_spec
        if spec is None or not spec.has_any_hook:
            contexts.append(None)
            continue
        start, end = batch.req_slices[req_idx]
        contexts.append(_build_or_update_context(req, start, end))
    return contexts


def prepare_hook_runtime(batch: Batch, num_layers: int) -> Tuple[
    List[Tuple[HookDispatchEntry, ...]],
    List[Tuple[HookDispatchEntry, ...]],
    List[HookContext | None],
]:
    if not batch.has_hooked_requests:
        return [], [], []

    layer_dispatch, intra_dispatch = _ensure_dispatch_tables(batch, num_layers)
    _ensure_special_dispatch_tables(batch)
    contexts = _build_contexts(batch)
    batch.hook_contexts = contexts
    return layer_dispatch, intra_dispatch, contexts


def dispatch_layer_entries(
    hidden_states: torch.Tensor,
    layer_idx: int,
    entries: Tuple[HookDispatchEntry, ...],
    contexts: List[HookContext | None],
    residual: torch.Tensor | None = None,
) -> torch.Tensor:
    if len(entries) == 0:
        return hidden_states

    for entry in entries:
        ctx = contexts[entry.req_index]
        if ctx is None:
            continue
        ctx.layer_idx = layer_idx
        start = entry.start
        end = entry.end
        ctx.residual_slice = None if residual is None else residual[start:end]
        segment = hidden_states[start:end]
        for callback_idx, callback in enumerate(entry.callbacks):
            try:
                out = callback(segment, ctx)
            except Exception as exc:
                raise RuntimeError(
                    f"Hook callback failed (point=layer, layer={layer_idx}, "
                    f"request_id={ctx.request_id}, callback_index={callback_idx})"
                ) from exc
            if out is not segment:
                hidden_states[start:end].copy_(out)
                segment = hidden_states[start:end]
    return hidden_states


def dispatch_special_point(
    hidden_states: torch.Tensor,
    batch: Batch,
    contexts: List[HookContext | None],
    *,
    point: str,
) -> torch.Tensor:
    if not batch.has_hooked_requests:
        return hidden_states

    if point == "post_embedding":
        dispatch = batch._hook_post_embedding_dispatch
    elif point == "pre_lm_head":
        dispatch = batch._hook_pre_lm_head_dispatch
    else:
        raise ValueError(f"Unknown hook point: {point}")

    for entry in dispatch:
        ctx = contexts[entry.req_index]
        if ctx is None:
            continue
        ctx.layer_idx = -1
        ctx.residual_slice = None
        start = ctx.slice_start
        end = ctx.slice_end
        segment = hidden_states[start:end]
        for callback_idx, callback in enumerate(entry.callbacks):
            try:
                out = callback(segment, ctx)
            except Exception as exc:
                raise RuntimeError(
                    f"Hook callback failed (point={point}, layer=-1, "
                    f"request_id={ctx.request_id}, callback_index={callback_idx})"
                ) from exc
            if out is not segment:
                hidden_states[start:end].copy_(out)
                segment = hidden_states[start:end]

    return hidden_states
