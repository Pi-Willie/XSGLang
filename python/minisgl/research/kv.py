from __future__ import annotations

from typing import Any

import torch


def _find_req(runtime: Any, request_id: int):
    if hasattr(runtime, "decode_manager"):
        for req in runtime.decode_manager.running_reqs:
            if req.uid == request_id:
                return req
    if hasattr(runtime, "prefill_manager"):
        for pending in runtime.prefill_manager.pending_list:
            chunked = pending.chunked_req
            if chunked is not None and chunked.uid == request_id:
                return chunked
    raise KeyError(f"Request {request_id} not found in runtime")


def kv_inspector(runtime: Any, request_id: int, layer_idx: int) -> dict[str, torch.Tensor]:
    """Inspect a request's KV cache tensors at a given layer using the runtime page table mapping."""
    req = _find_req(runtime, request_id)
    engine = runtime.engine if hasattr(runtime, "engine") else runtime
    k_cache = engine.ctx.kv_cache.k_cache(layer_idx)
    v_cache = engine.ctx.kv_cache.v_cache(layer_idx)
    page_indices = engine.page_table[req.table_idx, : req.device_len]

    flat_shape = (-1,) + tuple(k_cache.shape[2:])
    k_flat = k_cache.view(flat_shape)
    v_flat = v_cache.view(flat_shape)
    return {
        "k": k_flat[page_indices].clone(),
        "v": v_flat[page_indices].clone(),
    }
