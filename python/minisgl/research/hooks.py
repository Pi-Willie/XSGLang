from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from minisgl.hooks import ActivationCaptureResult, HookCallback, HookContext


def capture_hook(result: ActivationCaptureResult) -> HookCallback:
    """Create a read-only hook that captures hidden states for each request/layer/step."""

    def _hook(hidden_states: torch.Tensor, ctx: HookContext) -> torch.Tensor:
        result.add(ctx.request_id, ctx.layer_idx, hidden_states)
        return hidden_states

    return _hook


def capture_all(num_layers: int, result: ActivationCaptureResult) -> Dict[int, HookCallback]:
    """Create read-only capture hooks for every transformer layer."""
    cap = capture_hook(result)
    return {layer_idx: cap for layer_idx in range(num_layers)}


def activation_steering(vector: torch.Tensor, scale: float) -> HookCallback:
    """Return a write-hook that steers the last token hidden state by `scale * vector`."""
    vector = vector.detach().to(dtype=torch.float32)
    cached_vectors: Dict[tuple[str, torch.dtype], torch.Tensor] = {}

    def _hook(hidden_states: torch.Tensor, ctx: HookContext) -> torch.Tensor:
        key = (str(hidden_states.device), hidden_states.dtype)
        steer = cached_vectors.get(key)
        if steer is None:
            steer = vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
            cached_vectors[key] = steer
        hidden_states[-1].add_(scale * steer)
        return hidden_states

    return _hook


def gaussian_noise(
    std: float,
    mean: float = 0.0,
    *,
    only_last_token: bool = False,
    seed: int | None = None,
    state_key: str = "gaussian_noise_calls",
) -> HookCallback:
    """Return a write-hook that adds Gaussian noise to hidden states."""
    std = float(std)
    mean = float(mean)
    if std < 0.0:
        raise ValueError("Gaussian noise std must be >= 0")

    generators: Dict[str, torch.Generator] = {}

    def _generator(device: torch.device) -> torch.Generator | None:
        if seed is None:
            return None
        key = str(device)
        generator = generators.get(key)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            generators[key] = generator
        return generator

    def _hook(hidden_states: torch.Tensor, ctx: HookContext) -> torch.Tensor:
        ctx.user_state[state_key] = int(ctx.user_state.get(state_key, 0)) + 1
        if std == 0.0 and mean == 0.0:
            return hidden_states

        target = hidden_states[-1:] if only_last_token else hidden_states
        generator = _generator(hidden_states.device)
        noise = torch.randn(
            target.shape,
            device=hidden_states.device,
            dtype=torch.float32,
            generator=generator,
        )
        noise.mul_(std)
        if mean != 0.0:
            noise.add_(mean)
        target.add_(noise.to(dtype=target.dtype))
        return hidden_states

    return _hook


def activation_diff(group: str, state_key: str = "activation_diff") -> HookCallback:
    """Record activations for contrastive experiments and compute B-A when both groups are present."""

    def _hook(hidden_states: torch.Tensor, ctx: HookContext) -> torch.Tensor:
        slot = ctx.user_state.setdefault(state_key, {})
        layer_slot = slot.setdefault(ctx.layer_idx, {})
        layer_slot[group] = hidden_states[-1].detach().clone()
        if "a" in layer_slot and "b" in layer_slot:
            slot.setdefault("difference", {})[ctx.layer_idx] = layer_slot["b"] - layer_slot["a"]
        return hidden_states

    return _hook


def probe_hook(
    probe: torch.nn.Module,
    state_key: str = "probe_outputs",
    detach_to_cpu: bool = False,
) -> HookCallback:
    """Run a probe on hidden states and log outputs in per-request user state."""

    def _hook(hidden_states: torch.Tensor, ctx: HookContext) -> torch.Tensor:
        with torch.no_grad():
            out = probe(hidden_states)
            if detach_to_cpu:
                out = out.detach().cpu()
            else:
                out = out.detach().clone()
        layer_map = ctx.user_state.setdefault(state_key, {})
        layer_map.setdefault(ctx.layer_idx, []).append(out)
        return hidden_states

    return _hook


def logit_lens(
    final_norm,
    lm_head,
    top_k: int = 5,
    state_key: str = "logit_lens",
    store_full_probs: bool = False,
) -> HookCallback:
    """Project intermediate hidden states through final norm + lm_head and store top-k tokens."""

    def _hook(hidden_states: torch.Tensor, ctx: HookContext) -> torch.Tensor:
        with torch.no_grad():
            if ctx.residual_slice is not None and hasattr(final_norm, "forward_unfused"):
                norm_out = final_norm.forward_unfused(hidden_states, ctx.residual_slice)
            else:
                norm_out = final_norm.forward(hidden_states)
            normed = norm_out[0] if isinstance(norm_out, tuple) else norm_out

            if getattr(lm_head, "tp_size", 1) == 1:
                module = lm_head.tied_embedding or lm_head
                logits = F.linear(normed, module.weight, lm_head.bias)
            else:
                logits = lm_head.forward(normed)

            topk = torch.topk(logits[-1], k=top_k, dim=-1)
            payload = {
                "token_ids": topk.indices.detach().cpu().tolist(),
                "scores": topk.values.detach().cpu().tolist(),
            }
            if store_full_probs:
                payload["probs"] = torch.softmax(logits[-1], dim=-1).detach().cpu()
            layer_map = ctx.user_state.setdefault(state_key, {})
            layer_map.setdefault(ctx.layer_idx, []).append(payload)
        return hidden_states

    return _hook


def merge_hooks(
    *hook_maps: Dict[int, HookCallback] | Dict[int, List[HookCallback]]
) -> Dict[int, List[HookCallback]]:
    merged: Dict[int, List[HookCallback]] = {}
    for hook_map in hook_maps:
        for layer_idx, value in hook_map.items():
            callbacks = value if isinstance(value, list) else [value]
            merged.setdefault(layer_idx, []).extend(callbacks)
    return merged
