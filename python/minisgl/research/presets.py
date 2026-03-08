from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List

import torch
from minisgl.hooks import HookCallback, HookSpec, normalize_layer_hooks
from minisgl.utils import empty_like_cpu

from .hidden_states import hidden_state_trace_hook
from .hooks import activation_diff, activation_steering, gaussian_noise, probe_hook


@dataclass
class HookPresetRegistry:
    _presets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def list(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._presets.items()}

    def register(self, name: str, config: Dict[str, Any]) -> None:
        with self._lock:
            self._presets[name] = dict(config)

    def get(self, name: str) -> Dict[str, Any] | None:
        with self._lock:
            conf = self._presets.get(name)
            return dict(conf) if conf is not None else None


_GLOBAL_PRESET_REGISTRY = HookPresetRegistry()


def get_hook_preset_registry() -> HookPresetRegistry:
    return _GLOBAL_PRESET_REGISTRY


def _load_steering_vector(cfg: Dict[str, Any]) -> torch.Tensor:
    if "vector" in cfg:
        return torch.tensor(cfg["vector"], dtype=torch.float32)
    path = cfg.get("vector_path")
    if path is None:
        raise ValueError("Steering preset requires `vector` or `vector_path`")
    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, torch.Tensor):
        return loaded.to(torch.float32)
    raise ValueError(f"Unsupported steering vector payload at {path}: {type(loaded)}")


def _capture_callback(*, to_cpu: bool) -> HookCallback:
    def _hook(hidden_states, ctx):
        captured = hidden_states.clone()
        if to_cpu:
            host = empty_like_cpu(captured, pin_memory=True)
            host.copy_(captured, non_blocking=True)
            captured = host
        layer_map = ctx.user_state.setdefault("captures", {})
        layer_map.setdefault(ctx.layer_idx, []).append(captured)
        return hidden_states

    return _hook


def build_hook_spec_from_config(config: Dict[str, Any]) -> HookSpec:
    """Build a HookSpec from a JSON-serializable hook preset config."""
    hooks_cfg: List[Dict[str, Any]] = list(config.get("hooks", []))
    if "type" in config:  # allow single-hook shorthand
        hooks_cfg = [config]

    layer_hooks: Dict[int, List[HookCallback]] = {}
    intra_layer_hooks: Dict[int, List[HookCallback]] = {}
    has_writes = bool(config.get("has_writes", False))
    tier = int(config.get("tier", 1))

    for hook_cfg in hooks_cfg:
        hook_type = hook_cfg["type"]
        layers = hook_cfg.get("layers")
        if layers is None:
            layer = hook_cfg.get("layer")
            if layer is None:
                raise ValueError(f"Hook config missing `layer` or `layers`: {hook_cfg}")
            layers = [layer]

        if hook_type == "capture":
            cb = _capture_callback(to_cpu=bool(hook_cfg.get("to_cpu", False)))
            for layer_idx in layers:
                layer_hooks.setdefault(int(layer_idx), []).append(cb)

        elif hook_type == "steer":
            vector = _load_steering_vector(hook_cfg)
            scale = float(hook_cfg.get("scale", 1.0))
            cb = activation_steering(vector, scale)
            for layer_idx in layers:
                target = intra_layer_hooks if bool(hook_cfg.get("intra_layer", False)) else layer_hooks
                target.setdefault(int(layer_idx), []).append(cb)
            has_writes = True
            tier = max(tier, 2 if bool(hook_cfg.get("intra_layer", False)) else 1)

        elif hook_type == "gaussian_noise":
            cb = gaussian_noise(
                std=float(hook_cfg.get("std", 0.0)),
                mean=float(hook_cfg.get("mean", 0.0)),
                only_last_token=bool(hook_cfg.get("only_last_token", False)),
                seed=hook_cfg.get("seed"),
                state_key=str(hook_cfg.get("state_key", "gaussian_noise_calls")),
            )
            for layer_idx in layers:
                target = intra_layer_hooks if bool(hook_cfg.get("intra_layer", False)) else layer_hooks
                target.setdefault(int(layer_idx), []).append(cb)
            has_writes = True
            tier = max(tier, 2 if bool(hook_cfg.get("intra_layer", False)) else 1)

        elif hook_type == "probe":
            probe_path = hook_cfg.get("probe_path")
            if probe_path is None:
                raise ValueError("Probe preset requires `probe_path`")
            probe = torch.load(probe_path, map_location="cpu")
            if not isinstance(probe, torch.nn.Module):
                raise ValueError(f"Probe at {probe_path} is not an nn.Module")
            cb = probe_hook(probe=probe, detach_to_cpu=bool(hook_cfg.get("to_cpu", True)))
            for layer_idx in layers:
                layer_hooks.setdefault(int(layer_idx), []).append(cb)

        elif hook_type == "hidden_state_trace":
            cb = hidden_state_trace_hook(
                capture=str(hook_cfg.get("capture", "last_token")),
                to_cpu=bool(hook_cfg.get("to_cpu", True)),
                state_key=str(hook_cfg.get("state_key", "hidden_state_trace")),
            )
            for layer_idx in layers:
                layer_hooks.setdefault(int(layer_idx), []).append(cb)

        elif hook_type == "activation_diff":
            group = str(hook_cfg.get("group", "a"))
            cb = activation_diff(group=group)
            for layer_idx in layers:
                layer_hooks.setdefault(int(layer_idx), []).append(cb)

        else:
            raise ValueError(f"Unsupported hook preset type: {hook_type}")

    return HookSpec(
        layer_hooks=normalize_layer_hooks(layer_hooks),
        intra_layer_hooks=normalize_layer_hooks(intra_layer_hooks),
        has_writes=has_writes,
        tier=tier,
    )
