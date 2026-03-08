from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
from minisgl.core import Batch, get_global_ctx
from minisgl.layers import BaseOP, LinearReplicated, StateLessOP
from minisgl.utils import init_logger

logger = init_logger(__name__)

_OUTPUT_ALLOW_PATTERNS = [
    "minisgl_outputs.json",
    "*_head_config.json",
    "*_head.safetensors",
]


@dataclass(frozen=True)
class SampleOutputSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass
class ModelForwardOutput:
    logits: torch.Tensor
    sample_outputs: Dict[str, torch.Tensor] = field(default_factory=dict)


class _LayerNormHeadOp(BaseOP):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, eps: float) -> None:
        self.weight = weight
        self.bias = bias
        self.eps = eps
        self._normalized_shape = tuple(weight.shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            normalized_shape=self._normalized_shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )


class _SiLUHeadOp(StateLessOP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


class _GELUHeadOp(StateLessOP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class _TanhHeadOp(StateLessOP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


class _IdentityHeadOp(StateLessOP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SequentialHiddenStateHead:
    def __init__(
        self,
        *,
        name: str,
        ops: Sequence[BaseOP],
        output_shape: tuple[int, ...],
        dtype: torch.dtype,
        squeeze_last_dim: bool,
        source_path: str,
    ) -> None:
        self.name = name
        self.ops = tuple(ops)
        self.output_shape = output_shape
        self.dtype = dtype
        self.squeeze_last_dim = squeeze_last_dim
        self.source_path = source_path

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        for op in self.ops:
            x = op.forward(x)
        if self.squeeze_last_dim and x.dim() > 1 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        return x


def _resolve_output_artifact_dir(path: str) -> str | None:
    if not path:
        return None
    if os.path.isdir(path):
        return path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(path, allow_patterns=_OUTPUT_ALLOW_PATTERNS)
    except Exception:
        return None


def select_sample_hidden_states(hidden_states: torch.Tensor, batch: Batch | None = None) -> torch.Tensor:
    batch = batch or get_global_ctx().batch
    if batch.is_prefill:
        indices = batch.attn_metadata.get_last_indices(batch.size)
        hidden_states = hidden_states[indices].contiguous()
    return hidden_states[: batch.size]


def _weight_key(prefix: str, index: int, suffix: str) -> str:
    stem = f"{prefix}.{index}" if prefix else str(index)
    return f"{stem}.{suffix}"


def _build_linear_op(
    state_dict: Dict[str, torch.Tensor],
    *,
    prefix: str,
    index: int,
) -> LinearReplicated:
    weight = state_dict[_weight_key(prefix, index, "weight")]
    bias = state_dict.get(_weight_key(prefix, index, "bias"))
    op = LinearReplicated(
        input_size=int(weight.shape[1]),
        output_size=int(weight.shape[0]),
        has_bias=bias is not None,
    )
    op.weight = weight
    op.bias = bias
    return op


def _build_layer_norm_op(
    state_dict: Dict[str, torch.Tensor],
    *,
    prefix: str,
    index: int,
    eps: float,
) -> _LayerNormHeadOp:
    weight = state_dict[_weight_key(prefix, index, "weight")]
    bias = state_dict.get(_weight_key(prefix, index, "bias"))
    return _LayerNormHeadOp(weight=weight, bias=bias, eps=eps)


def _make_activation(layer_type: str) -> BaseOP:
    if layer_type == "silu":
        return _SiLUHeadOp()
    if layer_type == "gelu":
        return _GELUHeadOp()
    if layer_type == "tanh":
        return _TanhHeadOp()
    if layer_type in {"dropout", "identity"}:
        return _IdentityHeadOp()
    raise ValueError(f"Unsupported output-head layer type: {layer_type}")


def _build_sequential_head(
    *,
    name: str,
    state_dict: Dict[str, torch.Tensor],
    source_path: str,
    prefix: str,
    layers: Sequence[Dict[str, Any]],
    squeeze_last_dim: bool,
) -> SequentialHiddenStateHead:
    ops: List[BaseOP] = []
    output_shape: tuple[int, ...] = ()
    dtype: torch.dtype | None = None
    for index, layer in enumerate(layers):
        layer_type = str(layer["type"]).lower()
        if layer_type == "layer_norm":
            eps = float(layer.get("eps", 1e-5))
            op = _build_layer_norm_op(state_dict, prefix=prefix, index=index, eps=eps)
            output_shape = tuple(op.weight.shape)
            dtype = op.weight.dtype
        elif layer_type == "linear":
            op = _build_linear_op(state_dict, prefix=prefix, index=index)
            output_shape = (int(op.weight.shape[0]),)
            dtype = op.weight.dtype
        else:
            op = _make_activation(layer_type)
        ops.append(op)

    if dtype is None:
        raise ValueError(f"Unable to infer dtype for output head '{name}'")
    if squeeze_last_dim and output_shape == (1,):
        output_shape = ()
    return SequentialHiddenStateHead(
        name=name,
        ops=ops,
        output_shape=output_shape,
        dtype=dtype,
        squeeze_last_dim=squeeze_last_dim,
        source_path=source_path,
    )


def _load_safetensors(
    path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    import safetensors

    state_dict: Dict[str, torch.Tensor] = {}
    with safetensors.safe_open(path, framework="pt", device=str(device)) as f:
        for name in f.keys():
            state_dict[name] = f.get_tensor(name).to(dtype=dtype)
    return state_dict


def _load_manifest_heads(
    folder: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> List[SequentialHiddenStateHead]:
    manifest_path = os.path.join(folder, "minisgl_outputs.json")
    if not os.path.exists(manifest_path):
        return []

    with open(manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    entries = payload if isinstance(payload, list) else payload.get("outputs", [])
    heads: List[SequentialHiddenStateHead] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each minisgl output manifest entry must be an object")
        name = str(entry["name"])
        source = entry.get("input", "last_hidden_state")
        if source != "last_hidden_state":
            raise ValueError(
                f"Output '{name}' requests unsupported input source '{source}'. "
                "Only last_hidden_state is supported today."
            )
        weights_name = str(entry.get("weights", f"{name}_head.safetensors"))
        state_dict = _load_safetensors(
            os.path.join(folder, weights_name),
            device=device,
            dtype=dtype,
        )
        architecture = entry.get("architecture", {})
        if architecture.get("type", "sequential") != "sequential":
            raise ValueError(f"Output '{name}' uses unsupported architecture {architecture.get('type')}")
        prefix = str(architecture.get("prefix", ""))
        layers = architecture.get("layers", [])
        if not isinstance(layers, list) or len(layers) == 0:
            raise ValueError(f"Output '{name}' must define at least one architecture layer")
        heads.append(
            _build_sequential_head(
                name=name,
                state_dict=state_dict,
                source_path=folder,
                prefix=prefix,
                layers=layers,
                squeeze_last_dim=bool(entry.get("squeeze_last_dim", False)),
            )
        )
    return heads


def _load_legacy_value_head(
    folder: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> SequentialHiddenStateHead | None:
    config_path = os.path.join(folder, "value_head_config.json")
    weight_path = os.path.join(folder, "value_head.safetensors")
    if not (os.path.exists(config_path) and os.path.exists(weight_path)):
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    head_type = str(payload.get("value_head_type", "linear")).lower()
    if head_type == "linear":
        layers = [
            {"type": "layer_norm", "eps": 1e-5},
            {"type": "linear"},
        ]
    elif head_type == "mlp":
        layers = [
            {"type": "layer_norm", "eps": 1e-5},
            {"type": "linear"},
            {"type": "silu"},
            {"type": "dropout"},
            {"type": "linear"},
        ]
    else:
        raise ValueError(f"Unsupported legacy value head type: {head_type}")

    state_dict = _load_safetensors(weight_path, device=device, dtype=dtype)
    return _build_sequential_head(
        name=str(payload.get("output_name", "value")),
        state_dict=state_dict,
        source_path=folder,
        prefix="net",
        layers=layers,
        squeeze_last_dim=True,
    )


class AttachedOutputManager:
    def __init__(
        self,
        *,
        hidden_size: int,
        hidden_dtype: torch.dtype,
        attached_heads: Iterable[SequentialHiddenStateHead] = (),
    ) -> None:
        self.hidden_size = hidden_size
        self.hidden_dtype = hidden_dtype
        self._heads: Dict[str, SequentialHiddenStateHead] = {}
        for head in attached_heads:
            self.add_head(head)

    def add_head(self, head: SequentialHiddenStateHead) -> None:
        if head.name in self._heads:
            existing = self._heads[head.name]
            logger.warning(
                "Skipping duplicate attached output '%s' from %s; already loaded from %s",
                head.name,
                head.source_path,
                existing.source_path,
            )
            return
        self._heads[head.name] = head

    def output_specs(self) -> tuple[SampleOutputSpec, ...]:
        specs = [
            SampleOutputSpec(
                name="last_hidden_state",
                shape=(self.hidden_size,),
                dtype=self.hidden_dtype,
            )
        ]
        for head in self._heads.values():
            specs.append(
                SampleOutputSpec(
                    name=head.name,
                    shape=head.output_shape,
                    dtype=head.dtype,
                )
            )
        return tuple(specs)

    def available_output_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.output_specs())

    def forward(
        self,
        hidden_states: torch.Tensor,
        requested_outputs: Sequence[str],
    ) -> Dict[str, torch.Tensor]:
        requested = tuple(dict.fromkeys(str(name) for name in requested_outputs if name))
        if not requested:
            return {}

        outputs: Dict[str, torch.Tensor] = {}
        if "last_hidden_state" in requested:
            outputs["last_hidden_state"] = hidden_states
        for name in requested:
            if name == "last_hidden_state":
                continue
            head = self._heads.get(name)
            if head is None:
                continue
            outputs[name] = head.forward(hidden_states)
        return outputs


def discover_output_manager(
    *,
    hidden_size: int,
    hidden_dtype: torch.dtype,
    search_paths: Sequence[str | None],
    device: torch.device,
) -> AttachedOutputManager:
    manager = AttachedOutputManager(hidden_size=hidden_size, hidden_dtype=hidden_dtype)
    for path in search_paths:
        if path is None:
            continue
        folder = _resolve_output_artifact_dir(path)
        if folder is None:
            continue
        for head in _load_manifest_heads(folder, device=device, dtype=hidden_dtype):
            manager.add_head(head)
        legacy = _load_legacy_value_head(folder, device=device, dtype=hidden_dtype)
        if legacy is not None:
            manager.add_head(legacy)
    return manager
