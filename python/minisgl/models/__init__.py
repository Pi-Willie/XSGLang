from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .architectures import architecture_metadata, supported_architecture_names
from .config import ModelConfig, RotaryConfig
from .register import get_model_class

if TYPE_CHECKING:
    from .base import BaseLLMModel


def create_model(model_config: ModelConfig) -> "BaseLLMModel":
    return get_model_class(model_config.architectures[0], model_config)


def load_weight(*args: Any, **kwargs: Any) -> Any:
    from .weight import load_weight as _load_weight

    return _load_weight(*args, **kwargs)


__all__ = [
    "architecture_metadata",
    "create_model",
    "get_model_class",
    "load_weight",
    "ModelConfig",
    "RotaryConfig",
    "supported_architecture_names",
]
