from __future__ import annotations

import functools
from contextlib import contextmanager
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@contextmanager
def torch_dtype(dtype: torch.dtype):
    import torch  # real import when used

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    import torch.cuda.nvtx as nvtx

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            with nvtx.range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator


_PIN_MEMORY_SUPPORTED: bool | None = None


def _disable_pin_memory() -> None:
    global _PIN_MEMORY_SUPPORTED
    _PIN_MEMORY_SUPPORTED = False


def pin_memory_supported() -> bool:
    import torch

    global _PIN_MEMORY_SUPPORTED
    if _PIN_MEMORY_SUPPORTED is not None:
        return _PIN_MEMORY_SUPPORTED
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe = torch.empty(1, dtype=torch.uint8, device="cpu", pin_memory=True)
        _PIN_MEMORY_SUPPORTED = bool(probe.is_pinned())
    except Exception:
        _PIN_MEMORY_SUPPORTED = False
    return _PIN_MEMORY_SUPPORTED


def empty_cpu(
    *size,
    dtype: torch.dtype,
    pin_memory: bool = False,
) -> torch.Tensor:
    import torch

    if pin_memory and pin_memory_supported():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return torch.empty(*size, dtype=dtype, device="cpu", pin_memory=True)
        except Exception:
            _disable_pin_memory()
    return torch.empty(*size, dtype=dtype, device="cpu")


def empty_like_cpu(
    tensor: torch.Tensor,
    *,
    pin_memory: bool = False,
) -> torch.Tensor:
    import torch

    if pin_memory and pin_memory_supported():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return torch.empty_like(tensor, device="cpu", pin_memory=True)
        except Exception:
            _disable_pin_memory()
    return torch.empty_like(tensor, device="cpu")


def tensor_cpu(
    data,
    *,
    dtype: torch.dtype,
    pin_memory: bool = False,
) -> torch.Tensor:
    import torch

    if pin_memory and pin_memory_supported():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return torch.tensor(data, dtype=dtype, device="cpu", pin_memory=True)
        except Exception:
            _disable_pin_memory()
    return torch.tensor(data, dtype=dtype, device="cpu")


def stage_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_cpu or tensor.is_pinned() or not pin_memory_supported():
        return tensor
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return tensor.pin_memory()
    except Exception:
        _disable_pin_memory()
        return tensor


class PinnedRingBuffer:
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        *,
        slots: int = 1,
        min_capacity: int = 1,
    ) -> None:
        import torch

        if slots < 1:
            raise ValueError(f"slots must be >= 1, got {slots}")
        self.device = device
        self.dtype = dtype
        self.slots = slots
        self.min_capacity = max(1, min_capacity)
        self._cursor = -1
        self._host: list[torch.Tensor | None] = [None] * slots
        self._device: list[torch.Tensor | None] = [None] * slots

    def acquire(self, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        if size < 0:
            raise ValueError(f"size must be >= 0, got {size}")
        self._cursor = (self._cursor + 1) % self.slots
        capacity = self._capacity_for(size)
        host = self._host[self._cursor]
        device = self._device[self._cursor]
        if (
            host is None
            or device is None
            or host.numel() < size
            or device.numel() < size
            or host.is_inference()
            or device.is_inference()
        ):
            # These buffers are staging areas that must remain mutable even when
            # the scheduler is running under torch.inference_mode().
            with torch.inference_mode(False):
                host = empty_cpu(capacity, dtype=self.dtype, pin_memory=self.device.type == "cuda")
                device = torch.empty(capacity, dtype=self.dtype, device=self.device)
            self._host[self._cursor] = host
            self._device[self._cursor] = device
        assert host is not None and device is not None
        return host[:size], device[:size]

    def _capacity_for(self, size: int) -> int:
        capacity = self.min_capacity
        while capacity < size:
            capacity *= 2
        return capacity
