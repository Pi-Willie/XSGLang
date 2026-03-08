from typing import Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual

    def forward_unfused(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        # Match fused kernel numerics: do residual add + normalization math in fp32,
        # then cast back to the model dtype for downstream layers.
        updated_fp32 = residual.to(torch.float32) + x.to(torch.float32)
        inv_rms = torch.rsqrt(updated_fp32.square().mean(dim=-1, keepdim=True) + self.eps)
        normed_fp32 = updated_fp32 * inv_rms
        out_fp32 = normed_fp32 * self.weight.to(device=x.device, dtype=torch.float32)
        updated_residual = updated_fp32.to(dtype=x.dtype)
        return out_fp32.to(dtype=x.dtype), updated_residual
