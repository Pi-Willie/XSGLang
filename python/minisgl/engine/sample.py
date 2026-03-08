from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.utils import is_sm90_supported, nvtx_annotate, tensor_cpu

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None
    forced_next_tokens: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return tensor_cpu(data, dtype=dtype, pin_memory=device.type == "cuda").to(
        device,
        non_blocking=True,
    )


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    import flashinfer.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        forced_next_tokens = None
        if batch.forced_next_tokens is not None:
            forced_next_tokens = batch.forced_next_tokens.to(
                self.device,
                dtype=torch.int32,
                non_blocking=True,
            )
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None, forced_next_tokens=forced_next_tokens)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(
            temperatures,
            top_k=top_k,
            top_p=top_p,
            forced_next_tokens=forced_next_tokens,
        )

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:  # greedy sampling
                sampled = torch.argmax(logits, dim=-1)
            else:
                sampled = sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
            if args.forced_next_tokens is None:
                return sampled
            forced = args.forced_next_tokens
            if forced.shape[0] != sampled.shape[0]:
                raise ValueError(
                    "forced_next_tokens length does not match sampled batch size: "
                    f"{forced.shape[0]} vs {sampled.shape[0]}"
                )
            if torch.all(forced >= 0):
                return forced.to(dtype=sampled.dtype)
            sampled = sampled.clone()
            mask = forced >= 0
            sampled[mask] = forced[mask].to(dtype=sampled.dtype)
            return sampled
