from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from minisgl.core import HookProgram
from minisgl.hooks import HookCallback, HookSpec, LogitProcessor, normalize_layer_hooks


def _normalize_special_hooks(
    callbacks: HookCallback | Sequence[HookCallback] | None,
) -> Tuple[HookCallback, ...]:
    if callbacks is None:
        return ()
    if callable(callbacks):
        return (callbacks,)
    return tuple(callbacks)


def make_hook_program(
    *,
    layer_hooks: Mapping[int, HookCallback | Sequence[HookCallback]] | None = None,
    intra_layer_hooks: Mapping[int, HookCallback | Sequence[HookCallback]] | None = None,
    post_embedding_hooks: HookCallback | Sequence[HookCallback] | None = None,
    pre_lm_head_hooks: HookCallback | Sequence[HookCallback] | None = None,
    has_writes: bool = False,
    tier: int | None = None,
    logit_processor: LogitProcessor | None = None,
    label: str | None = None,
) -> HookProgram:
    """Build a HookProgram from plain callback maps.

    This keeps the common case lightweight: researchers can register callbacks directly
    without manually instantiating HookSpec or remembering which fields must be tuples.
    """

    resolved_tier = int(tier) if tier is not None else (2 if has_writes and intra_layer_hooks else 1)
    hook_spec = HookSpec(
        layer_hooks=normalize_layer_hooks(dict(layer_hooks or {})),
        intra_layer_hooks=normalize_layer_hooks(dict(intra_layer_hooks or {})),
        post_embedding_hooks=_normalize_special_hooks(post_embedding_hooks),
        pre_lm_head_hooks=_normalize_special_hooks(pre_lm_head_hooks),
        has_writes=bool(has_writes),
        tier=resolved_tier,
    )
    return HookProgram(hook_spec=hook_spec, logit_processor=logit_processor, label=label)


def chain_logit_processors(processors: Iterable[LogitProcessor | None]) -> LogitProcessor | None:
    active = [processor for processor in processors if processor is not None]
    if len(active) == 0:
        return None
    if len(active) == 1:
        return active[0]

    def _processor(logits):
        out = logits
        for processor in active:
            result = processor(out)
            out = out if result is out else result
        return out

    return _processor


def compose_hook_specs(*specs: HookSpec | None) -> HookSpec | None:
    active = [spec for spec in specs if spec is not None]
    if len(active) == 0:
        return None
    if len(active) == 1:
        return active[0]

    layer_hooks: Dict[int, List[HookCallback]] = {}
    intra_layer_hooks: Dict[int, List[HookCallback]] = {}
    post_embedding_hooks: List[HookCallback] = []
    pre_lm_head_hooks: List[HookCallback] = []
    has_writes = False
    tier = 1

    for spec in active:
        for layer_idx, callbacks in spec.layer_hooks.items():
            layer_hooks.setdefault(int(layer_idx), []).extend(callbacks)
        for layer_idx, callbacks in spec.intra_layer_hooks.items():
            intra_layer_hooks.setdefault(int(layer_idx), []).extend(callbacks)
        post_embedding_hooks.extend(spec.post_embedding_hooks)
        pre_lm_head_hooks.extend(spec.pre_lm_head_hooks)
        has_writes = has_writes or spec.has_writes
        tier = max(tier, spec.tier)

    return HookSpec(
        layer_hooks={k: tuple(v) for k, v in layer_hooks.items()},
        intra_layer_hooks={k: tuple(v) for k, v in intra_layer_hooks.items()},
        post_embedding_hooks=tuple(post_embedding_hooks),
        pre_lm_head_hooks=tuple(pre_lm_head_hooks),
        has_writes=has_writes,
        tier=tier,
    )


def compose_hook_programs(*programs: HookProgram | None, label: str | None = None) -> HookProgram | None:
    active = [program for program in programs if program is not None]
    if len(active) == 0:
        return None
    if len(active) == 1 and label is None:
        return active[0]

    hook_spec = compose_hook_specs(*(program.hook_spec for program in active))
    logit_processor = chain_logit_processors(program.logit_processor for program in active)
    labels = [program.label for program in active if program.label]

    combined_config = [program.hook_config for program in active if program.hook_config is not None]
    preset_names = [program.hook_preset_name for program in active if program.hook_preset_name]

    return HookProgram(
        hook_spec=hook_spec,
        logit_processor=logit_processor,
        hook_config=None if len(combined_config) == 0 else {"composed": combined_config},
        hook_preset_name=None if len(preset_names) == 0 else "+".join(preset_names),
        label=label or (" + ".join(labels) if labels else None),
    )


__all__ = [
    "chain_logit_processors",
    "compose_hook_programs",
    "compose_hook_specs",
    "make_hook_program",
]
