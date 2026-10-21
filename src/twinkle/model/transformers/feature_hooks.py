# Copyright (c) ModelScope Contributors. All rights reserved.
"""Optional training features applied to a constructed transformers model.

NEFTune (embedding noise), Liger kernels and the FSDP-aware pretrained-load
context manager are orthogonal features with no interaction with the rest of
the model surface. TransformersModel and the strategies keep thin same-named
delegation points (subclass overrides keep working); the logic lives in this
module, next to nothing else.
"""
from __future__ import annotations

import contextlib
import os
from typing import Any, Dict, Optional

import torch

from twinkle.utils import get_logger

# Env flags transformers/accelerate read to pick their FSDP-aware loading path (rank0 load,
# meta-device placement, no full CPU materialization per rank).
_FSDP_EFFICIENT_LOADING_ENV = {
    'ACCELERATE_USE_FSDP': 'true',
    'FSDP_CPU_RAM_EFFICIENT_LOADING': 'true',
}


@contextlib.contextmanager
def fsdp_pretrained_load_context(enabled: bool):
    """Enable the env flags required for transformers FSDP-aware loading when needed.

    A load-scoped transaction: the flags are restored (or popped) even when loading raises,
    so later imports in the same process never mistake themselves for FSDP-aware loaders.
    """
    if not enabled:
        yield
        return

    saved_env = {key: os.environ.get(key) for key in _FSDP_EFFICIENT_LOADING_ENV}
    os.environ.update(_FSDP_EFFICIENT_LOADING_ENV)
    try:
        yield
    finally:
        for key, old_val in saved_env.items():
            if old_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_val


def activate_neftune(model, neftune_noise_alpha: float) -> None:
    """Add NEFTune uniform noise to the embedding output during training only.

    Mirrors ``transformers.integrations.neftune.neftune_post_forward_hook``: scale is
    ``alpha / sqrt(seq_len * hidden)`` and the hook is a no-op when the module is in eval
    mode, so validation and generation see clean embeddings. Registered on the
    input-embedding module object, which a later LoRA/DDP wrap preserves (they wrap
    Linear layers / the top model, not this module).
    """
    embeddings = model.get_input_embeddings()
    if embeddings is None:
        get_logger().warning('neftune_noise_alpha is set but the model exposes no input embeddings; '
                             'skipping NEFTune.')
        return

    def neftune_post_forward_hook(module, input, output):
        if not module.training:
            return output
        dims = torch.tensor(output.size(1) * output.size(2))
        mag_norm = neftune_noise_alpha / torch.sqrt(dims)
        return output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)

    embeddings.register_forward_hook(neftune_post_forward_hook)


def apply_liger_kernel(model, liger_kernel_config: Optional[Dict[str, Any]] = None) -> None:
    """Patch the HF module's per-layer ops with Liger kernels via the mapping-driven ``kernelize``.

    ``liger_kernel_config`` (an ``{op: bool}`` dict) switches individual ops off on top of
    Liger's default set; None keeps the default. ``kernelize`` mutates the module in place
    (class / forward replacement), which is why it belongs at the end of construction,
    before the optimizer is built. The fused-linear-CE op is deliberately not a target
    here -- it skips the lm_head GEMM rather than swapping a layer, so it is a loss-path
    selection, not a kernelize op.
    """
    from twinkle.kernel import kernelize
    kernelize(model, liger_kernel_mapping(liger_kernel_config))


def liger_kernel_mapping(liger_kernel_config: Optional[Dict[str, Any]]) -> Optional[dict]:
    """DEFAULT_KERNEL_CONFIG minus any op the config switches off; None keeps the built-in default.

    Returning None rather than a copy of the default matters: ``kernelize`` logs
    family-skips at DEBUG on the default path but raises them to WARNING for any explicit
    mapping. 'fused_linear_cross_entropy' is not a kernelize op, so it matches no
    DEFAULT_KERNEL_CONFIG entry and is ignored here (the loss side consumes it instead).
    """
    if not liger_kernel_config:
        return None
    from twinkle.kernel.config import DEFAULT_KERNEL_CONFIG
    disabled = {op for op, enabled in liger_kernel_config.items() if not enabled}
    return {
        target: choice
        for target, choice in DEFAULT_KERNEL_CONFIG.items()
        if getattr(choice, 'op', None) not in disabled
    }
