# Copyright (c) ModelScope Contributors. All rights reserved.
"""Minimal mapping-driven kernel replacement.

Public API: ``kernelize``, ``hub`` (re-exported from ``twinkle.kernel``).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HubRef:
    """Lightweight reference to a HuggingFace Hub kernel layer.

    Resolved lazily by ``kernelize`` via the optional ``kernels`` package.
    """
    repo_id: str
    layer_name: str
    revision: str | None = None
    version: int | None = None
    backend: str | None = None
    trust_remote_code: bool = False


def hub(
    ref: str,
    *,
    revision: str | None = None,
    version: int | None = None,
    backend: str | None = None,
    trust_remote_code: bool = False,
) -> HubRef:
    """Build a ``HubRef`` for use as a ``kernelize`` mapping value.

    ``ref`` is ``'<repo_id>:<LayerName>'`` (e.g. ``'org/repo:SiluAndMul'``).
    Exactly one of ``revision`` or ``version`` must be supplied.
    """
    if (revision is None) == (version is None):
        raise ValueError('Exactly one of `revision` or `version` must be specified.')
    if ':' not in ref:
        raise ValueError(f"Hub ref must be 'repo_id:LayerName', got: {ref!r}")
    repo_id, layer_name = ref.rsplit(':', 1)
    return HubRef(repo_id, layer_name, revision, version, backend, trust_remote_code)