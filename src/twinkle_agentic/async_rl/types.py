# Copyright (c) ModelScope Contributors. All rights reserved.
"""The data vocabulary of the streaming control core.

``RolloutPolicy`` is the value object the driver hands to ``submit`` (a frozen snapshot of the
admission-time version, so the pin stays immutable while later publishes rebind the current
policy). ``PartitionAdmission`` is one admission window's record: what the staleness gate
opened, what the FIFO publish order retires, and what a caller-supplied ``push`` callable
receives as its window argument.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RolloutPolicy:
    """The policy one trajectory is admitted under: identity, version and adapter path.

    Frozen on purpose: the driver rebinds its current policy on publish (never mutates in
    place), so a handle's reference keeps the admission-time version even after release.
    """

    context_key: str
    adapter_name: Optional[str]
    version: int
    adapter_path: Optional[str]


@dataclass(frozen=True)
class PartitionAdmission:
    """One admission window: opened by the staleness gate, retired FIFO by a publish."""

    context_key: str
    partition_id: str
    step: int
    target_groups: int
    num_generations: int
