# Copyright (c) Model Scope Contributors. All rights reserved.
"""Archived TQ data-plane vocabulary: partition/group/batch descriptors and the rollout-output row.

Moved with the TransferQueue runtime that consumes them; the live control plane keeps only
``RLContext`` / ``RolloutPolicy`` / ``PartitionAdmission`` in ``async_rl/types.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from twinkle_agentic.async_rl.types import PartitionAdmission, RLContext

# Backward-compatible alias archived with its only callers (the TQ pipeline).
LoraContext = RLContext


@dataclass
class PromptGroup:
    """A prompt and the BatchMeta descriptor reserved for its generated samples."""

    context: RLContext
    partition: PartitionAdmission
    group_id: str
    prompt: dict[str, Any]
    batch_meta: Any

    @property
    def partition_id(self) -> str:
        return self.partition.partition_id

    @property
    def num_samples(self) -> int:
        return self.partition.num_generations


@dataclass
class PreparedPartition:
    """Partition prepared in TQ and ready for sampler submission."""

    admission: PartitionAdmission
    groups: tuple[PromptGroup, ...]
    sampling_params: Any


@dataclass
class ClaimedBatch:
    """A TQ-consumed batch and the BatchMeta descriptor selecting its samples."""

    admission: PartitionAdmission
    data: Any
    batch_meta: Any
    sample_tags: tuple[dict[str, Any], ...] = ()


class RolloutOutput(TypedDict, total=False):
    logprobs: list[float]
    rewards: float
    completion_length: int
    generation_idx: int
    rollout_policy_version: int
    rollout_adapter_path: str | None
    rollout_policy_versions: list[int]
    initial_policy_version: int
    final_policy_version: int
    policy_version_span: int
