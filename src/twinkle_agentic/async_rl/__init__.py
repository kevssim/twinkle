# Copyright (c) ModelScope Contributors. All rights reserved.
"""async_rl: the single-run streaming control core plus its supporting pieces.

The retained surface is exactly four modules: ``streaming_driver`` (the algorithm-agnostic
per-sample control kernel), ``weight_sync`` (weight publication strategies),
``cluster_adapter`` (adaptive resource planning and publication barrier) and
``score_worker`` (remote scoring). The previous TransferQueue runtime (three-worker
pipeline, scheduler, data plane, TQ samplers, context manager and types) is preserved
verbatim under ``legacy/`` and is imported from there only by explicit path.
"""
from twinkle_agentic.async_rl.cluster_adapter import (
    ClusterAdapter,
    ModelMemoryEstimate,
    RLResourceEstimator,
    RLResourcePlan,
    RolePlacement,
    apply_plan,
    elastic_roles,
    estimate_plan,
    flex_role_groups,
    resource_device_groups,
)
from twinkle_agentic.async_rl.score_worker import REF_ONLY_REMOTE_GROUP, ScoreWorker
from twinkle_agentic.async_rl.streaming_driver import (
    BufferRecord,
    DriverStats,
    GroupDecision,
    StreamingDriver,
)
from twinkle_agentic.async_rl.weight_sync import (
    AdapterSnapshotSync,
    InPlaceWeightSync,
    WeightSyncStrategy,
    build_weight_sync_strategy,
)

__all__ = [
    'AdapterSnapshotSync',
    'BufferRecord',
    'ClusterAdapter',
    'DriverStats',
    'GroupDecision',
    'InPlaceWeightSync',
    'ModelMemoryEstimate',
    'REF_ONLY_REMOTE_GROUP',
    'RLResourceEstimator',
    'RLResourcePlan',
    'RolePlacement',
    'ScoreWorker',
    'StreamingDriver',
    'WeightSyncStrategy',
    'apply_plan',
    'build_weight_sync_strategy',
    'elastic_roles',
    'estimate_plan',
    'flex_role_groups',
    'resource_device_groups',
]
