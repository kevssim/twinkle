# Copyright (c) ModelScope Contributors. All rights reserved.
"""Sparse delta weight synchronization.

Layout:
    config.py    DeltaSyncConfig and the TypedDicts exchanged between trainer and rollout
    pipeline.py  DeltaSyncMixin: transport-agnostic gather/bucket/flush pipeline and seed/steady state machine
    engine.py    DeltaShardedCheckpointEngine: NCCL transport (hooks + rollout receive) over the pipeline
    platform.py  resolve_delta_engine_cls: GPU -> NCCL engine, NPU -> HCCL engine (lazy import)
    wire.py      DeltaFlush / DeltaParam wire structures and checksum
    gather.py    sparse gather of per-rank deltas to rank 0
    spec.py      ShardSpec / BlockPlacement: where a local shard sits in the full tensor
    export.py    snapshot, diff and HF-coordinate delta export helpers
    trainer.py   DeltaTrainerMixin: trainer-side Ray lifecycle
    rollout.py   CudaDeltaReceiver / DeltaSamplerMixin: rollout-side lifecycle
    backends/    per-training-backend export hooks (base.py contract, fsdp.py implementation)
"""
from .config import DeltaSyncConfig
from .engine import DeltaShardedCheckpointEngine
from .platform import resolve_delta_engine_cls

__all__ = ['DeltaSyncConfig', 'DeltaShardedCheckpointEngine', 'resolve_delta_engine_cls']
