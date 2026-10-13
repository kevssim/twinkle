# Copyright (c) ModelScope Contributors. All rights reserved.
"""Sparse delta weight synchronization.

Layout:
    config.py    DeltaSyncConfig and the TypedDicts exchanged between trainer and rollout
    engine.py    DeltaShardedCheckpointEngine: gather, bucket and broadcast the changed elements
    wire.py      DeltaFlush / DeltaParam wire structures and checksum
    gather.py    sparse gather of per-rank deltas to rank 0
    spec.py      ShardSpec / BlockPlacement: where a local shard sits in the full tensor
    export.py    snapshot, diff and HF-coordinate delta export helpers
    trainer.py   CudaDeltaTrainerMixin: trainer-side Ray lifecycle
    rollout.py   CudaDeltaReceiver / CudaDeltaSamplerMixin: rollout-side lifecycle
    backends/    per-training-backend export hooks (base.py contract, fsdp.py implementation)
"""
from .config import DeltaSyncConfig
from .engine import DeltaShardedCheckpointEngine

__all__ = ['DeltaSyncConfig', 'DeltaShardedCheckpointEngine']
