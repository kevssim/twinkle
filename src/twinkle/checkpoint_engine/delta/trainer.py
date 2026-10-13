# Copyright (c) ModelScope Contributors. All rights reserved.
"""Trainer-side Ray lifecycle for the CUDA delta checkpoint engine."""
from __future__ import annotations

import asyncio
from datetime import timedelta
from time import perf_counter
from typing import Any

import torch.distributed as dist

from twinkle import remote_function
from ..nccl_checkpoint_engine import MasterMetadata
from .config import DeltaSyncConfig, DeltaTrainerMetadata, DeltaTransportEndpoint
from .engine import DeltaShardedCheckpointEngine


class CudaDeltaTrainerMixin:
    """Lifecycle RPCs only. Export hooks come from a DeltaTrainerBackend subclass on the same model."""
    _delta_engine: DeltaShardedCheckpointEngine | None = None
    _delta_metadata_group: dist.ProcessGroup | None = None
    _delta_master_metadata: MasterMetadata | None = None

    def _validate_delta_ranks(self) -> None:
        error = None
        try:
            self.validate_delta_backend()
        except Exception as validation_error:
            error = str(validation_error)
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error, group=self._delta_metadata_group)
        if any(errors):
            raise ValueError(f'Unsupported delta configuration: {errors}')

    def _initialize_delta_engine(self, config: DeltaSyncConfig) -> None:
        self._delta_metadata_group = dist.new_group(
            backend='gloo', timeout=timedelta(seconds=config.timeout_seconds))
        try:
            self._validate_delta_ranks()
        except Exception:
            dist.destroy_process_group(self._delta_metadata_group)
            self._delta_metadata_group = None
            raise
        self._delta_device = 'cuda'
        self._delta_pin_snapshots = config.delta_pin_snapshots
        self._delta_engine = DeltaShardedCheckpointEngine(
            bucket_size=config.bucket_bytes, batch_gather=config.gather_batch_size)
        self._delta_engine.is_master = dist.get_rank() == 0
        self._delta_master_metadata = self._delta_engine.prepare()

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def prepare_cuda_delta_sync(self, config: DeltaSyncConfig) -> DeltaTrainerMetadata:
        self._lazy_wrap_model()
        if self._delta_engine is None:
            self._initialize_delta_engine(config)
        return {'device': 'cuda', 'master_metadata': self._delta_master_metadata}

    @remote_function(dispatch='all', lazy_collect=True)
    def init_cuda_delta_transport(self, endpoint: DeltaTransportEndpoint) -> None:
        rank = 0 if dist.get_rank() == 0 else -1
        self._delta_engine.init_process_group(rank, endpoint['world_size'], endpoint['master_metadata'])

    @remote_function(dispatch='all', lazy_collect=True)
    def send_cuda_delta_weights(self) -> dict[str, Any]:
        started = perf_counter()
        metrics = asyncio.run(self._delta_engine.send_weights(self))
        return {'send_seconds': perf_counter() - started, 'checkpoint_engine': metrics}

    @remote_function(dispatch='all', lazy_collect=False)
    def close_cuda_delta_trainer(self) -> None:
        if self._delta_engine is not None:
            self._delta_engine.close()
            self._delta_engine = None
        snapshots = getattr(self, '_delta_shard_snap', {})
        snapshots.clear()
        if self._delta_metadata_group is not None:
            dist.destroy_process_group(self._delta_metadata_group)
            self._delta_metadata_group = None
