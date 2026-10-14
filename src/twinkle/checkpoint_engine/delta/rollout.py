# Copyright (c) ModelScope Contributors. All rights reserved.
"""Rollout-side Ray lifecycle and receivers for the CUDA delta checkpoint engine."""
from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from typing import Any, TYPE_CHECKING

import torch

from twinkle import remote_function
from .config import (DeltaReceiverPlacement, DeltaSyncConfig, DeltaTransportEndpoint,
                     DeltaWorkerCapabilities, DeltaWorkerMetadata)
from .engine import DeltaShardedCheckpointEngine

if TYPE_CHECKING:
    from ray.actor import ActorHandle

TensorFlush = tuple[list[tuple[str, torch.Tensor]], bool]


class CudaDeltaReceiver:
    """Checkpoint worker colocated with one vLLM worker"""

    def __init__(self, config: DeltaSyncConfig, endpoint: DeltaTransportEndpoint,
                 placement: DeltaReceiverPlacement) -> None:
        visible_device = placement.get('cuda_visible_device', placement['device_uuid'])
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_device
        print(f'[delta receiver rank={placement["rank"]}] '
              f'CUDA_VISIBLE_DEVICES={visible_device!r}, expected_uuid={placement["device_uuid"]}', flush=True)
        try:
            torch.cuda.set_device(0)
        except RuntimeError as error:
            raise RuntimeError(
                f'CUDA delta receiver initialization failed: CUDA_VISIBLE_DEVICES={visible_device!r}, '
                f'expected_uuid={placement["device_uuid"]}, '
                f'LD_PRELOAD={os.environ.get("LD_PRELOAD", "")!r}') from error
        actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
        if actual_uuid != placement['device_uuid']:
            raise ValueError(f'CUDA delta receiver GPU mismatch: expected {placement["device_uuid"]}, '
                             f'got {actual_uuid}, CUDA_VISIBLE_DEVICES={visible_device!r}')
        self.config = config
        self.zmq_handle = placement['zmq_handle']
        self.engine = DeltaShardedCheckpointEngine(bucket_size=config.bucket_bytes)
        self.engine.prepare()
        self.engine.init_process_group(placement['rank'], endpoint['world_size'], endpoint['master_metadata'])
        self._flushes: Iterator[TensorFlush] | None = None
        self._current: TensorFlush | None = None

    def ready(self) -> int:
        return self.engine.rank

    def next_flush(self) -> dict[str, bool] | None:
        if self._flushes is None:
            self._flushes = self.engine.receive_weights()
        self._current = next(self._flushes, None)
        if self._current is None:
            self._flushes = None
            return None
        return {'is_last': self._current[1]}

    def send_flush(self) -> None:
        from twinkle.sampler.vllm_sampler.bucketed_weight_transfer import BucketedWeightSender

        tensors, _ = self._current
        sender = BucketedWeightSender(
            zmq_handle=self.zmq_handle,
            bucket_size_mb=max(1, self.config.bucket_bytes >> 20), use_shm=False)
        try:
            asyncio.run(sender.async_send_weights(iter(tensors)))
        finally:
            self._current = None

    def close(self) -> None:
        self._current = None
        if self._flushes is not None:
            self._flushes.close()
            self._flushes = None
        self.engine.close()


class DeltaSamplerMixin:
    _delta_config: DeltaSyncConfig
    _delta_job_failed: bool = False
    _delta_weight_transfer_initialized: bool = False
    _delta_receivers: list[ActorHandle]

    async def _delta_rpc(self, method: str, kwargs: dict[str, Any] | None = None) -> list[Any]:
        return await self.engine.engine.collective_rpc(
            method, kwargs=kwargs or {}, timeout=self._delta_config.timeout_seconds)

    async def _reset_delta_prefix_cache(self) -> None:
        reset = await self.engine.engine.reset_prefix_cache(reset_connector=True)
        if reset is False:
            raise RuntimeError('prefix cache reset failed during delta synchronization')

    async def _clear_delta_cache(self) -> None:
        await self._reset_delta_prefix_cache()
        await self.engine.engine.reset_mm_cache()
        await self.engine.engine.reset_encoder_cache()

    @remote_function(dispatch='all', lazy_collect=False)
    def begin_delta_sync(self) -> None:
        async def begin():
            self.engine._delta_unavailable = True
            await self.engine.pause_delta_generation(self._delta_config.timeout_seconds)
            if self._delta_config.free_cache_engine:
                # Level 1 preserves the baseline needed by sparse updates.
                await self.engine.engine.sleep(level=1)
                await self.engine.engine.wake_up(tags=['weights'])
        self._run_in_loop(begin())

    @remote_function(dispatch='all', lazy_collect=False)
    def publish_delta_sampler(self) -> None:
        async def publish():
            if self._delta_config.free_cache_engine:
                await self.engine.engine.wake_up(tags=['kv_cache'])
                await self._reset_delta_prefix_cache()
            await self.engine.engine.resume_generation()
            self.engine._delta_unavailable = False
        self._run_in_loop(publish())

    @remote_function(dispatch='all', lazy_collect=False)
    def fail_delta_sampler(self) -> None:
        self._delta_job_failed = True
        self.engine._delta_unavailable = True
        self._run_in_loop(self.engine.pause_delta_generation(self._delta_config.timeout_seconds))

    @remote_function(dispatch='all', lazy_collect=False)
    def validate_delta_sampler(self, config: DeltaSyncConfig) -> DeltaWorkerCapabilities:
        self._delta_config = config
        if self._delta_job_failed:
            raise RuntimeError('delta sampler failed; restart the entire training/rollout job before reseeding')
        if config.free_cache_engine and not self.engine.enable_sleep_mode:
            raise ValueError('delta KV cache release requires enable_sleep_mode=True or free_cache_engine=False')
        results = self._run_in_loop(self._delta_rpc('validate_delta_worker'))
        if not results or any(result != results[0] for result in results):
            raise ValueError('inconsistent delta worker capabilities')
        if len(results) != results[0]['tp']:
            raise ValueError('delta topology requires one vLLM worker per TP rank')
        return results[0]

    @remote_function(dispatch='all', collect='flatten', lazy_collect=False)
    def delta_sampler_metadata(self) -> list[DeltaWorkerMetadata]:
        return self._run_in_loop(self._delta_rpc('delta_receiver_metadata'))

    def _delta_receiver_placements(
        self, metadata: list[DeltaWorkerMetadata], endpoint: DeltaTransportEndpoint,
    ) -> list[DeltaReceiverPlacement]:
        import ray

        workers = endpoint['rollout_workers']
        ranks = {(item['node_ip'], item['device_uuid']): rank for rank, item in enumerate(workers, start=1)}
        if not metadata or any(item not in workers for item in metadata):
            raise ValueError('local vLLM workers are missing from the delta topology')
        nodes = {node['NodeManagerAddress']: node['NodeID'] for node in ray.nodes() if node['Alive']}
        return [dict(rank=ranks[item['node_ip'], item['device_uuid']], node_id=nodes[item['node_ip']],
                     device_uuid=item['device_uuid'],
                     cuda_visible_device=item.get('cuda_visible_device', item['device_uuid']),
                     zmq_handle=item['zmq_handle'])
                for item in metadata]

    def _spawn_delta_receiver(
        self, actor, endpoint: DeltaTransportEndpoint, placement: DeltaReceiverPlacement,
    ) -> ActorHandle:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        return actor.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(placement['node_id'], soft=False),
            runtime_env={'env_vars': {
                'CUDA_VISIBLE_DEVICES': placement.get('cuda_visible_device', placement['device_uuid']),
                'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES': '1',
            }}).remote(self._delta_config, endpoint, placement)

    @remote_function(dispatch='all', lazy_collect=True)
    def init_delta_sampler_transport(self, endpoint: DeltaTransportEndpoint) -> None:
        import ray

        metadata = self._run_in_loop(self._delta_rpc('delta_receiver_metadata'))
        placements = self._delta_receiver_placements(metadata, endpoint)
        actor = ray.remote(num_cpus=0, num_gpus=0)(CudaDeltaReceiver)
        self._delta_receivers = []
        for placement in placements:
            self._delta_receivers.append(self._spawn_delta_receiver(actor, endpoint, placement))
        ray.get([receiver.ready.remote() for receiver in self._delta_receivers],
                timeout=self._delta_config.timeout_seconds)
        self._delta_weight_transfer_initialized = False

    async def _next_delta_flush(self) -> dict[str, bool] | None:
        flags = await asyncio.gather(*(receiver.next_flush.remote() for receiver in self._delta_receivers))
        if any(flag != flags[0] for flag in flags):
            raise ValueError('delta checkpoint receivers disagree about stream completion')
        return flags[0]

    async def _send_delta_flush(self) -> None:
        receivers = asyncio.create_task(self._delta_rpc('update_twinkle_delta_weights'))
        try:
            await asyncio.gather(*(receiver.send_flush.remote() for receiver in self._delta_receivers))
            await receivers
        finally:
            if not receivers.done():
                receivers.cancel()
            await asyncio.gather(receivers, return_exceptions=True)

    @remote_function(dispatch='all', lazy_collect=True)
    def receive_delta_weights(self) -> dict[str, bool]:
        async def receive():
            first = await self._next_delta_flush()
            if first is None:
                return {'changed': False}
            if not self._delta_weight_transfer_initialized:
                await self._delta_rpc('init_weight_transfer_engine', {'init_info': {}})
                self._delta_weight_transfer_initialized = True
            await self._delta_rpc('start_weight_update')

            saw_last = first['is_last']
            await self._send_delta_flush()
            following = await self._next_delta_flush()
            while following is not None:
                if saw_last:
                    raise ValueError('DeltaFlush stream yielded data after is_last=True')
                saw_last = following['is_last']
                await self._send_delta_flush()
                following = await self._next_delta_flush()
            if not saw_last:
                raise ValueError('DeltaFlush stream ended without is_last=True')
            await self._delta_rpc('finish_weight_update')
            await self._clear_delta_cache()
            return {'changed': True}
        return self._run_in_loop(receive())

    @remote_function(dispatch='all', lazy_collect=False)
    def close_delta_sampler(self) -> None:
        import ray

        self.engine._delta_unavailable = True
        self._run_in_loop(self.engine.pause_delta_generation(self._delta_config.timeout_seconds))
        receivers = getattr(self, '_delta_receivers', [])
        try:
            ray.get([receiver.close.remote() for receiver in receivers], timeout=self._delta_config.timeout_seconds)
        finally:
            for receiver in receivers:
                ray.kill(receiver, no_restart=True)
            self._delta_receivers = []
