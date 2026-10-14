# Copyright (c) ModelScope Contributors. All rights reserved.
# Adapted from https://github.com/volcengine/verl/blob/main/verl/checkpoint_engine/base.py
from contextlib import contextmanager
from time import perf_counter
from typing import List, Literal, Optional

from twinkle import Platform, get_logger
from .base import CheckpointEngine
from .delta import DeltaSyncConfig
from .mixin import CheckpointEngineMixin

logger = get_logger()

CheckpointEngineMode = Literal['auto', 'naive', 'colocate', 'standalone']
_VALID_MODES = {'auto', 'naive', 'colocate', 'standalone'}


class CheckpointEngineManager:
    """Weight synchronization manager for local and Ray deployments.

    ``mode`` selects one of three synchronization paths:

    * ``naive`` streams a local model's weight generator directly into a local sampler.
    * ``colocate`` connects Ray model and sampler actors sharing GPUs through CUDA IPC.
    * ``standalone`` connects disaggregated Ray actors through NCCL/HCCL.

    ``auto`` resolves local objects to ``naive`` and Ray actor handlers to ``standalone``. It never
    guesses ``colocate`` because actor placement cannot be inferred reliably from the driver.

    Delta synchronization follows verl's workflow ordering: finish training before syncing,
    await sync before submitting more training, and do not overlap sync and close calls.

    Architecture (following verl's CheckpointEngineManager):

        Trainer GPU(s)                          Rollout GPU(s)
        ┌──────────────────┐                    ┌──────────────────┐
        │ TransformersModel│                    │   vLLMSampler    │
        │  (Ray actors)    │                    │  (Ray actors)    │
        │        │         │                    │        │         │
        │        ▼         │                    │        ▼         │
        │ CheckpointEngine │ NCCL/HCCL/CUDA IPC │ CheckpointEngine │
        │  send_weights()  │ ─────────────────► │ receive_weights()│
        │                  │                    │        │         │
        │                  │                    │        ▼         │
        │                  │                    │ VLLMEngine       │
        │                  │                    │  update_weights()│
        │                  │                    │   (CUDA IPC)     │
        │                  │                    │        │         │
        │                  │                    │        ▼         │
        │                  │                    │ vLLM subprocess  │
        │                  │                    │  load_weights()  │
        └──────────────────┘                    └──────────────────┘

    Usage:
        >>> manager = CheckpointEngineManager(model=model, sampler=sampler)
        >>> manager.sync_weights()  # Call after each training step

    With colocated Ray actors, the caller also owns the memory schedule, because only it knows where
    in the loop the device is free. The sampler must have its weights resident to be written into --
    ``sleep(1)`` puts them on the host -- and the trainer has to step aside before a rollout:

        >>> manager = CheckpointEngineManager(model=model, sampler=sampler, mode='colocate')
        >>> sampler.wake_up(tags=['weights'])   # able to receive, still without a KV cache
        >>> manager.sync_weights()
        >>> model.offload_to_cpu()              # the trainer's turn is over
        >>> sampler.wake_up()                   # KV cache, then generate
        >>> ...
        >>> sampler.sleep()                     # and back the other way
        >>> model.reload_to_gpu()

    This is deliberately not done for the caller: waking a sampler that is already awake is not a
    no-op in vLLM, so guessing the current state here would trade one foot-gun for another.
    """

    def __init__(
        self,
        model: 'CheckpointEngineMixin',
        sampler: 'CheckpointEngineMixin',
        platform: str = 'GPU',
        mode: CheckpointEngineMode = 'auto',
        sync_backend: Literal['full', 'delta_sharded'] = 'full',
        delta_config: Optional[DeltaSyncConfig] = None,
    ) -> None:
        self.model = model
        self.sampler = sampler
        self.requested_mode = mode
        self.mode = self._resolve_mode(mode, model, sampler)
        self.backend_cls = self.decide_backend_engine(platform, self.mode)
        if sync_backend not in ('full', 'delta_sharded'):
            raise ValueError(f'Unknown weight sync backend: {sync_backend}')
        if sync_backend == 'delta_sharded' and self.mode != 'standalone':
            raise ValueError('delta_sharded requires standalone trainer and rollout actors')
        self.sync_backend = sync_backend
        self.delta_config = delta_config or DeltaSyncConfig()
        if sync_backend == 'delta_sharded' and Platform.get_platform(platform).__name__ != 'GPU':
            raise ValueError('delta_sharded currently supports CUDA only')
        self._delta_replica_count = None
        self._delta_round = 0
        self._delta_failed = False
        self.last_delta_metrics = None

        if self.mode == 'colocate':
            # Each side builds its own engine inside its worker, so both have to be told which one.
            self.model.set_checkpoint_engine_backend('ipc')
            self.sampler.set_checkpoint_engine_backend('ipc')

        # LoRA sync state: tracks whether the first full sync has been done.
        # After the first sync, only LoRA adapter weights are transferred.
        self.base_sync_done: bool = False
        # Cached peft_config dict for LoRA-only sync.
        # Fetched lazily from the model on first LoRA sync.
        self._peft_config: dict | None = None
        self._model_keys: Optional[List[str]] = None

    @staticmethod
    def _resolve_mode(
        mode: CheckpointEngineMode,
        model: 'CheckpointEngineMixin',
        sampler: 'CheckpointEngineMixin',
    ) -> Literal['naive', 'colocate', 'standalone']:
        if mode not in _VALID_MODES:
            valid = ', '.join(sorted(_VALID_MODES))
            raise ValueError(f'Unknown checkpoint engine mode {mode!r}; expected one of: {valid}.')

        model_has_actors = bool(getattr(model, '_actors', None))
        sampler_has_actors = bool(getattr(sampler, '_actors', None))
        if model_has_actors != sampler_has_actors:
            raise ValueError(
                'CheckpointEngineManager requires model and sampler to use the same deployment shape: '
                'both must be local objects or both must be Ray actor handlers.')

        if mode == 'auto':
            return 'standalone' if model_has_actors else 'naive'
        if mode == 'naive' and model_has_actors:
            raise ValueError("mode='naive' requires local model and sampler objects without Ray actors.")
        if mode in ('colocate', 'standalone') and not model_has_actors:
            raise ValueError(f"mode={mode!r} requires model and sampler to be Ray actor handlers.")
        return mode

    @staticmethod
    def decide_backend_engine(
        platform: Optional[str] = None,
        mode: Literal['naive', 'colocate', 'standalone'] = 'standalone',
    ) -> Optional['CheckpointEngine']:
        if mode == 'naive':
            return None

        platform_name = Platform.get_platform(platform).__name__
        if mode == 'colocate':
            from twinkle.checkpoint_engine import IPCCheckpointEngine
            return IPCCheckpointEngine
        if mode != 'standalone':
            raise ValueError(f'Cannot select a backend for unresolved mode {mode!r}.')
        if platform_name == 'GPU':
            from twinkle.checkpoint_engine import NCCLCheckpointEngine
            return NCCLCheckpointEngine
        elif platform_name == 'NPU':
            from twinkle.checkpoint_engine import HCCLCheckpointEngine
            return HCCLCheckpointEngine
        else:
            raise NotImplementedError

    def sync_weights(self, merge_and_sync=True):
        """
        Synchronize the weights between the model and the sampler.

        This method ensures that the sampler's weights are consistent with the model's
        current state. It supports two synchronization modes: full merge-and-sync or
        separate base-and-LoRA sync.

        Args:
            merge_and_sync (bool, optional): Whether to merge and sync the weights.
                - If True: LoRA weights are merged into the base model, then the
                combined weights are synchronized to the sampler on every call.
                - If False: On the first call, base model weights are synced to the
                sampler. On subsequent calls, only the LoRA adapter weights are
                synced incrementally.
                Defaults to True.

        Returns:
            None
        """
        if self.sync_backend == 'delta_sharded':
            with self._delta_rpc_timeouts():
                self._sync_delta_weights(merge_and_sync)
            return
        if self.mode == 'naive':
            self._sync_weights_naive(merge_and_sync)
            return

        model_metadata = self.model.prepare_checkpoint_engine([True]
                                                              + [False] * (self.model.device_mesh.world_size - 1))
        self.sampler.prepare_checkpoint_engine(False)
        model_kwargs, sampler_kwargs = self.backend_cls.build_topology(
            self.model.device_mesh.world_size,
            self.sampler.device_mesh.data_world_size,
            [model_metadata],
        )
        # Launch both init calls concurrently — TCPStore server (model rank 0)
        # blocks until all clients (sampler ranks) connect, so these MUST NOT
        # be serialised.  lazy_collect=True makes them return futures.
        model_init = self.model.init_checkpoint_process_group(**model_kwargs)
        sampler_init = self.sampler.init_checkpoint_process_group(**sampler_kwargs)
        model_init()  # wait for model init to complete
        sampler_init()  # wait for sampler init to complete

        peft_config = None
        if self.base_sync_done and not merge_and_sync:
            if self._peft_config is None:
                self._peft_config = self.model.get_peft_config_dict()
            peft_config = self._peft_config

        self._ensure_model_keys()

        model_result = self.model.send_weights(
            base_sync_done=self.base_sync_done, merge_and_sync=merge_and_sync, model_keys=self._model_keys)
        sampler_result = self.sampler.receive_weights(base_sync_done=self.base_sync_done, peft_config=peft_config)
        model_result()
        sampler_result()

        self.model.finalize_checkpoint_engine()
        self.sampler.finalize_checkpoint_engine()

        if not self.base_sync_done:
            self.base_sync_done = True
            if not merge_and_sync:
                logger.info('Base model sync completed, subsequent syncs will be LoRA-only')

    def _prepare_delta_transport(self):
        capabilities = self.sampler.validate_delta_sampler(self.delta_config)
        replicas = capabilities if isinstance(capabilities, list) else [capabilities]
        workers = self.sampler.delta_sampler_metadata()
        if len(workers) != sum(replica['tp'] for replica in replicas):
            raise ValueError('delta topology does not include every rollout TP worker')
        devices = {(worker['node_ip'], worker['device_uuid']) for worker in workers}
        if len(devices) != len(workers):
            raise ValueError('delta checkpoint receivers must occupy distinct rollout GPUs')
        metadata = self.model.prepare_delta_sync(self.delta_config)
        return dict(metadata, world_size=len(workers) + 1, rollout_workers=workers), len(replicas)

    def _sync_delta_weights(self, merge_and_sync):
        if not merge_and_sync:
            raise ValueError('delta_sharded does not support LoRA synchronization')
        if self._delta_failed:
            raise RuntimeError('delta session failed; restart the entire training/rollout job before full seed')
        started = perf_counter()
        seed = self._delta_replica_count is None
        if seed:
            endpoint, replica_count = self._prepare_delta_transport()
        try:
            self.sampler.begin_delta_sync()
            if seed:
                trainer_init = self.model.init_delta_transport(endpoint)
                sampler_init = self.sampler.init_delta_sampler_transport(endpoint)
                trainer_init()
                sampler_init()
                self._delta_replica_count = replica_count
            sender = self.model.send_delta_weights()
            receiver = self.sampler.receive_delta_weights()
            metrics = sender()
            worker_metrics = receiver()
            results = worker_metrics if isinstance(worker_metrics, list) else [worker_metrics]
            if len(results) != self._delta_replica_count or any(
                    result['changed'] != results[0]['changed'] for result in results):
                raise ValueError('delta rollout replicas disagree about update completion')
            self.sampler.publish_delta_sampler()
            self._delta_round += 1
            self.base_sync_done = True
            self.last_delta_metrics = {'trainers': metrics, 'workers': worker_metrics,
                                       'total_seconds': perf_counter() - started, 'seed': seed}
        except Exception:
            self._delta_failed = True
            try:
                self.sampler.fail_delta_sampler()
            except Exception:
                logger.exception('Could not pause failed delta sampler; restart rollout actors before serving')
            raise

    def close_sync_session(self):
        """Release delta snapshots and communicators; keep rollout paused until a new full seed."""
        with self._delta_rpc_timeouts():
            if self.sync_backend == 'delta_sharded':
                self._close_delta_session()

    @contextmanager
    def _delta_rpc_timeouts(self):
        missing = object()
        previous = [(actor, getattr(actor, '_ray_get_timeout', missing)) for actor in (self.model, self.sampler)]
        try:
            for actor, _ in previous:
                actor._ray_get_timeout = self.delta_config.timeout_seconds
            yield
        finally:
            for actor, timeout in previous:
                if timeout is missing:
                    # Ray driver proxies can inherit nn.Module without initializing its registries.
                    object.__delattr__(actor, '_ray_get_timeout')
                else:
                    actor._ray_get_timeout = timeout

    def _close_delta_session(self):
        errors = []
        close_methods = (self.sampler.close_delta_sampler, self.model.close_delta_trainer)
        for close in close_methods:
            try:
                close()
            except Exception as error:
                errors.append(error)
        if errors:
            self._delta_failed = True
            raise RuntimeError('delta cleanup failed; restart the affected actors') from errors[0]
        self._delta_replica_count = None
        self._delta_round = 0
        # Cleanup cannot make an unknown partially updated model safe for a new session.
        self.base_sync_done = False

    def _ensure_model_keys(self):
        if self._model_keys is not None:
            return

        if hasattr(self.sampler, 'get_state_keys'):
            self._model_keys = self.sampler.get_state_keys()

        if self._model_keys is None:
            self._model_keys = []

        # vLLM may have grouped params - use word boundaries to avoid substring matches
        import re
        _STACKED_MAPPINGS = [
            (re.compile(r'\bqkv_proj\b'), ('q_proj', 'k_proj', 'v_proj', 'q', 'k', 'v')),
            (re.compile(r'\bgate_up_proj\b'), ('gate_proj', 'up_proj')),
            (re.compile(r'\bin_proj_qkvz\b'), ('in_proj_qkv', 'in_proj_z')),
            (re.compile(r'\bin_proj_ba\b'), ('in_proj_b', 'in_proj_a')),
            (re.compile(r'\blanguage_model\.model\b'), ('model.language_model', )),
            (re.compile(r'^visual\.'), ('model.visual.', )),
        ]

        def _expand_keys(keys):
            result = set(keys)
            for key in keys:
                for pattern, individuals in _STACKED_MAPPINGS:
                    if pattern.search(key):
                        for ind in individuals:
                            result.add(pattern.sub(ind, key))
            return result

        # Two passes for chain expansion (e.g., language_model.model + qkv_proj)
        expanded = _expand_keys(self._model_keys)
        expanded = _expand_keys(expanded)
        self._model_keys = list(expanded)

    def _sync_weights_naive(self, merge_and_sync):
        """Stream model weights directly into a local sampler."""
        peft_config = None
        if self.base_sync_done and not merge_and_sync:
            if self._peft_config is None:
                self._peft_config = self.model.get_peft_config_dict()
            peft_config = self._peft_config

        self._ensure_model_keys()
        weights = self.model._get_weight_generator(
            base_sync_done=self.base_sync_done,
            merge_and_sync=merge_and_sync,
            model_keys=self._model_keys,
        )
        self.sampler.receive_weights(
            weights=weights,
            base_sync_done=self.base_sync_done,
            peft_config=peft_config,
        )

        if not self.base_sync_done:
            self.base_sync_done = True
            if not merge_and_sync:
                logger.info('Base model sync completed, subsequent syncs will be LoRA-only')
