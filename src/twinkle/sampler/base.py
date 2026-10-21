# Copyright (c) Modelscope Contributors. All rights reserved.
"""The sampler component: blocking ``sample`` plus the engine-agnostic async generation
protocol (``submit_generation`` / ``get_generation_status`` / ``collect_generation`` /
``cancel_generation``), cooperative pause/resume and abort/resume-merge partial rollout.

Contract for a concrete engine sampler: own a running background event loop at
``self._async_loop``, initialise ``self._generation_submissions`` to a dict, and implement
``async def _generate_inputs(self, inputs, sampling_params, *, adapter_name, adapter_path,
use_base_model, allow_partial_rollout) -> list[SampleResponse]``. Everything else here is
plain component code: with or without infra, the class instantiates and these methods run.
"""
from abc import ABC, abstractmethod
import asyncio
import time
import zlib
from copy import copy
from dataclasses import dataclass
from concurrent.futures import CancelledError, Future, InvalidStateError
from typing import Any, Awaitable, Callable, List, Optional, Sequence, Type, Union

import twinkle
from twinkle import remote_function
from twinkle.data_format import (InputFeature, PoolingParams, PoolingResponse, SampleResponse, SampledSequence,
                                 SamplingParams, Trajectory, is_encoded)
from twinkle.patch import Patch
from twinkle.perf import timed
from twinkle.sampler import generation
from twinkle.template import Template
from twinkle.utils import construct_class, get_logger
from peft import PeftConfig


@dataclass
class GenerationOwner:
    worker_id: str
    submitted_at: float
    collection: Any = None
    cancelling: bool = False
    submitting: bool = False


@dataclass
class PartialRolloutOutcome:
    """What one sample's resume loop produced.

    ``attempt_states`` carries the opaque per-attempt state the caller's ``run_attempt`` returned -- the
    TransferQueue path uses it for the ``RolloutPolicy`` of each segment so it can attribute merged tokens
    to policy versions; the plain core path leaves it ``None``. This mixin never inspects it, so a caller
    can thread whatever per-attempt context it needs through the resume loop without the loop knowing what
    it is.
    """
    response: SampleResponse
    attempt_states: tuple[Any, ...]
    attempts: int
    was_aborted: bool
    resumed_partial_output: bool


class Sampler(ABC):
    # Domain declarations consumed by infra's WorkerPool: which of this component's
    # method names are configuration calls (recorded for replay on future workers),
    # pause/resume controls (their completion is booked on the pool), barrier-exempt
    # controls, and the generation method. infra itself knows none of these names.
    _REPLAY_METHODS = frozenset({'set_template', 'add_adapter_to_sampler', 'apply_patch'})
    _PAUSE_METHODS = {'pause_generation': True, 'resume_generation': False}
    _CONTROL_METHODS = frozenset({'pause_generation', 'resume_generation', 'abort_all_inflight',
                                  '_get_generation_status', '_cancel_generation', '_cancel_all_generations',
                                  '_generation_count'})
    _GENERATION_METHODS = frozenset({'sample'})


    def __init__(self):
        self.engine = None
        self.template = None

    @abstractmethod
    def sample(
        self,
        inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
        sampling_params: Optional[SamplingParams] = None,
        adapter_name: str = '',
        *,
        num_samples: int = 1,
    ) -> List[SampleResponse]:
        """Sample responses for given inputs.

        Args:
            inputs: Either InputFeature(s) or Trajectory(s).
                - InputFeature: Must contain 'input_ids'. For multimodal, include 'images'/'videos'.
                - Trajectory: Must contain 'messages'. Requires template to be set.
            sampling_params: Sampling parameters.
            adapter_name: Optional LoRA adapter name.
            num_samples: Number of completions to generate per input prompt.
                        When > 1, returns num_samples sequences for each input.

        Returns:
            SampleResponse containing sampled sequences.
            Total sequences = len(inputs) * num_samples.
        """
        pass

    def encode(
        self,
        inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
        pooling_params: Optional[PoolingParams] = None,
        adapter_name: str = '',
    ) -> List[PoolingResponse]:
        """Run a pooling (non-generative) forward and return the pooled output per input.

        The pooling counterpart of :meth:`sample`: one forward per input, no decoding, producing an
        embedding vector / class scores / a relevance score. Only samplers whose backend can serve a
        pooling head (vLLM, SGLang) override this; the default raises so a backend without pooling
        support fails loudly rather than silently returning hidden states that look like embeddings.

        Args:
            inputs: Either InputFeature(s) or Trajectory(s), as for :meth:`sample`.
            pooling_params: Which head to run and how to post-process it. Defaults to a plain embedding.
            adapter_name: Optional LoRA adapter name.

        Returns:
            One PoolingResponse per input, in input order.
        """
        raise NotImplementedError(f'{type(self).__name__} does not support pooling/encode; use a vLLM or '
                                  'SGLang sampler, or a HF forward via build_model(task=...).')

    @abstractmethod
    def apply_patch(self, patch_cls: Union[Patch, Type[Patch], str], **kwargs) -> None:
        ...

    @staticmethod
    def _not_encoded(inputs: Any) -> bool:
        """Check if inputs are not yet encoded (i.e., is Trajectory, not InputFeature).

        Delegates to the single shared predicate so the three backends and the wire
        schema cannot drift apart.
        """
        assert isinstance(inputs, dict), f'Expected dict, got {type(inputs)}'
        return not is_encoded(inputs)

    def _is_trajectory(self, inputs: Any) -> bool:
        """Check if inputs are Trajectory type (not encoded)."""
        if isinstance(inputs, list):
            if not inputs:
                return False
            inputs = inputs[0]
        if isinstance(inputs, dict):
            return self._not_encoded(inputs)
        return False

    def _normalize_inputs(self, inputs) -> List:
        if isinstance(inputs, dict):
            return [inputs]
        return list(inputs)

    def encode_trajectory(self,
                          trajectory: Trajectory,
                          adapter_name: str = '',
                          add_generation_prompt: bool = True) -> InputFeature:
        if self.template is None:
            raise ValueError(f"Template not set for adapter '{adapter_name}'. Use set_template() first.")
        return generation.encode_trajectory(self.template, trajectory, add_generation_prompt)

    def decode_response(self, token_ids: List[int], adapter_name: str = '') -> str:
        """Decode token ids to text."""
        template = self.template
        if template is None:
            raise ValueError(f"Template not set for adapter '{adapter_name}'. Use set_template() first.")
        return template.decode(token_ids)

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs):
        if 'model_id' not in kwargs and hasattr(self, 'model_id'):
            kwargs['model_id'] = self.model_id
        template = construct_class(template_cls, Template, twinkle.template, **kwargs)
        self.template = template

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def add_adapter_to_sampler(self, adapter_name: str, config: PeftConfig) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Async generation submission: strategies and the submit/poll/collect
    # protocol. Driver-side orchestration methods probe the optional infra
    # worker pool (``self._worker_pool``) and fall back to the no-pool path
    # below; worker-side methods live on the engine's event loop.
    # ------------------------------------------------------------------

    @staticmethod
    def _dispatch_generation(
        worker_count: int,
        worker_index: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        **_dispatch_kwargs,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Slice CS inputs while allowing a prompt count smaller than DP size."""
        sliced_args = list(args)
        sliced_kwargs = dict(kwargs)
        if len(sliced_args) > 1:
            inputs = sliced_args[1]
            target = ('args', 1)
        elif 'inputs' in sliced_kwargs:
            inputs = sliced_kwargs['inputs']
            target = ('kwargs', 'inputs')
        else:
            raise ValueError('submit_generation requires inputs')

        input_list = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
        target_worker = sliced_kwargs.pop('target_worker', None)
        if target_worker is not None:
            if type(target_worker) is not int or not 0 <= target_worker < worker_count:
                raise ValueError(f'target_worker must be in [0, {worker_count}), got {target_worker!r}.')
            shard = input_list if worker_index == target_worker else []
        elif len(input_list) < worker_count:
            if sliced_args:
                submission_id = sliced_args[0]
            else:
                submission_id = sliced_kwargs.get('submission_id')
            if submission_id is None:
                raise ValueError('submit_generation requires submission_id')
            first_worker = zlib.crc32(str(submission_id).encode('utf-8')) % worker_count
            shard = [item for index, item in enumerate(input_list) if (first_worker + index) % worker_count == worker_index]
        else:
            size, remainder = divmod(len(input_list), worker_count)
            start = worker_index * size + min(worker_index, remainder)
            stop = (worker_index + 1) * size + min(worker_index + 1, remainder)
            shard = input_list[start:stop]
        if target[0] == 'args':
            sliced_args[target[1]] = shard
        else:
            sliced_kwargs[target[1]] = shard
        return tuple(sliced_args), sliced_kwargs


    @staticmethod
    def _subset_normalize(args, kwargs):
        args, kwargs = list(args), dict(kwargs)
        inputs = args[0] if args else kwargs['inputs']
        if isinstance(inputs, tuple):
            inputs = list(inputs)
        elif not isinstance(inputs, list) and not getattr(inputs, '_is_lazy_collect', False):
            inputs = [inputs]
        if args:
            args[0] = inputs
        else:
            kwargs['inputs'] = inputs
        return tuple(args), kwargs


    @staticmethod
    def _subset_dispatch(worker_count, worker_index, args, kwargs, **_):
        args, kwargs = list(args), dict(kwargs)
        inputs = args[0] if args else kwargs['inputs']
        batch = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
        size, remainder = divmod(len(batch), worker_count)
        start = worker_index * size + min(worker_index, remainder)
        stop = start + size + (worker_index < remainder)
        shard = batch[start:stop]
        for index, value in enumerate(args):
            if index > 0 and isinstance(value, list) and len(value) == len(batch):
                args[index] = value[start:stop]
        for key, value in kwargs.items():
            if key != 'inputs' and isinstance(value, list) and len(value) == len(batch):
                kwargs[key] = value[start:stop]
        if args:
            args[0] = shard
        else:
            kwargs['inputs'] = shard
        return tuple(args), kwargs


    @staticmethod
    @staticmethod
    async def _gather_generation_tasks(coroutines):
        """一个输入失败或取消后，等同批次的全部生成协程结束再释放 submission。"""
        tasks = []
        try:
            for coroutine in coroutines:
                tasks.append(asyncio.create_task(coroutine))
            # 父任务取消不自动再次取消正在执行清理的子任务。
            return await asyncio.shield(asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                if not task.done() and not task.cancelling():
                    task.cancel()
            cleanup = asyncio.gather(*tasks, return_exceptions=True)
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
            cleanup.result()
            raise

    def _submit_in_loop(self, coro, *, submission_id=None) -> Future:
        if submission_id is None:
            return asyncio.run_coroutine_threadsafe(coro, self._async_loop)

        future = Future()

        def completed(task):
            self._running_generation_submissions.discard(submission_id)
            if task.cancelled():
                future.cancel()
                return
            error = task.exception()
            try:
                if error is None:
                    future.set_result(task.result())
                else:
                    future.set_exception(error)
            except InvalidStateError:
                # Future 的取消可先于 Task 的清理完成；仍须取出 Task 的异常。
                pass

        async def schedule():
            try:
                task = self._async_loop.create_task(coro)
            except BaseException:
                coro.close()
                raise
            self._running_generation_submissions.add(submission_id)
            task.add_done_callback(completed)

            def cancel_task():
                if not task.done() and not task.cancelling():
                    task.cancel()

            def cancellation_requested(result):
                if result.cancelled() and not task.done():
                    self._async_loop.call_soon_threadsafe(cancel_task)

            future.add_done_callback(cancellation_requested)

        scheduling = schedule()
        try:
            scheduled = asyncio.run_coroutine_threadsafe(scheduling, self._async_loop)
        except BaseException:
            scheduling.close()
            coro.close()
            raise
        try:
            scheduled.result()
        except BaseException:
            future.cancel()
            raise
        return future


    def _check_generation_control(self):
        if getattr(self, '_admission_owned_elsewhere', False):
            raise RuntimeError('Adaptive generation control belongs to the original sampler driver.')


    def _generation_owner(self, submission_id):
        self._check_generation_control()
        owner = self._generation_owners.get(submission_id)
        if owner is None:
            raise KeyError(f'unknown generation submission: {submission_id}')
        return owner


    def _release_generation_owner(self, submission_id, owner):
        with self._worker_lock:
            if self._generation_owners.get(submission_id) is not owner:
                return
            self._generation_owners.pop(submission_id)
            worker = self._sampler_workers[owner.worker_id]
            worker.inflight -= 1
            elapsed = time.monotonic() - owner.submitted_at
            worker.service_time += elapsed
            self.service_time += elapsed


    def _reap_generation_owners(self):
        with self._worker_lock:
            owners = tuple(self._generation_owners.items())
        for submission_id, owner in owners:
            with self._worker_lock:
                if self._generation_owners.get(submission_id) is not owner:
                    continue
                collection = owner.collection
                if owner.submitting or collection == 'submitting':
                    continue
                needs_status = collection is not None and not (
                    hasattr(collection, 'ready') and not collection.ready())
                cancelling = owner.cancelling and collection is None
            # The status/cancel RPCs are issued without the worker lock: on a serial
            # actor their replies sit behind an entire generation otherwise, and every
            # dispatch on this sampler would wait for that.
            if needs_status:
                status = self._get_generation_status(submission_id, _worker_ids=(owner.worker_id,))
                if status['status'] == 'missing':
                    self._release_generation_owner(submission_id, owner)
            elif cancelling:
                status = self._cancel_generation(submission_id, _worker_ids=(owner.worker_id,))
                if status['status'] in ('missing', 'cancelled', 'completed'):
                    self._release_generation_owner(submission_id, owner)


    def submit_generation(self, submission_id: str, inputs: Any, sampling_params=None, adapter_name='', adapter_path=None,
                          *, use_base_model=False, allow_partial_rollout=False, target_worker=None):
        self._check_generation_control()
        kwargs = dict(use_base_model=use_base_model, allow_partial_rollout=allow_partial_rollout)
        if not hasattr(self, '_sampler_workers'):
            return self._submit_generation(submission_id, inputs, sampling_params, adapter_name, adapter_path,
                                           target_worker=target_worker, **kwargs)
        self._reap_generation_owners()
        with self._worker_lock:
            if submission_id in self._generation_owners:
                raise KeyError(f'generation submission already exists: {submission_id}')
            candidates = self._sampling_worker_ids()
            if not candidates:
                raise RuntimeError('No ACTIVE sampler worker is available for generation admission.')
            if target_worker is not None:
                if type(target_worker) is not int or not 0 <= target_worker < len(candidates):
                    raise ValueError(f'target_worker must be in [0, {len(candidates)}), got {target_worker!r}.')
                worker_id = candidates[target_worker]
            else:
                worker_id = self._worker_pool.select_worker('submit_generation', (submission_id, inputs),
                                                            dict(adapter_name=adapter_name))
                if worker_id is None:
                    raise RuntimeError('No ACTIVE sampler worker is available for generation admission.')
            owner = GenerationOwner(worker_id, time.monotonic())
            self._generation_owners[submission_id] = owner
            self._sampler_workers[worker_id].inflight += 1
            owner.submitting = True
        try:
            return self._submit_generation(submission_id, inputs, sampling_params, adapter_name, adapter_path,
                                           target_worker=0, _worker_ids=(worker_id,), **kwargs)
        except BaseException:
            with self._worker_lock:
                if self._generation_owners.get(submission_id) is owner:
                    owner.cancelling = True
            raise
        finally:
            with self._worker_lock:
                owner.submitting = False
                cancel_requested = self._generation_owners.get(submission_id) is owner and owner.cancelling
            if cancel_requested:
                try:
                    self._reap_generation_owners()
                except Exception:
                    # 控制 RPC 失败时保留 owner，后续排干操作继续确认真实取消状态。
                    pass


    def get_generation_status(self, submission_id: str):
        self._check_generation_control()
        if not hasattr(self, '_sampler_workers'):
            return self._get_generation_status(submission_id)
        with self._worker_lock:
            owner = self._generation_owner(submission_id)
            if owner.submitting:
                return {'submission_id': submission_id,
                        'status': 'cancellation_requested' if owner.cancelling else 'running'}
            return self._get_generation_status(submission_id, _worker_ids=(owner.worker_id,))


    def collect_generation(self, submission_id: str):
        self._check_generation_control()
        if not hasattr(self, '_sampler_workers'):
            from twinkle.utils.lazy_collect import get_lazy_collect
            result = self._collect_generation(submission_id)
            if not getattr(self, '_lazy_collect', get_lazy_collect()) and callable(result):
                return result()
            return result
        with self._worker_lock:
            owner = self._generation_owner(submission_id)
            if owner.submitting or owner.collection is not None or owner.cancelling:
                raise RuntimeError(f'generation submission is already being submitted or consumed: {submission_id}')
            status = self._get_generation_status(submission_id, _worker_ids=(owner.worker_id,))
            if status['status'] not in ('completed', 'failed', 'cancelled'):
                raise RuntimeError(f'generation submission is still running: {submission_id}')
            owner.collection = 'submitting'
        try:
            result = self._collect_generation(submission_id, _worker_ids=(owner.worker_id,))
        except BaseException:
            with self._worker_lock:
                owner.collection = None
                owner.cancelling = True
            try:
                self._reap_generation_owners()
            except Exception:
                pass
            raise
        with self._worker_lock:
            owner.collection = result
        self._reap_generation_owners()
        return result


    def cancel_generation(self, submission_id: str):
        self._check_generation_control()
        if not hasattr(self, '_sampler_workers'):
            return self._cancel_generation(submission_id)
        with self._worker_lock:
            owner = self._generation_owner(submission_id)
            if owner.collection is not None:
                raise RuntimeError(f'generation submission is already being consumed: {submission_id}')
            owner.cancelling = True
            if owner.submitting:
                return {'submission_id': submission_id, 'status': 'cancellation_requested'}
            result = self._cancel_generation(submission_id, _worker_ids=(owner.worker_id,))
            if result['status'] in ('missing', 'cancelled', 'completed'):
                self._release_generation_owner(submission_id, owner)
            return result


    def cancel_all_generations(self):
        self._check_generation_control()
        if not hasattr(self, '_sampler_workers'):
            return self._cancel_all_generations()
        with self._worker_lock:
            owners = tuple(self._generation_owners.items())
        cancelled = 0
        for submission_id, owner in owners:
            with self._worker_lock:
                if self._generation_owners.get(submission_id) is owner and owner.collection is None:
                    result = self.cancel_generation(submission_id)
                    cancelled += result['status'] == 'cancelled'
        self._reap_generation_owners()
        return {'submissions': len(owners), 'cancelled': cancelled}


    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def _generation_count(self):
        running = getattr(self, '_running_generation_submissions', ())
        for submission_id, future in tuple(self._generation_submissions.items()):
            if future.cancelled() and submission_id not in running:
                self._generation_submissions.pop(submission_id, None)
        return (len(self._generation_submissions) + len(running)
                + len(getattr(self, '_inflight_request_ids', ()))
                + getattr(self, '_active_partial_rollouts', 0))


    async def _generate_inputs(
        self,
        inputs: Any,
        sampling_params: SamplingParams | dict[str, Any] | None,
        *,
        adapter_name: str,
        adapter_path: str | None,
        use_base_model: bool,
        allow_partial_rollout: bool = False,
    ) -> list[SampleResponse]:
        """Run one admitted submission's generation on the engine. Supplied by each backend."""
        raise NotImplementedError(f'{type(self).__name__} must implement _generate_inputs')


    @remote_function(dispatch=_dispatch_generation, collect='none', lazy_collect=False)
    @timed
    def _submit_generation(
        self,
        submission_id: str,
        inputs: Any,
        sampling_params: SamplingParams | dict[str, Any] | None = None,
        adapter_name: str = '',
        adapter_path: str | None = None,
        *,
        use_base_model: bool = False,
        allow_partial_rollout: bool = False,
        target_worker: int | None = None,
    ) -> dict[str, Any]:
        """Submit a sampling shard without blocking the Ray actor.

        The generated responses stay local to this DP worker until :meth:`collect_generation` consumes
        them. This gives a driver-overlapped RL loop (and the HTTP data-plane service) fast admission:
        the call returns as soon as the work is scheduled on the sampler's background event loop, so the
        caller can go train the previous batch while this one generates.

        ``allow_partial_rollout`` makes each generation resumable: when an in-place weight republish aborts
        it (:func:`~twinkle.sampler.partial_rollout.abort_all_inflight`), the generation
        continues from the tokens it already produced on the fresh weights instead of returning truncated,
        so a deep buffer can overwrite the sampler's single live weight copy soundly. Off by default: an
        abort then simply ends the generation with whatever it produced.
        """
        if submission_id in self._generation_submissions:
            raise KeyError(f'generation submission already exists: {submission_id}')
        if not hasattr(self, '_running_generation_submissions'):
            self._running_generation_submissions = set()

        future = self._submit_in_loop(
            self._generate_inputs(
                inputs, sampling_params, adapter_name=adapter_name, adapter_path=adapter_path,
                use_base_model=use_base_model, allow_partial_rollout=allow_partial_rollout),
            submission_id=submission_id)
        self._generation_submissions[submission_id] = future
        return {'submission_id': submission_id, 'status': 'running'}


    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def _get_generation_status(self, submission_id: str) -> dict[str, Any]:
        """Return this DP worker's submission state without waiting."""
        future = self._generation_submissions.get(submission_id)
        if future is None:
            return {
                'submission_id': submission_id,
                'status': 'missing',
                'error': f'unknown generation submission: {submission_id}',
            }
        if future.cancelled():
            running = submission_id in getattr(self, '_running_generation_submissions', ())
            return {'submission_id': submission_id, 'status': 'cancellation_requested' if running else 'cancelled'}
        if not future.done():
            return {'submission_id': submission_id, 'status': 'running'}
        try:
            error = future.exception()
        except CancelledError:
            running = submission_id in getattr(self, '_running_generation_submissions', ())
            return {'submission_id': submission_id, 'status': 'cancellation_requested' if running else 'cancelled'}
        if error is not None:
            return {
                'submission_id': submission_id,
                'status': 'failed',
                'error': f'{type(error).__name__}: {error}',
            }
        return {'submission_id': submission_id, 'status': 'completed'}


    @remote_function(dispatch='all', collect='flatten', lazy_collect=True)
    @timed
    def _collect_generation(self, submission_id: str) -> list[SampleResponse]:
        """Consume completed responses as a LazyCollect without entering the driver."""
        future = self._generation_submissions.get(submission_id)
        if future is None:
            raise KeyError(f'unknown generation submission: {submission_id}')
        if not future.done() or (future.cancelled()
                                 and submission_id in getattr(self, '_running_generation_submissions', ())):
            raise RuntimeError(f'generation submission is still running: {submission_id}')
        try:
            return future.result()
        finally:
            self._generation_submissions.pop(submission_id, None)


    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def _cancel_generation(self, submission_id: str) -> dict[str, Any]:
        """请求取消；真正结束前保留 submission，供排干操作确认。"""
        future = self._generation_submissions.get(submission_id)
        if future is None:
            return {'submission_id': submission_id, 'status': 'missing'}
        was_done = future.done()
        cancelled = future.cancel()
        running = submission_id in getattr(self, '_running_generation_submissions', ())
        if running:
            status = 'cancellation_requested'
        elif cancelled or future.cancelled():
            status = 'cancelled'
        elif was_done:
            status = 'completed'
        else:
            status = 'cancellation_requested'
        if status != 'cancellation_requested':
            self._generation_submissions.pop(submission_id, None)
        return {
            'submission_id': submission_id,
            'status': status,
        }


    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    @timed
    def _cancel_all_generations(self) -> dict[str, int]:
        """取消所有 retained submissions，但不提前清除仍在执行的记录。"""
        submissions = tuple(self._generation_submissions)
        results = [self._cancel_generation(submission_id) for submission_id in submissions]
        return {'submissions': len(submissions), 'cancelled': sum(r['status'] == 'cancelled' for r in results)}

    # ------------------------------------------------------------------
    # Partial rollout: cooperative abort mid-decode and resumable generations.
    # The engine hooks (_abort_request / _sample_engine_request) are supplied
    # by each backend; the gate and merge logic here is backend-agnostic.
    # ------------------------------------------------------------------

    def _register_inflight(self, request_id: str) -> None:
        """Record a request id as in-flight so :meth:`abort_all_inflight` can interrupt it.

        Called from ``_sample_single`` on the sampler's background event loop; the registry is only ever
        touched from that single loop thread, so a plain set needs no locking.
        """
        ids = getattr(self, '_inflight_request_ids', None)
        if ids is None:
            ids = set()
            self._inflight_request_ids = ids
        ids.add(request_id)


    def _unregister_inflight(self, request_id: str) -> None:
        ids = getattr(self, '_inflight_request_ids', None)
        if ids is not None:
            ids.discard(request_id)


    async def _abort_request(self, request_id: str) -> None:
        """Abort one in-flight engine request; it then finishes with ``stop_reason='abort'`` + partials."""
        await self.engine.abort_request(request_id)


    async def _sample_engine_request(self, request_id: str, coroutine) -> SampleResponse:
        """取消调用方时显式 abort，并等真实引擎调用结束后才注销请求。"""
        try:
            await self._wait_generation_gate()
            task = asyncio.create_task(coroutine)
        except BaseException:
            coroutine.close()
            raise
        self._register_inflight(request_id)
        try:
            return await asyncio.shield(task)
        except BaseException:
            async def cleanup():
                try:
                    await self._abort_request(request_id)
                except BaseException as error:
                    logger.warning('取消引擎请求 %s 失败，等待请求自然结束：%s', request_id, error)
                await asyncio.gather(task, return_exceptions=True)

            cleanup_task = asyncio.create_task(cleanup())
            while not cleanup_task.done():
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    continue
            cleanup_task.result()
            raise
        finally:
            self._unregister_inflight(request_id)


    async def _abort_all_inflight(self) -> int:
        """Abort every in-flight request on this worker; returns how many abort calls were issued.

        Snapshots the registry first: a request may finish on its own between the snapshot and its abort,
        which the engine treats as a no-op, so a per-request failure is logged and skipped rather than
        aborting the whole sweep.
        """
        request_ids = list(getattr(self, '_inflight_request_ids', ()) or ())
        aborted = 0
        for request_id in request_ids:
            try:
                await self._abort_request(request_id)
                aborted += 1
            except Exception as error:
                logger.warning('partial rollout: failed to abort in-flight request %s: %s', request_id, error)
        return aborted


    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    @timed
    def abort_all_inflight(self) -> dict[str, int]:
        """Abort every in-flight generation on this DP worker (called on an in-place weight republish).

        The abort coroutines must run on the engine's own loop, so this schedules ``_abort_all_inflight``
        there and blocks the actor thread on it -- the same loop/thread split the rest of the submission
        API uses, and blocking here is what makes the republish wait until no generation is mid-decode
        before the weights are overwritten.
        """
        future = asyncio.run_coroutine_threadsafe(self._abort_all_inflight(), self._async_loop)
        return {'aborted': future.result()}


    def _ensure_pause_state(self) -> None:
        if getattr(self, '_generation_resume_gate', None) is None:
            self._generation_resume_gate = asyncio.Event()
            self._generation_resume_gate.set()
            self._generation_paused = False
            self._active_partial_rollouts = 0
            self._rollouts_waiting_at_gate = 0


    async def _wait_generation_gate(self) -> None:
        self._ensure_pause_state()
        if self._generation_resume_gate.is_set():
            return
        self._rollouts_waiting_at_gate += 1
        try:
            await self._generation_resume_gate.wait()
        finally:
            self._rollouts_waiting_at_gate -= 1


    async def _pause_generation(self) -> int:
        self._ensure_pause_state()
        if self._generation_paused:
            while (getattr(self, '_inflight_request_ids', None)
                   or self._rollouts_waiting_at_gate < self._active_partial_rollouts):
                await asyncio.sleep(0.001)
            return 0
        self._generation_paused = True
        self._generation_resume_gate.clear()
        aborted = await self._abort_all_inflight()
        # Every unfinished rollout must leave engine decode and stop at the closed resume gate. Requests that
        # finish naturally decrement _active_partial_rollouts instead and therefore do not delay the barrier.
        while (getattr(self, '_inflight_request_ids', None)
               or self._rollouts_waiting_at_gate < self._active_partial_rollouts):
            await asyncio.sleep(0.001)
        return aborted


    async def _resume_generation(self) -> None:
        self._ensure_pause_state()
        self._generation_paused = False
        self._generation_resume_gate.set()


    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    @timed
    def pause_generation(self) -> dict[str, int]:
        """Stop decode at a resumable barrier before a shared sampler releases its GPUs."""
        future = asyncio.run_coroutine_threadsafe(self._pause_generation(), self._async_loop)
        return {'aborted': future.result()}


    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def resume_generation(self) -> None:
        """Release requests paused by :meth:`pause_generation`; idempotent."""
        future = asyncio.run_coroutine_threadsafe(self._resume_generation(), self._async_loop)
        future.result()


    def _merge_partial_responses(
        self,
        responses: list[SampleResponse],
        *,
        stop_reason: Optional[str] = None,
    ) -> SampleResponse:
        """Concatenate the segments of one resumed generation into a single ``SampleResponse``.

        Tokens and per-token logprobs are joined in attempt order; the final segment's ``new_input_feature``
        (prompt + all generated tokens, re-run through the template) and ``routed_experts`` are carried so
        downstream training sees one contiguous sample. ``stop_reason`` overrides the last segment's reason
        -- used to report ``'length'`` when a resume hit the token budget mid-abort.
        """
        sequences = [response.sequences[0] for response in responses]
        tokens = [token for sequence in sequences for token in sequence.tokens]
        logprobs = [logprob for sequence in sequences for logprob in (sequence.logprobs or [])]
        final_sequence = sequences[-1]
        return SampleResponse(
            prompt_token_ids=responses[0].prompt_token_ids,
            sequences=[
                SampledSequence(
                    stop_reason=stop_reason or final_sequence.stop_reason,
                    tokens=tokens,
                    logprobs=logprobs,
                    decoded=self.template.decode(tokens),
                    new_input_feature=final_sequence.new_input_feature,
                    routed_experts=final_sequence.routed_experts,
                )
            ],
        )


    async def _run_partial_rollout(
        self,
        original_input: dict[str, Any],
        sampling_params: SamplingParams,
        *,
        run_attempt: Callable[[Any, SamplingParams, int], Awaitable[tuple[SampleResponse, Any]]],
        allow_partial_rollout: bool = False,
        max_retries: int = 0,
        retry_delay_s: float = 0.0,
    ) -> PartialRolloutOutcome:
        """Drive one sample's generate-until-finished loop, resuming across aborts when enabled.

        ``run_attempt(current_input, attempt_params, attempt_index)`` performs a single generation attempt
        and returns ``(response, attempt_state)``; it owns everything attempt-specific -- the engine call
        and, on the TransferQueue path, the policy acquire/release around it -- so this loop stays free of
        any control-plane or engine knowledge. On each attempt the token budget is reduced by what earlier
        segments already produced. A response whose ``stop_reason`` is ``'abort'``/``'error'`` is not final:
        with partial rollout on and tokens in hand, the loop records the segment, advances ``current_input``
        to ``new_input_feature`` and retries; otherwise it resets to the original input and retries. A clean
        stop ends the loop, merging any accumulated segments first.
        """
        current_input = original_input
        partial_responses: list[SampleResponse] = []
        partial_states: list[Any] = []
        generated_tokens = 0
        last_error: Optional[Exception] = None
        was_aborted = False
        resumed_partial_output = False
        attempts = 0
        retries_used = 0
        self._ensure_pause_state()
        self._active_partial_rollouts += 1
        try:
            while retries_used <= max_retries:
                await self._wait_generation_gate()
                attempt_params = copy(sampling_params)
                if allow_partial_rollout and attempt_params.max_tokens is not None:
                    attempt_params.max_tokens -= generated_tokens
                attempt = attempts
                attempts += 1
                handover_abort = False
                try:
                    response, attempt_state = await run_attempt(current_input, attempt_params, attempt)
                    sequence = response.sequences[0]
                except Exception as exc:
                    last_error = exc
                else:
                    if sequence.stop_reason not in {'abort', 'error'}:
                        if not allow_partial_rollout or not partial_responses:
                            return PartialRolloutOutcome(response, (attempt_state, ), attempts, was_aborted,
                                                         resumed_partial_output)
                        partial_responses.append(response)
                        partial_states.append(attempt_state)
                        return PartialRolloutOutcome(
                            self._merge_partial_responses(partial_responses), tuple(partial_states), attempts,
                            was_aborted, resumed_partial_output)

                    last_error = RuntimeError(f'generation stopped with {sequence.stop_reason}')
                    was_aborted = was_aborted or sequence.stop_reason == 'abort'
                    handover_abort = sequence.stop_reason == 'abort' and self._generation_paused
                    if allow_partial_rollout and sequence.tokens:
                        resumed_partial_output = True
                        partial_responses.append(response)
                        partial_states.append(attempt_state)
                        generated_tokens += len(sequence.tokens)
                        current_input = sequence.new_input_feature
                        if sampling_params.max_tokens is not None and generated_tokens >= sampling_params.max_tokens:
                            return PartialRolloutOutcome(
                                self._merge_partial_responses(partial_responses, stop_reason='length'),
                                tuple(partial_states), attempts, was_aborted, resumed_partial_output)
                    elif not allow_partial_rollout:
                        current_input = original_input

                # Resource handover is not a generation failure and does not consume the ordinary retry budget.
                if handover_abort:
                    continue
                retries_used += 1
                if retries_used <= max_retries:
                    await asyncio.sleep(retry_delay_s)

            error_detail = f'{type(last_error).__name__}: {last_error}'
            error = RuntimeError(f'generation failed after {attempts} attempts; last error: {error_detail}')
            raise error from last_error
        finally:
            self._active_partial_rollouts -= 1
