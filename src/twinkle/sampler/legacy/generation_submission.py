# Copyright (c) ModelScope Contributors. All rights reserved.
"""Engine-agnostic non-blocking generation submission for the core samplers.

A sampler's :meth:`sample` blocks the calling (driver) thread until every prompt finishes. Online RL with
an overlapped rollout -- generating batch ``N+1`` on the sampler actor while the trainer trains batch ``N``
-- needs the opposite: admit a generation without waiting, then collect it later. This capability adds exactly
that quartet -- :meth:`submit_generation` (schedules the work on the sampler's background event loop and
returns immediately), :meth:`get_generation_status`, :meth:`collect_generation`, and
:meth:`cancel_generation` / :meth:`cancel_all_generations` to drop an in-flight or a retained submission.

It is pure ``Future`` bookkeeping over the actor's event loop; only the actual generation is
engine-specific, so a concrete sampler supplies the single ``async def _generate_inputs(...)`` coroutine
(the async core its blocking :meth:`sample` also runs) and declares this capability. Because it lives here in the core
``twinkle.sampler`` package, both the plain :class:`~twinkle.sampler.vllm_sampler.vLLMSampler` /
:class:`~twinkle.sampler.sglang_sampler.SGLangSampler` (driver-overlapped RL) and the higher-level
``twinkle_agentic`` TransferQueue samplers (which re-export it) share one implementation, so a concurrency
fix never has to be mirrored across backends or layers.

The methods are a plain ``ASYNC_SUBMISSION_METHODS`` function bundle: an engine sampler declares it via
``_CAPABILITIES`` and ``remote_class`` attaches it, so no mixin base ever enters the engine's MRO.

Contract for a concrete subclass: be decorated with ``@remote_class()``, own a running background event
loop at ``self._async_loop``, initialise ``self._generation_submissions`` to a dict, and implement
``async def _generate_inputs(self, inputs, sampling_params, *, adapter_name, adapter_path,
use_base_model, allow_partial_rollout) -> list[SampleResponse]``.
"""
from __future__ import annotations

import asyncio
import time
import zlib
from dataclasses import dataclass
from concurrent.futures import CancelledError, Future, InvalidStateError
from typing import Any

from twinkle import remote_function
from twinkle.data_format import SampleResponse, SamplingParams
from twinkle.perf import timed


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


@dataclass
class GenerationOwner:
    worker_id: str
    submitted_at: float
    collection: Any = None
    cancelling: bool = False
    submitting: bool = False


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


# The capability bundle: engine samplers declare ``_CAPABILITIES = {**ASYNC_SUBMISSION_METHODS, ...}`` and
# ``remote_class`` attaches every function before its method scan. The two ``_subset_*``
# hooks are the subset-dispatch strategy the infra wrapper looks up by that generic name.
# ``_generate_inputs`` is NOT in the bundle: it is the engine-overridden contract hook each
# backend defines in its own class body (the module-level version above is the raising default).
ASYNC_SUBMISSION_METHODS = {
    '_submit_in_loop': _submit_in_loop,
    '_gather_generation_tasks': _gather_generation_tasks,
    '_check_generation_control': _check_generation_control,
    '_generation_owner': _generation_owner,
    '_release_generation_owner': _release_generation_owner,
    '_reap_generation_owners': _reap_generation_owners,
    'submit_generation': submit_generation,
    'get_generation_status': get_generation_status,
    'collect_generation': collect_generation,
    'cancel_generation': cancel_generation,
    'cancel_all_generations': cancel_all_generations,
    '_generation_count': _generation_count,
    '_submit_generation': _submit_generation,
    '_get_generation_status': _get_generation_status,
    '_collect_generation': _collect_generation,
    '_cancel_generation': _cancel_generation,
    '_cancel_all_generations': _cancel_all_generations,
    '_subset_normalize': staticmethod(_subset_normalize),
    '_subset_dispatch': staticmethod(_subset_dispatch),
}
