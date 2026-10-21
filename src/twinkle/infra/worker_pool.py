# Copyright (c) ModelScope Contributors. All rights reserved.
"""Driver-side admission ledger and lock protocol for a resident worker pool.

The component (e.g. a sampler handle) owns the pool instance; the infra dispatch
wrapper drives it through one attachment point (``self._worker_pool``). Which of
the component's method names are configuration / pause / control / generation
methods is declared by the component itself through class attributes -- this
module knows none of the names. The worker-side bookkeeping of continuous work
stays in ``continuous_work``; the pool only aggregates it into the routing view.
"""
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from twinkle.utils import get_logger
from .continuous_work import _cw_drop_locked, _select_workers, get_worker_id

logger = get_logger()


@dataclass
class WorkerSlot:
    """One resident engine actor: placement, admission state and request load."""

    worker_id: str
    actor: Any
    group: str
    ranks: Tuple[int, ...]
    shared: bool = False
    state: str = 'WARMING'  # WARMING / ACTIVE / DRAINING / PAUSED
    inflight: int = 0
    dispatching: int = 0
    service_time: float = 0.0


@dataclass(frozen=True)
class WorkerSlotView:
    """Immutable snapshot of one slot for routing decisions."""

    worker_id: str
    state: str
    shared: bool
    inflight: int
    dispatching: int
    load: int  # continuous-work requests charged and not yet returned


class WorkerPoolView:
    """A consistent read-only snapshot of the pool plus its continuous-work loads."""

    def __init__(self, slots: Tuple[WorkerSlotView, ...], suspended: Tuple[str, ...] = ()):
        self._slots = slots
        self._suspended = frozenset(suspended)
        self._by_id = {slot.worker_id: slot for slot in slots}

    @property
    def active_ids(self) -> Tuple[str, ...]:
        """Worker ids currently ACTIVE."""
        return tuple(slot.worker_id for slot in self._slots if slot.state == 'ACTIVE')

    @property
    def admissible_ids(self) -> Tuple[str, ...]:
        """Worker ids a router may pick: ACTIVE and not suspended."""
        return tuple(
            slot.worker_id for slot in self._slots
            if slot.state == 'ACTIVE' and slot.worker_id not in self._suspended)

    def slot(self, worker_id: str) -> Optional[WorkerSlotView]:
        return self._by_id.get(worker_id)

    def __iter__(self):
        return iter(self._slots)

    def __len__(self):
        return len(self._slots)


class Router:
    """Pluggable driver-side routing policy.

    ``select`` receives the full call context and a pool view, and returns the
    worker ids to route to, or ``None`` to abstain (fall back to the default
    dispatch over all workers). The default router always abstains, which is
    exactly the pre-router behaviour. Routers answer only "who receives the
    call"; the pool keeps answering "how to reach them safely".
    """

    def select(self, method: str, args: tuple, kwargs: dict, view: WorkerPoolView) -> Optional[Sequence[str]]:
        return None


class LeastInflightRouter(Router):
    """The pool's built-in default: rotate the starting point, then pick the admissible
    worker with the fewest in-flight requests. Carried over verbatim from the sampler
    admission policy; the rotation counter is router state, so replacing the router
    resets the rotation deliberately."""

    def __init__(self):
        self._rotation = 0

    def select(self, method: str, args: tuple, kwargs: dict, view: WorkerPoolView) -> Optional[Sequence[str]]:
        ids = view.admissible_ids
        if not ids:
            return None
        offset = self._rotation % len(ids)
        rotated = ids[offset:] + ids[:offset]
        winner = min(rotated, key=lambda worker_id: view.slot(worker_id).inflight)
        self._rotation = ids.index(winner) + 1
        return (winner,)


class WorkerPool:
    """Admission ledger, call reservation protocol and drain lifecycle of a worker pool.

    The lock protocol matches the wrapped dispatch exactly: selection, barrier
    and RPC submission share one critical section (``begin_call`` holds the lock
    and ``end_dispatch`` releases it once submission finished), while result
    collection always happens outside it. Which component methods are
    configuration or control methods is read from the owner's class attributes
    (``_REPLAY_METHODS`` / ``_PAUSE_METHODS`` / ``_CONTROL_METHODS`` /
    ``_GENERATION_METHODS``), all defaulting to empty for components that have
    no such domain.
    """

    def __init__(self, owner: Any, *, router: Optional[Router] = None):
        self._owner = owner
        self._lock = threading.RLock()
        self._default_router = LeastInflightRouter()
        self.router = router
        self.workers: Dict[str, WorkerSlot] = {}
        self.suspended_workers = set()
        self.paused_generation_workers = set()
        self.creating_slots = set()
        self.admission_closed = False
        self.configurations: List[Tuple[str, tuple, dict]] = []
        self.service_time = 0.0
        self.router = router

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @property
    def configured(self) -> bool:
        return bool(self.workers)

    # ---- component-declared domain (infra knows none of these names) ----

    @property
    def replay_methods(self) -> frozenset:
        return frozenset(getattr(type(self._owner), '_REPLAY_METHODS', ()))

    @property
    def pause_methods(self) -> Dict[str, bool]:
        return dict(getattr(type(self._owner), '_PAUSE_METHODS', {}))

    @property
    def control_methods(self) -> frozenset:
        return frozenset(getattr(type(self._owner), '_CONTROL_METHODS', ()))

    @property
    def generation_methods(self) -> frozenset:
        return frozenset(getattr(type(self._owner), '_GENERATION_METHODS', ()))

    def view(self) -> WorkerPoolView:
        """Snapshot the pool (and its continuous-work loads) for a router."""
        state = getattr(self._owner, '_continous_work_state', None)
        load: Dict[str, int] = {}
        if state is not None:
            with state['lock']:
                load = dict(state['load'])
        with self._lock:
            slots = tuple(
                WorkerSlotView(slot.worker_id, slot.state, slot.shared, slot.inflight, slot.dispatching,
                               load.get(slot.worker_id, 0))
                for slot in self.workers.values())
            suspended = tuple(self.suspended_workers)
        return WorkerPoolView(slots, suspended)

    def select_worker(self, method: str, args: tuple, kwargs: dict) -> Optional[str]:
        """Single-worker routing decision, taken under the pool lock (reentrant: callers
        that already hold the worker lock simply nest).

        The pluggable ``self.router`` answers first; abstaining (``None`` or empty) falls
        back to the built-in least-inflight default. A router that names a worker which is
        not admissible fails loudly. Returns ``None`` only when no worker is admissible.
        """
        with self._lock:
            view = self.view()
            if not view.admissible_ids:
                return None
            selection = None
            if self.router is not None:
                selection = self.router.select(method, args, kwargs, view)
            if not selection:
                selection = self._default_router.select(method, args, kwargs, view)
            if not selection:
                return None
            admissible = set(view.admissible_ids)
            for worker_id in selection:
                if worker_id not in admissible:
                    raise RuntimeError(
                        f'router selected non-admissible worker {worker_id!r} for {method!r}')
            return selection[0]

    # ---- call reservation protocol (driven by the infra dispatch wrapper) ----

    def configuration_lock(self, method: str) -> Optional[threading.RLock]:
        """The lock serializing configuration calls against dispatch, or None."""
        if method in self.replay_methods or method in self.pause_methods:
            return self._lock
        return None

    def begin_call(self, worker_ids: Optional[Sequence[str]], worker_selector: Optional[str]) -> Optional[Tuple[str]]:
        """Reserve the target workers: one critical section for selection, barrier and submission.

        Returns the reserved worker ids (dispatching charged), or ``None`` when
        the pool has no configured workers -- the wrapper then takes the plain
        non-reserved path.
        """
        if not self.workers:
            return None
        self._lock.acquire()
        try:
            if worker_ids is None and worker_selector is not None:
                worker_ids = getattr(self._owner, worker_selector)()
            actors = _select_workers(self._owner, worker_ids)
            worker_ids = tuple(get_worker_id(actor) for actor in actors)
            for worker_id in worker_ids:
                self.workers[worker_id].dispatching += 1
            return worker_ids
        except BaseException:
            self._lock.release()
            raise

    def end_dispatch(self) -> None:
        """Release the critical section once RPC submission finished."""
        self._lock.release()

    def end_call(self, worker_ids: Sequence[str]) -> None:
        with self._lock:
            for worker_id in worker_ids:
                slot = self.workers.get(worker_id)
                if slot is not None:
                    slot.dispatching -= 1

    def call_completed(self, method: str, worker_ids: Sequence[str]) -> None:
        with self._lock:
            enter = self.pause_methods.get(method)
            if enter is None:
                return
            if enter:
                self.paused_generation_workers.update(worker_ids)
            else:
                self.paused_generation_workers.difference_update(worker_ids)

    def conflicts(self, current_method: str, pending_method: str, worker_id: str) -> bool:
        """Whether the barrier must wait for a pending method on this worker.

        Only a pending generation method can be skipped past (by control methods
        or on paused workers); anything else in flight is waited for.
        """
        if pending_method not in self.generation_methods:
            return True
        return (current_method not in self.control_methods
                and worker_id not in self.paused_generation_workers)

    def settle_controls(self, method: str, submissions) -> None:
        """After a dispatch or collect failure, confirm each issued pause/resume RPC."""
        if method not in self.pause_methods:
            return
        import ray

        for worker_id, ref in submissions:
            try:
                ray.get(ref)
            except BaseException:
                logger.warning('Pause/resume RPC %s did not complete on worker %s.', method, worker_id, exc_info=True)
            else:
                self.call_completed(method, (worker_id, ))

    def record_configuration(self, method: str, args: tuple, kwargs: dict) -> None:
        """Append a configuration call to the replay feed for future workers."""
        if method not in self.replay_methods:
            return
        with self._lock:
            self.configurations.append((method, tuple(args), dict(kwargs)))

    # ---- drain and lifecycle ----

    def worker_ids(self, *, shared: Optional[bool] = None, states: Optional[Sequence[str]] = None) -> Tuple[str]:
        with self._lock:
            return tuple(worker_id for worker_id, slot in self.workers.items()
                         if (shared is None or slot.shared == shared)
                         and (states is None or slot.state in states))

    def worker_ranks(self, worker_ids: Sequence[str]) -> Tuple[Tuple[int, ...]]:
        with self._lock:
            return tuple(self.workers[worker_id].ranks for worker_id in worker_ids)

    def sampling_worker_ids(self) -> Optional[Tuple[str]]:
        """Worker ids admitted for new requests: ACTIVE and not suspended."""
        with self._lock:
            if self.admission_closed:
                raise RuntimeError('Worker pool admission is closed.')
            return tuple(worker_id for worker_id, slot in self.workers.items()
                         if slot.state == 'ACTIVE' and worker_id not in self.suspended_workers)

    def activate_workers(self, worker_ids: Sequence[str]) -> None:
        with self._lock:
            slots = [self.workers[worker_id] for worker_id in worker_ids]
            if any(slot.state == 'DRAINING' for slot in slots):
                raise RuntimeError('A draining worker cannot accept new requests.')
            for slot in slots:
                slot.state = 'ACTIVE'

    def drain_workers(self, worker_ids: Sequence[str]) -> None:
        with self._lock:
            for worker_id in worker_ids:
                self.workers[worker_id].state = 'DRAINING'

    def pause_workers(self, worker_ids: Sequence[str]) -> None:
        with self._lock:
            for worker_id in worker_ids:
                slot = self.workers[worker_id]
                if slot.state in ('ACTIVE', 'WARMING'):
                    slot.state = 'PAUSED'

    def suspend_admission(self, worker_ids: Sequence[str]) -> None:
        with self._lock:
            self.suspended_workers.update(worker_ids)

    def resume_admission(self, worker_ids: Sequence[str]) -> None:
        with self._lock:
            self.suspended_workers.difference_update(worker_ids)

    def close_admission(self) -> None:
        with self._lock:
            self.admission_closed = True

    def worker_inflight(self, worker_ids: Sequence[str]) -> Tuple[int]:
        with self._lock:
            return tuple(self.workers[worker_id].inflight for worker_id in worker_ids)

    def workers_idle(self, engine_count: Callable[[Sequence[str]], Any], worker_ids: Sequence[str]) -> bool:
        """Whether the workers may retire: control state only, never generation outputs.

        ``engine_count`` (a component-supplied probe, e.g. an in-flight engine
        count) is invoked outside the pool lock so a slow RPC cannot stretch the
        critical section.
        """
        import ray

        worker_ids = tuple(worker_ids)
        with self._lock:
            if any(self.workers[worker_id].inflight or self.workers[worker_id].dispatching
                   for worker_id in worker_ids):
                return False
        state = getattr(self._owner, '_continous_work_state', None)
        if state is not None:
            with state['lock']:
                refs = [ref for ref, (owner, _) in state['owners'].items() if owner in worker_ids]
            if refs:
                ready, _ = ray.wait(refs, num_returns=len(refs), timeout=0)
                if len(ready) != len(refs):
                    return False
                with state['lock']:
                    _cw_drop_locked(state, ready)
        counts = engine_count(worker_ids)
        counts = [counts] if isinstance(counts, int) else counts
        return all(count == 0 for count in counts)

    # ---- membership changes ----

    def reserve_slots(self, reservations: Sequence[Tuple[str, Tuple[int, ...]]]) -> None:
        """Claim placement slots for actors being constructed (conflicts raise)."""
        with self._lock:
            occupied = {(slot.group, slot.ranks) for slot in self.workers.values()}
            if self.admission_closed or set(reservations) & (occupied | self.creating_slots):
                raise RuntimeError('Worker pool is closing or a selected worker slot is occupied.')
            self.creating_slots.update(reservations)

    def release_slots(self, reservations: Sequence[Tuple[str, Tuple[int, ...]]]) -> None:
        with self._lock:
            self.creating_slots.difference_update(reservations)

    def install_slots(self, actors: Sequence[Any], ranks_per_actor: Sequence[Tuple[int, ...]], group: str, *,
                      shared: bool = False, state: str = 'WARMING') -> None:
        with self._lock:
            for actor, ranks in zip(actors, ranks_per_actor):
                worker_id = get_worker_id(actor)
                self.workers[worker_id] = WorkerSlot(worker_id, actor, group, ranks, shared=shared, state=state)

    def forget_workers(self, worker_ids: Sequence[str]) -> None:
        """Drop workers from the ledger after their actors were terminated.

        Runs in one critical section and re-checks ``dispatching`` so a worker
        that received a new reservation during a non-atomic drain fails loudly
        instead of being dropped with live callers.
        """
        worker_ids = set(worker_ids)
        with self._lock:
            if any(self.workers[worker_id].dispatching for worker_id in worker_ids if worker_id in self.workers):
                raise RuntimeError('Worker received a new dispatch while being removed.')
            for worker_id in worker_ids:
                self.workers.pop(worker_id, None)
                self.suspended_workers.discard(worker_id)
                self.paused_generation_workers.discard(worker_id)
            state = getattr(self._owner, '_continous_work_state', None)
            if state is not None:
                with state['lock']:
                    refs = [ref for ref, (owner, _) in state['owners'].items() if owner in worker_ids]
                    _cw_drop_locked(state, refs)
                    for worker_id in worker_ids:
                        state['load'].pop(worker_id, None)


_POOL_FACADE_FLAG = '_worker_pool_facade_attached'


def with_worker_pool(cls) -> type:
    """Give a component class the worker-pool facade: lifecycle methods, compatibility
    views and serialization hygiene, all injected here so the component's own source
    stays free of any infra knowledge.

    The class only declares its domain (``_REPLAY_METHODS`` etc.); everything the
    dispatch wrapper and the component family's historical private names need is
    attached by this function. ``remote_class`` invokes it automatically for any
    class that declares a pool domain, before the method scan, so attached methods
    (``_worker_ready`` included) are collected like any other method.
    """
    if getattr(cls, _POOL_FACADE_FLAG, False):
        return cls

    def _pool_getter(self) -> WorkerPool:
        pool = self.__dict__.get('_worker_pool')
        if pool is None:
            pool = WorkerPool(self)
            # setdefault is atomic: concurrent first accesses still converge on one pool
            self.__dict__.setdefault('_worker_pool', pool)
            pool = self.__dict__['_worker_pool']
        return pool

    setattr(cls, '_worker_pool', property(_pool_getter))

    def _workers_getter(self):
        # Historical family name for the admission ledger; raises while unconfigured so
        # ``hasattr`` keeps meaning "configure_workers ran".
        pool = _pool_getter(self)
        if not pool.configured:
            raise AttributeError('_sampler_workers')
        return pool.workers

    setattr(cls, '_sampler_workers', property(_workers_getter))

    def _lock_getter(self):
        return _pool_getter(self).lock

    setattr(cls, '_worker_lock', property(_lock_getter))

    orig_getstate = cls.__dict__.get('__getstate__')

    def _getstate(self):
        state = orig_getstate(self) if orig_getstate is not None else self.__dict__.copy()
        if hasattr(self, '_actors'):
            # The remote payload only needs the handler's serializable metadata;
            # admission and in-flight state stay with the original driver.
            for key in ('_worker_pool', '_continous_work_state', '_generation_owners', '_worker_factory'):
                state.pop(key, None)
            state['_admission_owned_elsewhere'] = True
        return state

    setattr(cls, '__getstate__', _getstate)

    def configure_workers(self, ranks, *, group=None, shared=False, state='WARMING'):
        """Register existing actors on the driver; no training mesh or engine-internal change."""
        if not hasattr(self, '_actors') or getattr(self, '_admission_owned_elsewhere', False):
            raise RuntimeError('Sampler worker lifecycle requires its original Ray driver handler.')
        pool = _pool_getter(self)
        if pool.configured:
            raise RuntimeError('Sampler workers have already been configured.')
        ranks = tuple(ranks)
        if not ranks or any(type(rank) is not int or rank < 0 for rank in ranks) or len(set(ranks)) != len(ranks):
            raise ValueError('Sampler physical ranks must be a non-empty sequence of unique non-negative integers.')
        from twinkle.infra._ray import RayHelper
        width = RayHelper.resource_manager.get_config(group or self.remote_group).gpus_per_worker
        if not self._actors or len(ranks) != len(self._actors) * width:
            raise ValueError('Physical ranks must match gpus_per_worker for every complete sampler actor slot.')
        if state not in ('WARMING', 'ACTIVE', 'DRAINING', 'PAUSED'):
            raise ValueError(f'Unknown sampler worker state: {state!r}.')
        # Family-local submission state, initialized here to keep the historical protocol
        # (generation-submission bookkeeping keys) unchanged.
        self._generation_owners = {}
        pool.install_slots(self._actors, tuple(ranks[index * width:(index + 1) * width] for index in
                                               range(len(self._actors))),
                           group or self.remote_group, shared=shared, state=state)

    setattr(cls, 'configure_workers', configure_workers)

    def add_workers(self, group, ranks, *, worker_slots=None, shared=False, engine_args=None):
        """Create actors from fully reserved slots; they do not admit requests until configured."""
        from twinkle.infra._ray import RayHelper
        from twinkle.utils import DeviceMesh

        if not hasattr(self, '_sampler_workers'):
            raise RuntimeError('Call configure_workers before adding sampler workers.')
        pool = _pool_getter(self)
        config = RayHelper.resource_manager.get_config(group)
        placements = RayHelper.resource_manager.get_group(group)
        width = getattr(config, 'gpus_per_worker', 1)
        ranks = tuple(ranks)
        if any(type(rank) is not int or rank < 0 for rank in ranks) or len(set(ranks)) != len(ranks):
            raise ValueError('Sampler physical ranks must be unique non-negative integers.')
        if len(ranks) != len(placements) * width:
            raise ValueError('Physical ranks must match the reserved sampler worker slots.')
        slots = tuple(range(len(placements))) if worker_slots is None else tuple(worker_slots)
        if any(type(slot) is not int or not 0 <= slot < len(placements) for slot in slots) or len(set(slots)) != len(slots):
            raise ValueError(f'Invalid sampler worker_slots: {slots!r}.')
        if not slots:
            return ()
        selected_ranks = tuple(ranks[slot * width:(slot + 1) * width] for slot in slots)
        reservations = {(group, slot_ranks) for slot_ranks in selected_ranks}
        pool.reserve_slots(reservations)
        actors = []
        worker_ids = ()
        try:
            cls_, args, kwargs, torch_kwargs = self._worker_factory
            kwargs = dict(kwargs, remote_group=group, device_mesh=DeviceMesh.from_sizes(dp_size=len(slots)))
            if engine_args is not None:
                kwargs['engine_args'] = dict(kwargs.get('engine_args', {}), **engine_args)
            actors = RayHelper.create_workers(
                cls_, group, 'all', *args, instance_id=uuid.uuid4().hex,
                extra_torch_kwargs=torch_kwargs, max_concurrency=self._max_concurrency,
                worker_slots=list(slots), **kwargs)
            worker_ids = tuple(get_worker_id(actor) for actor in actors)
            with pool.lock:
                if pool.admission_closed:
                    raise RuntimeError('Sampler closed while new workers were being constructed.')
                self._actors = tuple(self._actors) + tuple(actors)
                pool.install_slots(actors, selected_ranks, group, shared=shared)
                self._worker_ready(_worker_ids=worker_ids)
                # Configuration calls are mutually exclusive with installing and replaying on
                # new actors, so no patch is missed or applied twice.
                for method, args, kwargs in pool.configurations:
                    result = getattr(self, method)(*args, _worker_ids=worker_ids, **kwargs)
                    if callable(result):
                        result()
                return worker_ids
        except BaseException:
            try:
                _discard_workers(self, actors)
            except Exception:
                logger.warning('Failed to fully shut down sampler actors after a construction error.',
                               exc_info=True)
            raise
        finally:
            pool.release_slots(reservations)

    setattr(cls, 'add_workers', add_workers)

    from twinkle.infra import remote_function  # noqa: E402  (import cycle is one-way: dispatch imports this module lazily at call time)

    def _worker_ready(self):
        return True

    setattr(cls, '_worker_ready', remote_function(dispatch='all', collect='none', lazy_collect=False)(_worker_ready))

    def worker_ids(self, *, shared=None, states=None):
        return _pool_getter(self).worker_ids(shared=shared, states=states)

    def worker_ranks(self, worker_ids):
        return _pool_getter(self).worker_ranks(worker_ids)

    def _sampling_worker_ids(self):
        if getattr(self, '_admission_owned_elsewhere', False):
            raise RuntimeError('Adaptive sampling must be admitted by its original driver, not a serialized handler.')
        pool = _pool_getter(self)
        if not pool.configured:
            return None
        return pool.sampling_worker_ids()

    def activate_workers(self, worker_ids):
        _pool_getter(self).activate_workers(worker_ids)

    def drain_workers(self, worker_ids):
        _pool_getter(self).drain_workers(worker_ids)

    def pause_workers(self, worker_ids):
        _pool_getter(self).pause_workers(worker_ids)

    def suspend_admission(self, worker_ids):
        _pool_getter(self).suspend_admission(worker_ids)

    def resume_admission(self, worker_ids):
        _pool_getter(self).resume_admission(worker_ids)

    def close_admission(self):
        _pool_getter(self).close_admission()

    def worker_inflight(self, worker_ids):
        getattr(self, '_reap_generation_owners', lambda: None)()
        return _pool_getter(self).worker_inflight(worker_ids)

    def inflight_count(self):
        return sum(self.worker_inflight(self.worker_ids()))

    def workers_idle(self, worker_ids):
        """Whether the workers may retire, judged from control state only."""
        return _pool_getter(self).workers_idle(
            lambda ids: self._generation_count(_worker_ids=ids), worker_ids)

    def remove_workers(self, worker_ids):
        worker_ids = tuple(worker_ids)
        pool = _pool_getter(self)
        with pool.lock:
            workers = [pool.workers[worker_id] for worker_id in worker_ids]
            if any(worker.state == 'ACTIVE' for worker in workers):
                raise RuntimeError('Drain or pause sampler workers before removing them.')
        # The idle probe issues remote calls; it runs outside the lock so a slow
        # engine RPC cannot stretch the critical section.
        if not self.workers_idle(worker_ids):
            return False
        _discard_workers(self, tuple(worker.actor for worker in workers))
        return True

    def _discard_workers(self, actors):
        """Terminate engines and actors; keep cleaning the rest if one close fails."""
        import ray

        actors = tuple(actors)
        worker_ids = {get_worker_id(actor) for actor in actors}
        error = None
        shutdown_refs = []
        for actor in actors:
            try:
                shutdown_refs.append(actor.shutdown.remote())
            except BaseException as exc:
                error = error or exc
        try:
            for ref in shutdown_refs:
                try:
                    ray.get(ref)
                except BaseException as exc:
                    error = error or exc
        finally:
            for actor in actors:
                try:
                    ray.kill(actor, no_restart=True)
                except BaseException as exc:
                    error = error or exc
            pool = self.__dict__.get('_worker_pool')
            if pool is not None:
                pool.forget_workers(worker_ids)
            self._actors = tuple(actor for actor in self._actors if get_worker_id(actor) not in worker_ids)
            if getattr(self, '_generation_owners', None) is not None:
                self._generation_owners = {
                    key: owner for key, owner in self._generation_owners.items() if owner.worker_id not in worker_ids
                }
        if error is not None:
            raise error

    def close_workers(self):
        """Only for terminating the whole sampler: close admission, cancel and clean up."""
        if getattr(self, '_admission_owned_elsewhere', False):
            raise RuntimeError('Sampler lifecycle belongs to its original driver handler.')
        if not hasattr(self, '_actors'):
            return self.shutdown()
        if hasattr(self, '_sampler_workers'):
            self.close_admission()
        try:
            cancel = getattr(self, 'cancel_all_generations', None)
            if self._actors and cancel is not None:
                cancel()
        finally:
            _discard_workers(self, tuple(self._actors))

    for name, func in [('worker_ids', worker_ids), ('worker_ranks', worker_ranks),
                       ('_sampling_worker_ids', _sampling_worker_ids),
                       ('activate_workers', activate_workers), ('drain_workers', drain_workers),
                       ('pause_workers', pause_workers), ('suspend_admission', suspend_admission),
                       ('resume_admission', resume_admission), ('close_admission', close_admission),
                       ('worker_inflight', worker_inflight), ('workers_idle', workers_idle),
                       ('remove_workers', remove_workers), ('_discard_workers', _discard_workers),
                       ('close_workers', close_workers)]:
        setattr(cls, name, func)
    setattr(cls, 'inflight_count', property(inflight_count))
    setattr(cls, _POOL_FACADE_FLAG, True)
    return cls
