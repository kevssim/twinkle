# Copyright (c) ModelScope Contributors. All rights reserved.
"""Behaviour-pinning tests for the sampler admission protocol.

These exercise the driver-side orchestration that ``remote_function``'s wrapper
performs together with the ``Sampler`` hooks (``_begin_worker_call`` /
``_end_worker_call`` / configuration locking / recording / replay / drain
lifecycle) against real Ray actors -- no GPU, no engine, no ``remote_class``
construction. They pin the CURRENT behaviour so the worker-pool refactor can be
verified as behaviour-preserving.
"""
from __future__ import annotations

import threading
import time
import zlib

import pytest

ray = pytest.importorskip('ray')

import twinkle.infra as infra  # noqa: E402
from twinkle.infra import remote_function  # noqa: E402
from twinkle.infra.continuous_work import get_worker_id  # noqa: E402
from twinkle.infra.worker_pool import WorkerSlot, with_worker_pool  # noqa: E402
from twinkle.sampler.base import Sampler  # noqa: E402

PROBE_SLEEP = 0.4


@ray.remote(max_concurrency=4)
class _ProbeActor:
    """A stand-in engine actor: records timestamped events, sleeps on demand.

    ``max_concurrency`` mirrors a real sampler actor, where methods run side by
    side -- without it the actor's own serial queue would mask the driver-side
    barrier these tests exist to pin.
    """

    def __init__(self, name: str):
        self.name = name
        self.events = []
        self.pending = 0

    def probe(self, items):
        self.events.append(('probe', time.time(), len(items)))
        time.sleep(PROBE_SLEEP)
        return [(self.name, item) for item in items]

    def sel_probe(self, items):
        self.events.append(('sel_probe', time.time(), len(items)))
        time.sleep(PROBE_SLEEP)
        return [(self.name, item) for item in items]

    def _submit_generation(self, submission_id, inputs, sampling_params=None, adapter_name='',
                            adapter_path=None, **kwargs):
        self.events.append(('submit_generation', time.time(), submission_id))
        return {'submission_id': submission_id, 'status': 'running'}

    def _cancel_all_generations(self):
        self.events.append(('cancel_all_generations', time.time(), None))
        return {'submissions': 0, 'cancelled': 0}

    def _cancel_generation(self, submission_id):
        return {'submission_id': submission_id, 'status': 'missing'}

    def _twinkle_async_probe(self, items):
        return self.probe(items)

    def cw_probe(self, items):
        self.events.append(('cw_probe', time.time(), len(items)))
        time.sleep(0.1)
        self.events.append(('cw_probe_end', time.time()))
        return [(self.name, item) for item in items]

    def _twinkle_async_cw_probe(self, items):
        return self.cw_probe(items)

    def dp_probe(self, items):
        self.events.append(('dp_probe', time.time(), len(items)))
        return len(items)

    def slow_control(self, seconds):
        self.events.append(('slow_control', time.time()))
        time.sleep(seconds)
        return self.name

    def control_probe(self):
        self.events.append(('control', time.time()))
        return 'ok'

    def set_template(self, template_cls=None, **kwargs):
        self.events.append(('set_template_begin', time.time()))
        time.sleep(PROBE_SLEEP)
        self.events.append(('set_template_end', time.time()))
        return 'template-set'

    def pause_generation(self):
        return True

    def resume_generation(self):
        return True

    def _generation_count(self):
        return self.pending

    def _set_pending(self, count):
        self.pending = count
        return count

    def _events(self):
        return list(self.events)

    def shutdown(self):
        return True


class _ProbeSampler(Sampler):
    """Driver-side sampler whose hooks are the real ``Sampler`` protocol."""

    @remote_function(dispatch='slice', collect='flatten', lazy_collect=True)
    def probe(self, items):
        return []

    @remote_function(dispatch='slice', collect='flatten', enable_continous_work=True)
    def cw_probe(self, items):
        return []

    @remote_function(dispatch='slice_dp', collect='first', lazy_collect=False)
    def dp_probe(self, items):
        return 0

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def slow_control(self, seconds):
        return None

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def control_probe(self):
        return None

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def set_template(self, template_cls=None, **kwargs):
        return None

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def pause_generation(self):
        return True

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def resume_generation(self):
        return True

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def _generation_count(self):
        return 0

    def _pick_second(self):
        return [self._second_worker_id]

    def _reap_generation_owners(self):
        """No-op stand-in for the async-submission capability hook real samplers carry."""

    @remote_function(dispatch='slice', collect='flatten', lazy_collect=False, worker_selector='_pick_second')
    def sel_probe(self, items):
        return []

    def sample(self, inputs, sampling_params=None, adapter_name='', *, num_samples=1):  # abstract
        return []

    async def _generate_inputs(self, inputs, sampling_params=None, **_):
        return []  # engine-overridden contract hook, stubbed for the probe

    def apply_patch(self, patch_cls, **kwargs):  # abstract
        return None


# Tests are infra consumers, so they mount the pool facade explicitly instead of going
# through remote_class (whose construction path is bypassed by this hand-built driver).
# The submit protocol itself is inherited from the Sampler base class body.
with_worker_pool(_ProbeSampler)


@pytest.fixture(scope='module', autouse=True)
def _ray_and_ray_mode():
    ray.init(ignore_reinit_error=True, num_cpus=4, logging_level='ERROR')
    prev_mode = infra._mode
    infra._mode = 'ray'
    try:
        yield
    finally:
        infra._mode = prev_mode


def _install(driver, actor_names, *, state='ACTIVE'):
    """Lay down the admission state that ``configure_workers`` would create."""
    actors = [_ProbeActor.remote(name) for name in actor_names]
    driver._actors = tuple(actors)
    driver._generation_owners = {}
    driver._worker_pool._default_router._rotation = 0
    driver._worker_pool.install_slots(actors, tuple((index, ) for index in range(len(actors))), 'probe-group',
                                      state=state)
    driver._second_worker_id = get_worker_id(actors[1])
    return actors


def _probe_events(actor):
    return ray.get(actor._events.remote())


@pytest.fixture
def driver():
    sampler = _ProbeSampler()
    _install(sampler, ['w0', 'w1'])
    sampler._max_concurrency = None
    yield sampler
    for actor in sampler._actors:
        ray.kill(actor, no_restart=True)


# --- A: reservation bookkeeping -------------------------------------------------


def test_dispatching_settles_before_lazy_result_collected(driver):
    """The reservation is repaid when the wrapper returns, not when the lazy future resolves."""
    worker_ids = list(driver._sampler_workers)
    future = driver.probe(['a', 'b'])
    assert all(driver._sampler_workers[worker_id].dispatching == 0 for worker_id in worker_ids)
    refs = getattr(future, '_futures', None)
    assert refs, 'lazy collect must expose its futures'
    import ray as _ray
    ready, _ = _ray.wait(refs, num_returns=len(refs), timeout=0)
    assert ready == [], 'probe should still be sleeping at this point'
    output = future()
    assert sorted(output) == [('w0', 'a'), ('w1', 'b')]


def test_lock_not_held_while_result_waited(driver):
    """A plain call releases the worker lock at submission; waiting happens outside it."""
    holder = threading.Event()

    def hold():
        driver.slow_control(PROBE_SLEEP)
        holder.set()

    threading.Thread(target=hold, daemon=True).start()
    time.sleep(0.1)  # the slow call is now submitted and waiting on its result
    # the worker lock itself must be free (only dispatch, never the result wait, holds it);
    # the in-flight call still shows up through the reservation counters instead.
    assert driver._worker_lock.acquire(blocking=False), 'worker lock held while a result is being waited'
    driver._worker_lock.release()
    assert sum(worker.dispatching for worker in driver._sampler_workers.values()) >= 1
    assert not holder.is_set()
    assert holder.wait(timeout=10)  # drain background thread


def test_unknown_worker_ids_raise_and_release_lock(driver):
    """A bogus _worker_ids fails loudly and the next call still goes through."""
    with pytest.raises(ValueError, match='Unknown actor IDs'):
        driver.probe(['a'], _worker_ids=['bogus-id'])
    assert driver.control_probe() == 'ok'


# --- B: configuration serialization and recording --------------------------------


def test_configuration_recorded_without_worker_ids(driver):
    """Default-all configuration calls are recorded (replay feed); plain calls are not."""
    driver.set_template('qwen')
    recorded = driver._worker_pool.configurations
    assert recorded == [('set_template', ('qwen', ), {})]
    driver.control_probe()
    assert driver._worker_pool.configurations == recorded, 'non-configuration calls must not be recorded'
    driver.set_template('llama', path='x')
    assert driver._worker_pool.configurations[-1] == ('set_template', ('llama', ), {'path': 'x'})


def test_configuration_call_holds_lock_until_collected(driver):
    """A configuration call keeps the worker lock until its result is collected, blocking dispatch."""
    done = threading.Event()
    begin = threading.Event()

    def configure():
        begin.set()
        driver.set_template('qwen')
        done.set()

    threading.Thread(target=configure, daemon=True).start()
    assert begin.wait(timeout=5)
    time.sleep(0.1)  # the configuration call is now sleeping inside the actor
    # the submission part is over, yet the lock must still be held by the
    # configuration call -- only the collection at the end of the wrapper releases it
    assert not driver._worker_lock.acquire(blocking=False), \
        'worker lock must stay held by a configuration call until its result is collected'
    driver.control_probe()  # blocks until set_template finishes
    events = _probe_events(driver._actors[0])
    control = [e for e in events if e[0] == 'control'][-1]
    template_end = [e for e in events if e[0] == 'set_template_end'][-1]
    assert control[1] >= template_end[1], 'control probe ran while set_template was still in flight'
    assert done.wait(timeout=10)


# --- C: drain and lifecycle ------------------------------------------------------


def test_workers_idle_reflects_inflight_and_engine_counts(driver):
    worker_id = next(iter(driver._sampler_workers))
    driver._sampler_workers[worker_id].inflight = 2
    assert driver.workers_idle([worker_id]) is False
    driver._sampler_workers[worker_id].inflight = 0
    assert driver.workers_idle([worker_id]) is True
    ray.get(driver._actors[0]._set_pending.remote(1))
    assert driver.workers_idle([worker_id]) is False
    ray.get(driver._actors[0]._set_pending.remote(0))
    assert driver.workers_idle([worker_id]) is True


def test_remove_workers_requires_drained_state(driver):
    worker_id = next(iter(driver._sampler_workers))
    with pytest.raises(RuntimeError, match='Drain or pause'):
        driver.remove_workers([worker_id])
    driver.drain_workers([worker_id])
    assert driver.remove_workers([worker_id]) is True
    assert worker_id not in driver._sampler_workers
    assert get_worker_id(driver._actors[0]) != worker_id
    assert len(driver._actors) == 1


def test_pause_resume_generation_bookkeeping(driver):
    worker_id = next(iter(driver._worker_pool.workers))
    driver.pause_generation(_worker_ids=[worker_id])
    assert driver._worker_pool.paused_generation_workers == {worker_id}
    driver.resume_generation(_worker_ids=[worker_id])
    assert driver._worker_pool.paused_generation_workers == set()


# --- D: subset routing -----------------------------------------------------------


def test_explicit_worker_ids_route_to_one_worker(driver):
    target = get_worker_id(driver._actors[1])
    future = driver.probe(['a', 'b', 'c'], _worker_ids=[target])
    # subset + lazy collect returns the bare future (no driver lifecycle captured)
    output = future() if callable(future) and not isinstance(future, list) else future
    assert output == [('w1', 'a'), ('w1', 'b'), ('w1', 'c')]
    assert _probe_events(driver._actors[0]) == []
    assert len(_probe_events(driver._actors[1])) == 1


def test_subset_slice_dp_gets_all_inputs(driver):
    target = get_worker_id(driver._actors[0])
    output = driver.dp_probe(['a', 'b', 'c'], _worker_ids=[target])
    assert output == 3  # collect='first': the single worker's count
    assert len(_probe_events(driver._actors[0])) == 1
    assert _probe_events(driver._actors[1]) == []


def test_worker_selector_method_routes(driver):
    output = driver.sel_probe(['a', 'b'])
    assert output == [('w1', 'a'), ('w1', 'b')]
    assert _probe_events(driver._actors[0]) == []
    assert len(_probe_events(driver._actors[1])) == 1


# --- E: barrier -------------------------------------------------------------------


def test_configuration_waits_for_continuous_work(driver):
    driver._max_concurrency = 4
    done = threading.Event()
    submitted = threading.Event()

    def run_continuous():
        submitted.set()
        driver.cw_probe(['a', 'b', 'c', 'd'])
        done.set()

    threading.Thread(target=run_continuous, daemon=True).start()
    assert submitted.wait(timeout=5)
    time.sleep(0.05)  # the continuous requests are now in flight on both workers
    driver.set_template('qwen')
    for actor in driver._actors:
        events = _probe_events(actor)
        cw_end = [e for e in events if e[0] == 'cw_probe_end']
        template_begin = [e for e in events if e[0] == 'set_template_begin']
        if cw_end:
            assert template_begin, 'set_template never ran'
            assert template_begin[-1][1] >= cw_end[-1][1], \
                'set_template started on the actor while continuous work was still running'
    assert done.wait(timeout=10)


def test_worker_call_conflicts_predicate(driver):
    conflicts = driver._worker_pool.conflicts
    assert conflicts('pause_generation', 'sample', 'w0') is False
    assert conflicts('resume_generation', 'sample', 'w0') is False
    assert conflicts('abort_all_inflight', 'sample', 'w0') is False
    assert conflicts('set_template', 'sample', 'w0') is True
    assert conflicts('set_template', 'probe', 'w0') is True
    bare = _ProbeSampler()  # declared domain but no configured workers
    assert bare._worker_pool.conflicts('anything', 'sample', 'w0') is True


# --- F: routing policies -----------------------------------------------------------


def test_continuous_work_least_load_balance(driver):
    driver._max_concurrency = 4
    output = driver.cw_probe(['a', 'b', 'c', 'd'])
    assert len(output) == 4
    names = [name for name, _ in output]
    assert sorted(names) == ['w0', 'w0', 'w1', 'w1'], '4 requests must split 2/2 over 2 idle workers'


def test_router_extension_point(driver):
    """A pluggable Router wins over the built-in default, abstention falls back to it,
    and naming a non-admissible worker fails loudly."""
    import zlib

    from twinkle.infra.worker_pool import Router

    pool = driver._worker_pool
    workers = list(pool.sampling_worker_ids())

    class StickyRouter(Router):

        def select(self, method, args, kwargs, view):
            submission_id = args[0]
            return (workers[zlib.crc32(str(submission_id).encode('utf-8')) % len(workers)],)

    try:
        pool.router = StickyRouter()
        first = pool.select_worker('submit_generation', ('sid-router-a', [1]), {})
        second = pool.select_worker('submit_generation', ('sid-router-a', [2]), {})
        assert first == second, 'same submission id must stay sticky'
        assert first in workers

        class AbstainingRouter(Router):

            def select(self, method, args, kwargs, view):
                return None

        pool.router = AbstainingRouter()
        fallback = pool.select_worker('submit_generation', ('sid-router-c', [3]), {})
        assert fallback in workers, 'abstention falls back to the built-in default'

        class RogueRouter(Router):

            def select(self, method, args, kwargs, view):
                return ('nonexistent-worker',)

        pool.router = RogueRouter()
        with pytest.raises(RuntimeError, match='non-admissible worker'):
            pool.select_worker('submit_generation', ('sid-router-x', [4]), {})
    finally:
        pool.router = None


def test_dispatch_generation_sticky_for_small_batch():
    """Same submission id routes to one worker; the batch smaller than the DP size lands whole."""
    sticky_a = Sampler._dispatch_generation(2, 0, ('sid-a', ['x']), {})
    sticky_b = Sampler._dispatch_generation(2, 1, ('sid-a', ['x']), {})
    shards = [sticky_a[0][1], sticky_b[0][1]]
    assert sorted(shards, key=len) == [[], ['x']], 'one worker gets everything, the other nothing'
    # ids with different crc32 modulo route to different workers
    ids = ['sid-%d' % i for i in range(8)]
    mods = {i: zlib.crc32(sid.encode('utf-8')) % 2 for i, sid in enumerate(ids)}
    id_a = next(sid for i, sid in enumerate(ids) if mods[i] == 0)
    id_b = next(sid for i, sid in enumerate(ids) if mods[i] == 1)
    first_a = Sampler._dispatch_generation(2, 0, (id_a, ['x']), {})[0][1]
    first_b = Sampler._dispatch_generation(2, 0, (id_b, ['x']), {})[0][1]
    assert [first_a, first_b] in ([['x'], []], [[], ['x']])
    # a batch at least the worker count is split evenly
    a = Sampler._dispatch_generation(2, 0, ('sid-a', [0, 1, 2, 3, 4]), {})
    b = Sampler._dispatch_generation(2, 1, ('sid-a', [0, 1, 2, 3, 4]), {})
    assert (a[0][1], b[0][1]) == ([0, 1, 2], [3, 4])
    # explicit target_worker wins
    t0 = Sampler._dispatch_generation(2, 0, ('sid-a', ['x']), {'target_worker': 1})
    t1 = Sampler._dispatch_generation(2, 1, ('sid-a', ['x']), {'target_worker': 1})
    assert t0[0][1] == [] and t1[0][1] == ['x'], 'only the targeted worker receives the batch'
    assert 'target_worker' not in t0[1]


# --- G: serialization boundary -----------------------------------------------------


def test_submit_protocol_lives_on_class_body():
    """The async generation protocol is declared on the Sampler class body itself: grep-visible,
    IDE-navigable, present without any decoration -- no injection, no bundle dictionary."""
    for name in ('submit_generation', 'get_generation_status', 'collect_generation', 'cancel_generation',
                 'pause_generation', 'abort_all_inflight', '_dispatch_generation', '_subset_normalize',
                 '_subset_dispatch', '_generate_inputs'):
        assert name in vars(Sampler), f'{name} must be defined on the Sampler class body'
    # the scheduling metadata lists reference only worker-side names, all of which
    # the class body now defines directly
    for listed in ('_PAUSE_METHODS', '_CONTROL_METHODS'):
        for name in getattr(Sampler, listed):
            assert name in vars(Sampler), f'{name} listed in {listed} must exist on the class body'
    # the dispatch strategy is a plain static callable on the class, usable undecorated
    t = Sampler._dispatch_generation(1, 0, ('sid', ['x']), {})
    assert t[0][1] == ['x']


def test_remote_class_auto_attaches_pool_facade(driver):
    """remote_class decorates any pool-domain declaration with the facade, automatically."""
    from twinkle.infra import remote_class

    class _Declared:
        _REPLAY_METHODS = frozenset({'set_template'})

        def set_template(self, template_cls=None, **kwargs):
            return 'set'

    undeclared = type('Undeclared', (), {'set_template': lambda self, **kw: 'set'})
    decorated = remote_class()(_Declared)
    remote_class()(undeclared)  # no pool-domain declaration: no facade
    for name in ('configure_workers', 'add_workers', 'remove_workers', 'workers_idle', '_worker_ready'):
        assert hasattr(decorated, name), f'facade method {name} missing after decoration'
        assert not hasattr(undeclared, name), f'{name} must not leak onto undeclared classes'
    pool = _Declared()  # property access mounts the pool
    _ = pool._worker_pool
    assert isinstance(pool._worker_pool.workers, dict)


def test_getstate_strips_admission_state(driver):
    state = driver.__getstate__()
    for key in ('_worker_pool', '_generation_owners', '_continous_work_state', '_worker_factory'):
        assert key not in state, f'{key} must not cross the Ray boundary'
    assert state['_admission_owned_elsewhere'] is True


# --- H: pure helpers ---------------------------------------------------------------


def test_dispatch_sampling_inputs_shards():
    dispatch = _ProbeSampler._subset_dispatch
    args, kwargs = dispatch(3, 0, (['a', 'b', 'c', 'd', 'e'], ), {'tags': [1, 2, 3, 4, 5]})
    assert args[0] == ['a', 'b'] and kwargs['tags'] == [1, 2]
    args, kwargs = dispatch(3, 1, (['a', 'b', 'c', 'd', 'e'], ), {'tags': [1, 2, 3, 4, 5]})
    assert args[0] == ['c', 'd'] and kwargs['tags'] == [3, 4]
    args, kwargs = dispatch(3, 2, (['a', 'b', 'c', 'd', 'e'], ), {'tags': [1, 2, 3, 4, 5]})
    assert args[0] == ['e'] and kwargs['tags'] == [5]
    args, kwargs = dispatch(2, 0, ({'k': 1}, ), {})
    assert args[0] == [{'k': 1}], 'a single dict input is treated as a one-item batch'


def test_normalize_sampling_args():
    """The first positional argument is the request list; dict/tuple inputs become lists."""
    normalize = _ProbeSampler._subset_normalize
    args, kwargs = normalize(({'k': 1}, ), {})
    assert args[0] == [{'k': 1}]
    args, kwargs = normalize((({'a': 1}, {'b': 2}), ), {})
    assert args[0] == [{'a': 1}, {'b': 2}]
    args, kwargs = normalize((['a', 'b'], ), {})
    assert args[0] == ['a', 'b']
    args, kwargs = normalize((), {'inputs': {'k': 1}})
    assert kwargs['inputs'] == [{'k': 1}]


def test_configure_workers_validation(driver):
    fresh = _ProbeSampler()
    with pytest.raises(RuntimeError, match='original Ray driver handler'):
        fresh.configure_workers([0])
    fresh._actors = ()
    with pytest.raises(ValueError, match='non-empty sequence'):
        fresh.configure_workers([])
    with pytest.raises(ValueError, match='unique non-negative'):
        fresh.configure_workers([0, 0])
    driver._admission_owned_elsewhere = True
    try:
        with pytest.raises(RuntimeError, match='original Ray driver handler'):
            driver.configure_workers([0])
    finally:
        driver._admission_owned_elsewhere = False
