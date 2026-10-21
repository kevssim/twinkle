# Copyright (c) ModelScope Contributors. All rights reserved.
"""Algorithm-agnostic per-sample streaming control flow for online RL.

:class:`StreamingDriver` replaces the retired batch-granular deep-buffer skeleton with a per-sample one:
trajectories are
admitted ONE AT A TIME (each an independent sampler submission), completions are polled as-completed
into an in-process ready buffer, and the consumer trains WHATEVER IS READY instead of waiting for a
fixed batch to finish together. A slot freed by a completion is immediately backfilled from the prompt
stream, so the engine stays loaded and no straggler tail idles it -- at every staleness setting.

sync and async are not separate code paths here: they are points on the ``max_staleness`` knob of the
driver's inline version accounting (0 = synchronous: the admission window closes while its samples
train; 1 = one-step overlap; >1 = deep buffer). The driver never branches on the regime.

Control-plane mapping (the load-bearing design decision):

* ONE PARTITION = ONE PUBLISH CYCLE. A publication is triggered EXTERNALLY, via :meth:`request_publish` --
  the driver owns neither the cadence nor the weight push; the controller decides WHEN and supplies the push.
  A partition does NOT declare fixed members -- it is an admission WINDOW opened via
  ``request_rollout_partition`` (which is
  where the staleness gate lives: at most ``max_staleness + 1`` windows live at once) and closed by the
  publish that ends its cycle. Samples flow through the shared ready buffer in completion order, so a
  straggler never blocks a window: it lands in a later one and, if by then its version lag exceeds
  ``max_staleness``, the post-publish stale scan drops it.
* The POLICY VERSION bumps once per publish (``on_partition_trained``), never per optimizer step, so a
  sample's lag (``current_version - record.version``) is measured in publish cycles exactly as
  ``max_staleness`` is defined. Windows are published oldest-live
  first, which satisfies ``on_partition_training_started``'s FIFO requirement because windows open (and
  therefore age) in step order.
* Backpressure is a sample-count cap, ``buffer_depth``: admission pauses while
  ``in_flight + ready >= buffer_depth``. The consumer sizes the cap so one window's fill
  (``groups_per_partition * num_generations``) is at least one full training pull -- otherwise the
  driver could deadlock with a buffer too shallow to assemble a batch while admission is capped.

Injected seams (the external contract a consumer wires up; the driver knows no algorithm). The four
engine seams (submit/poll/collect/cancel) may instead be omitted together with a ``rollout`` object
exposing ``admit``/``poll``/``collect``/``cancel``, which default-binds them (an explicit seam wins):

* ``submit(prompt, trajectory_idx, policy) -> handle``: admit ONE trajectory without blocking, pinned to
  ``policy`` (its ``adapter_path`` travels with the submission under snapshot publication; ``None`` under
  in-place). ``prompt`` is the :class:`PromptRef` the driver drew: ``prompt.row`` is the opaque prompt payload
  the dataloader yielded and ``prompt.ordinal`` is the driver-assigned per-draw id to stamp as ``prompt_id``.
  The consumer builds the trajectory from ``prompt.row`` directly -- never an index lookup into a resident
  prompt table.
* ``poll(handles) -> completed_handles``: non-blocking as-completed query over the in-flight handles
  (a subset of ``handles``; unknown values are ignored).
* ``collect(handle) -> sample``: fetch one completed trajectory's training sample.
* ``assembly_ready(buffer) -> Optional[List[BufferRecord]]``: the algorithm-specific pull rule --
  inspect the ready buffer and, when it holds at least one full training unit (GRPO/RFT: whole groups,
  since every sibling of a group is admitted together under one policy and therefore shares ONE
  version, so a group is complete or untrainable; PPO/GKD: individual samples), remove and return
  exactly the records of one batch (an exact multiple of the consumer's optimizer-step batch, so the
  exact-global-batch invariant holds); ``None`` when not ready. Records left over stay buffered.
* ``consume(records) -> Optional[int]`` (OPTIONAL): score and train the pulled batch (the existing
  ``_train_rollout_batch`` half of a dev loop). Returns the number of OPTIMIZER STEPS it ran (a falsy
  or ``None`` return counts as one), which the external controller counts toward its publish cadence.
  Providing it selects the
  BLOCKING regime (:meth:`run` trains in-process -- the PPO/RFT/distill path). Omitting it
  (``consume=None``) selects the ITERATOR regime: :meth:`run` is bridged onto a background thread and
  the assembled batches are yielded by ``for batch in driver:``, the recipe trains each and calls
  :meth:`commit` to report its real optimizer steps (which become the bridge's consume return; the
  controller requests a publish against that count, so a version bump still lands only after training).
  The three iterator-regime seams below are then how the
  algorithm plugs in without the driver knowing it.
* ``score(sample) -> Optional[sample]`` (optional, iterator regime): per-SAMPLE scoring (PRM/ORM) applied
  the instant a trajectory is collected, BEFORE it enters the ready buffer, so both the group barrier and
  the assembler see a scored sample. A returned value replaces the sample; returning ``None`` keeps it
  (an in-place seam). Not per-group -- group-relative advantage is the assembler's job.
* ``assemble(records) -> batch`` (OPTIONAL OVERRIDE, iterator regime): replace the driver's FIXED output
  format -- the bare list of scored samples -- with a custom batch object. Default (``None``): each
  pulled batch is yielded as ``List[RolloutSample]`` with ``group_key`` / ``policy_version`` stamped by
  the kernel and ``orm_reward`` / ``prm_process`` / ``baseline`` written by the scoring hooks. Only a
  caller that must train on a different container passes this.
* ``group_complete(group_key, records) -> GroupDecision`` (DEPRECATED, dev-compat only): the one-shot
  intra-group barrier hook. Its three former jobs (PRM scoring, the variance verdict, the remax
  baseline) all have dedicated kernel parameters now (``score_worker``'s PRM channel, ``regenerate``,
  ``baseline_fn``); passing BOTH the hook and any of those is rejected. Kept only for the frozen dev
  builders; new callers use the decomposed parameters. The returned decision is ``ACCEPT`` (train
  as-is), ``AUGMENT`` (same as ACCEPT) or ``REGENERATE`` (drop the group and re-admit its prompt
  under the current version, bounded by ``max_group_regenerations``, past which it is accepted).
* ``regenerate(group_key, samples) -> bool`` (optional): the variance-style verdict -- the kernel calls it
  once per completed group AFTER both scoring channels; ``True`` drops the group and re-admits its
  prompt under the current version (same REGENERATE path and ``max_group_regenerations`` cap as the
  deprecated hook). ``None`` (default): no group is ever regenerated.
* ``baseline_fn(group_key, samples) -> Optional[float]`` (optional): the remax-style group baseline --
  its return value is written onto every sample in the group (``sample.baseline``). ``None`` (default):
  no baseline is written.
* ``cancel(handle)``: drop an in-flight generation (budget hit, drain, or error).
* ``score_worker`` (OPTIONAL collaborator, not a callable): when given, the kernel submits BOTH scoring
  channels PER SAMPLE at collection time -- non-blocking, one submission per poll batch, so scoring runs
  continuously alongside generation and training instead of waiting for a group to complete or for the
  driver thread to be free. Finished results are reaped onto the samples each pass (``sample.orm_reward``
  rows; plus the per-token process rewards written onto ``sample.prm_process`` when the worker carries a
  wired PRM channel), and the group barrier additionally requires a group's scores to be reaped before
  judging it; the reap wait is what ``stats.score_time`` measures. That channel's consumer may be an
  algorithm hook (``regenerate`` / ``baseline_fn`` / the deprecated ``group_complete``) or the training
  half reading ``sample.prm_process`` downstream. A pure-ORM worker scores the ORM channel alone.
* ``resource_adapter`` (OPTIONAL collaborator, not a callable): when given, its decision window is
  fed at every pull boundary (``observe``) and its scaling decision is produced and consumed
  inside the publish bracket (``decide`` + ``register_decision`` before the push, whose weight
  sync path applies the replica change).
* publication: NOT an injected seam -- it is an external SIGNAL. The controller calls
  :meth:`StreamingDriver.request_publish` with its own ``push(window) -> Optional[str]`` (a
  :class:`~twinkle_agentic.async_rl.weight_sync.WeightSyncStrategy.publish` callable); the driver brackets
  that push with its version/window accounting at the next pass boundary. A driver built with no control
  plane (``run_id=None``, the infer regime) pins the version at 0: no windows, no pins, and it never
  publishes.
* ``group_key(prompt) -> str`` (optional): the training-group identifier base stamped on every trajectory of
  one prompt, receiving the drawn :class:`PromptRef` (default ``str(prompt.ordinal)``, the driver-assigned
  per-draw id; the driver appends its ``#<group_seq>`` epoch namespace).
* ``prune()`` (optional): drop published artifacts no live version and no in-flight pin reference.
* ``initial_adapter_path()`` (optional): the version-0 pin (the pre-training adapter path for
  snapshot publication, ``None`` for in-place).
* ``reached_max()`` (optional): whether the optimizer-step budget is exhausted.
* ``sleep(seconds)`` (optional): idle wait injected for testability; defaults to ``time.sleep``.

Staleness enforcement: a record's version lag is bounded at TWO points, both counted in
``stats.dropped_stale`` (a dropped record's pin was already released at collect). (1) AT COLLECT: a
straggler can stay in flight across more than ``max_staleness`` publishes -- the post-publish scan never
sees the in-flight set -- so a completion is dropped the moment it is fetched if its lag already exceeds
the bound. This is what keeps a late arrival from ENTERING the buffer untrainable and being consumed
stale before the next scan could catch it. (2) AFTER EVERY PUBLISH: the buffer is scanned and each record
whose lag now exceeds ``max_staleness`` is dropped, covering records that aged past the bound while
buffered. Dropping is per-record but sound for group consumers: siblings share one version (admitted
back-to-back under one policy), so a stale group is dropped whole -- at collect and/or across a scan --
and ``assembly_ready`` never pulls an incomplete group, so no unusable record survives past one publish
cycle.

Drain: when the prompt stream is exhausted, admission stops, in-flight generations are polled to
completion (blocking via ``sleep`` between polls), the consumer keeps training the ready buffer until
``assembly_ready`` declines, and then whatever remains (an incomplete group, or a remainder below one
full batch) is dropped and counted -- there is no future batch to carry it into. Every window still
live is cleared. A budget hit (``reached_max``) or an error instead cancels every STILL-in-flight
submission (releasing its pin), and clears every window in the ``finally``, leaking nothing.
"""

from __future__ import annotations

import math
import queue
import shutil
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from .types import PartitionAdmission, RolloutPolicy

#: One in-flight trajectory: the sampler handle to poll/collect it by, the policy it was pinned to
#: (released once collected or cancelled, so its adapter path can then be pruned), and the training
#: group key it will carry into the ready buffer (every trajectory of one prompt shares it, see
#: :class:`BufferRecord`).
_Inflight = Tuple[Any, RolloutPolicy, str]


class GroupDecision(Enum):
    """What the injected ``group_complete`` seam decides once a training group is fully collected.

    The driver is algorithm-agnostic: it fires ``group_complete`` exactly once per group, the instant that
    group's ``num_generations`` trajectories are all in the ready buffer -- an INTRA-group barrier, so other
    groups keep streaming and overlapping and this never degrades into a fixed-batch barrier. The decision
    tells the driver what to do with the completed group:

    * ``ACCEPT``: the group is trainable as-is; leave it buffered for ``assembly_ready`` to pull.
    * ``AUGMENT``: the hook wrote extra fields onto the samples (e.g. a ReMax greedy ``baseline``) in
      addition to the mandatory scoring. The driver keeps the group exactly as ``ACCEPT`` does -- the
      augmentation is the hook's own side effect, invisible to the driver's buffer.
    * ``REGENERATE``: the group is unusable (e.g. a zero-variance dynamic-sampling group). The driver DROPS
      its buffered records and re-admits the same prompt as a fresh group under the CURRENT policy version
      (the re-admitted siblings are pinned to the new admission version, the conservative staleness anchor),
      bounded by ``max_group_regenerations``; past the cap the group is accepted as-is, matching the
      fixed-batch dynamic-sampling fallback.
    """
    ACCEPT = 'accept'
    AUGMENT = 'augment'
    REGENERATE = 'regenerate'


#: Internal marker the iterator bridge pushes onto its batch queue once the background :meth:`run` thread
#: has finished, so a blocked :meth:`StreamingDriver.__next__` wakes and ends the iteration (or re-raises
#: the run's error) instead of waiting forever.
_ITER_DONE = object()


@dataclass(frozen=True)
class PromptRef:
    """One prompt row drawn from the stream, paired with the driver-assigned ``ordinal``.

    The driver eats a dataloader of prompt ROWS (opaque payloads) and assigns each fresh draw a monotonic
    ordinal: a stable per-draw identifier that seeds the group key (so every trajectory of one prompt shares
    it, and the SAME row re-drawn in a later epoch gets a NEW ordinal and therefore a NEW group) and is what
    the consumer stamps as the sample's ``prompt_id``. The driver never inspects ``row`` -- it only pairs it
    with the ordinal and hands the pair to ``submit``, so the consumer builds a trajectory from the row it is
    given instead of looking one up by index in a resident prompt table.
    """
    ordinal: int
    row: Any


@dataclass(eq=False)
class BufferRecord:
    """One completed trajectory sitting in the driver's ready buffer.

    The unit ``assembly_ready`` inspects and pulls. ``version`` is the policy version the trajectory
    was admitted under (its lag vs the current version is what the post-publish stale scan and the
    consumer's metrics read); ``group_key`` identifies the training group the sample belongs to --
    every trajectory of one prompt EXPANSION shares it (all are admitted back-to-back under one policy,
    so all carry the same ``version``), which is what lets a group consumer require whole groups and the
    stale scan treat a stale group as dropped whole. ``sample`` is opaque to the driver (a dev
    ``RolloutSample``).

    A record carries NO pin: its version pin was acquired at admission and released the moment the
    trajectory was COLLECTED (see :meth:`StreamingDriver.run`), because the pin guards the SAMPLER-side
    adapter path and a collected sample is trained in-process. So leaving the buffer (consumed, stale-
    dropped or drained) is pure bookkeeping -- releasing again there would double-release and raise.
    """
    sample: Any
    version: int
    group_key: str
    #: The pending score submission for this trajectory: ``(ref, index)`` where ``ref`` is the
    #: poll batch's shared score result (a LazyCollect under Ray, the plain value locally) and
    #: ``index`` is this record's position in that batch. ``None`` once the score is reaped onto
    #: ``sample.orm_reward`` (and ``sample.prm_process`` for the PRM channel).
    score_ref: Any = None


@dataclass
class DriverStats:
    """Lifetime counters of one :meth:`StreamingDriver.run` (the consumer's metrics source)."""
    admitted: int = 0               # trajectories submitted to the sampler
    collected: int = 0              # completions fetched from the sampler (a stale one is then dropped)
    consumed: int = 0               # buffered samples handed to ``consume``
    off_policy_consumed: int = 0    # consumed samples whose version lagged the training version
    dropped_stale: int = 0          # samples dropped for exceeding max_staleness (at collect or by a scan)
    dropped_incomplete: int = 0     # samples left in the buffer at drain (no future batch to join)
    train_pulls: int = 0            # ``consume`` calls that returned a positive step count
    optimizer_steps: int = 0        # optimizer steps reported by ``consume``
    publishes: int = 0              # weight publications (partition cycles closed)
    final_version: int = 0          # policy version at exit (= publishes)
    #: Wall-clock seconds the driver thread spent blocked inside ``consume`` (training). The generation
    #: itself runs on the (disaggregated) sampler in the background and is NOT measured here -- the
    #: algorithm-agnostic driver cannot see sampler GPU time without instrumenting the data plane.
    train_active_time: float = 0.0
    #: Wall-clock seconds spent in the idle ``sleep`` branch: trajectories in flight, none completed this
    #: pass, buffer not yet trainable -- i.e. the trainer starved for ready work (the trainer bubble).
    idle_time: float = 0.0
    #: The previous pull's trainer-bubble ratio, folded BY THE DRIVER at each batch boundary:
    #: ``idle_delta / (train_delta + idle_delta)`` where the deltas are taken exactly at the pull
    #: return (train time is accumulated only AFTER a pull returns, so the train delta here is the
    #: PREVIOUS pull's training and the idle delta is the starvation bubble spent waiting for THIS
    #: pull). This keeps the driver-internal accounting order inside the driver -- a consumer that
    #: folded per-step deltas itself would read a constant 0.0. ``0.0`` before the first pull.
    last_pull_idle_ratio: float = 0.0
    submit_time: float = 0.0
    poll_time: float = 0.0
    collect_time: float = 0.0
    score_time: float = 0.0
    publish_time: float = 0.0
    capacity_blocked_passes: int = 0
    staleness_blocked_passes: int = 0
    ready_high_watermark: int = 0
    inflight_high_watermark: int = 0
    #: Wall-clock seconds for the whole :meth:`StreamingDriver.run`, set in its ``finally`` (so it is
    #: defined on every exit path: drain, budget hit, or error). ``total >= train_active + idle``; the
    #: residual is the driver's own admit/poll/collect/publish overhead.
    total_time: float = 0.0


def _no_prune() -> None:
    """Default ``prune`` seam: in-place publication keeps no per-version artifacts to drop."""


def _no_initial_path() -> None:
    """Default ``initial_adapter_path`` seam: in-place has no version-0 adapter path to pin."""


def _prompt_ordinal_key(ref: PromptRef) -> str:
    """Default ``group_key`` seam: the driver-assigned draw ordinal (a stable per-draw group identifier)."""
    return str(ref.ordinal)


#: The single policy token the infer regime pins every trajectory to: version 0, no adapter path, a
#: context key that never collides with a real run_id. ``_acquire_policy`` hands this out and
#: ``_release_pin`` ignores it, so the run loop's acquire/release pair is a no-op in that regime.
_NULL_POLICY = RolloutPolicy('__null__', None, 0, None)


class StreamingDriver:
    """Per-sample streaming control flow -- a PURE rollout pump over an optional control plane.

    Construct with the prompt stream (``prompt_stream``, a dataloader yielding BATCHES of prompt rows, which
    the driver flattens and reads for ``num_train_epochs`` passes), the
    regime knobs (``max_staleness``; ``num_generations`` trajectories per prompt; ``groups_per_partition``
    = the groups one publish cycle is expected to train, used as the window's sizing declaration;
    ``buffer_depth`` = the in-flight + ready backpressure cap in samples), the data-plane seams, and the
    regime selector ``run_id``: a string runs the RL regime (the driver keeps its own version / window /
    pin accounting inline), omitting it runs the infer regime (version pinned at 0, no windows, no pins,
    nothing publishes -- no RL scaffolding).

    The driver owns NO weight/model control and NO publish cadence: publication is an external signal
    (:meth:`request_publish`) carrying the caller's own push callable, which the driver brackets with its
    version/window accounting. Then call :meth:`run` once (or iterate); :attr:`stats` holds the lifetime
    counters afterwards.
    """

    def __init__(self,
                 *,
                 prompt_stream: Iterable[Any],
                 num_train_epochs: float = 1.0,
                 max_staleness: int,
                 num_generations: int,
                 groups_per_partition: int,
                 buffer_depth: int,
                 submit: Optional[Callable[[Any, int, RolloutPolicy], Any]] = None,
                 poll: Optional[Callable[[Sequence[Any]], Sequence[Any]]] = None,
                 collect: Optional[Callable[[Any], Any]] = None,
                 assembly_ready: Optional[Callable[[List[BufferRecord]], Optional[List[BufferRecord]]]] = None,
                 cancel: Optional[Callable[[Any], None]] = None,
                 rollout: Optional[Any] = None,
                 pull_rule: Optional[Literal['group', 'prefix']] = None,
                 need_rows: Optional[int] = None,
                 run_id: Optional[str] = None,
                 consume: Optional[Callable[[List[BufferRecord]], Optional[int]]] = None,
                 group_key: Optional[Callable[[Any], str]] = None,
                 score: Optional[Callable[[Any], Any]] = None,
                 assemble: Optional[Callable[[List[BufferRecord]], Any]] = None,
                 group_complete: Optional[Callable[[str, List[BufferRecord]], GroupDecision]] = None,
                 max_group_regenerations: int = 0,
                 regenerate: Optional[Callable[[str, List[Any]], bool]] = None,
                 baseline_fn: Optional[Callable[[str, List[Any]], Optional[float]]] = None,
                 prune: Optional[Callable[[], None]] = None,
                 initial_adapter_path: Optional[Callable[[], Optional[str]]] = None,
                 snapshot_model: Optional[Any] = None,
                 snapshot_dir: Optional[str] = None,
                 adapter_name: Optional[str] = None,
                 max_steps: Optional[int] = None,
                 reached_max: Optional[Callable[[], bool]] = None,
                 score_worker: Optional[Any] = None,
                 resource_adapter: Optional[Any] = None,
                 sleep: Optional[Callable[[float], None]] = None) -> None:
        # Fail loudly on a mis-wired seam: a non-callable injected here would otherwise surface deep
        # inside the run loop (or silently skip a lifecycle step), far from the construction that caused it.
        # The group-barrier algorithm hooks decompose the deprecated ``group_complete`` seam: PRM
        # scoring is the ``score_worker`` collaborator's own channel, the variance-style verdict is
        # ``regenerate`` and the remax-style baseline is ``baseline_fn``. All three are optional and
        # stack freely; wiring BOTH the deprecated hook and any of its replacements is rejected.
        self._regenerate = regenerate
        self._baseline_fn = baseline_fn
        if group_complete is not None and (regenerate is not None or baseline_fn is not None
                                            or score_worker is not None):
            raise ValueError('StreamingDriver group_complete is deprecated: its former jobs are now the '
                             'score_worker / regenerate / baseline_fn parameters; pass one or the other, '
                             'not both.')
        for name, fn in (('regenerate', regenerate), ('baseline_fn', baseline_fn)):
            if fn is not None and not callable(fn):
                raise TypeError(f'StreamingDriver optional seam {name!r} must be callable or None, '
                                f'got {type(fn).__name__}')
        # The four engine seams default-bind to ``rollout`` (admit/poll/collect/cancel) when omitted;
        # an explicitly injected seam overrides its default, so dev's hand-wired builders keep working.
        self._rollout = rollout
        if submit is None and rollout is not None:
            submit = self._default_submit
        if poll is None and rollout is not None:
            poll = rollout.poll
        if collect is None and rollout is not None:
            collect = self._default_collect
        if cancel is None and rollout is not None:
            cancel = rollout.cancel
        # The pull rule is the same kind of defaulted knob: 'group' waits for whole num_generations
        # groups and re-emits them contiguous, 'prefix' takes a whole-unit multiple of the buffer front.
        # An explicit assembly_ready callable overrides either built-in.
        if pull_rule is not None and pull_rule not in ('group', 'prefix'):
            raise ValueError(f"pull_rule must be 'group' or 'prefix', got {pull_rule!r}")
        if pull_rule is not None and need_rows is None:
            raise ValueError(f'pull_rule={pull_rule!r} needs need_rows (train_batch_size * ga) to size a pull')
        if need_rows is not None and need_rows <= 0:
            raise ValueError(f'need_rows must be positive, got {need_rows}')
        if assembly_ready is None and pull_rule == 'group':
            assembly_ready = self._make_group_pull_rule(num_generations, need_rows)
        if assembly_ready is None and pull_rule == 'prefix':
            assembly_ready = self._make_prefix_pull_rule(need_rows)
        # Two more defaulted collaborators, same duck-typed style as ``rollout``: ``score_worker``
        # (protocol: ``score_group(samples) -> [N][n_scorers]`` rows) default-binds the ORM scoring
        # -- 'group' runs it at the group barrier, 'prefix' at the pull boundary on the whole batch,
        # so a per-sample consumer batches instead of one round-trip per sample; ``resource_adapter``
        # (protocol: ``observe/decide/register_decision``) is fed at every pull boundary and inside
        # the publish bracket BEFORE the push, so its scaling decision is produced and consumed in
        # the same publication.
        self._pull_rule = pull_rule
        self._score_worker = score_worker
        if score_worker is not None and not callable(getattr(score_worker, 'score_group', None)):
            raise TypeError(f'StreamingDriver score_worker needs a score_group(samples) method, '
                            f'got {type(score_worker).__name__}')
        self._resource_adapter = resource_adapter
        if resource_adapter is not None:
            missing = [name for name in ('observe', 'decide', 'register_decision')
                       if not callable(getattr(resource_adapter, name, None))]
            if missing:
                raise TypeError(f'StreamingDriver resource_adapter needs observe/decide/register_decision '
                                f'methods, {type(resource_adapter).__name__} lacks {missing}')
        for name, fn in (('submit', submit), ('poll', poll), ('collect', collect),
                         ('assembly_ready', assembly_ready), ('cancel', cancel)):
            if not callable(fn):
                raise TypeError(f'StreamingDriver seam {name!r} must be callable, got {type(fn).__name__} '
                                f'-- inject it explicitly or pass a rollout to default-bind it')
        # ``consume`` is optional: providing it selects the blocking regime (run() trains in-process, the
        # PPO/RFT/distill path); omitting it selects the iterator regime (run() is bridged to __next__ and
        # the recipe trains, see __iter__). score/assemble/group_complete are the iterator regime's
        # per-sample scoring, batch-assembly and group-barrier seams -- all optional and algorithm-agnostic.
        for name, fn in (('consume', consume), ('score', score), ('assemble', assemble),
                         ('group_complete', group_complete)):
            if fn is not None and not callable(fn):
                raise TypeError(f'StreamingDriver optional seam {name!r} must be callable or None, '
                                f'got {type(fn).__name__}')
        if max_group_regenerations < 0:
            raise ValueError(f'max_group_regenerations must be non-negative, got {max_group_regenerations}')
        # The regime is selected by ``run_id`` alone: a string means the RL regime (the version
        # discipline below is live), ``None`` means the infer regime (the policy pins to the null token,
        # windows are unbounded, nothing ever publishes -- admission is gated only by ``buffer_depth``).
        if max_staleness < 0:
            raise ValueError(f'max_staleness must be non-negative, got {max_staleness}')
        if num_generations <= 0:
            raise ValueError(f'num_generations must be positive, got {num_generations}')
        if groups_per_partition <= 0:
            raise ValueError(f'groups_per_partition must be positive, got {groups_per_partition}')
        if buffer_depth <= 0:
            raise ValueError(f'buffer_depth must be positive, got {buffer_depth}')
        if num_train_epochs <= 0:
            raise ValueError(f'num_train_epochs must be > 0, got {num_train_epochs}')
        # The version/window/pin accounting, held flat on the driver itself: the current policy
        # (rebound, never mutated, on each publish), the adapter-path pin counts (a collected or
        # cancelled trajectory releases its pin exactly once), the live admission windows in step
        # order (FIFO publish), and which step is currently training. The infer regime pins every
        # admission to the null policy (version 0, no path, no pins) and never opens a window.
        self._run_id = run_id
        self._policy: RolloutPolicy = _NULL_POLICY
        self._policy_pins: Dict[str, int] = {}
        self._windows: List[PartitionAdmission] = []
        self._next_step = 0
        self._training_step: Optional[int] = None
        self._prompt_stream = prompt_stream
        #: Passes over ``prompt_stream`` before it counts as exhausted (a fractional epoch still runs one
        #: truncated pass, ceil). The driver loops them itself because a dataloader reads one pass per
        #: ``iter()`` and does not count epochs -- see :meth:`_iter_rows`.
        self._num_train_epochs = num_train_epochs
        #: Monotonic counter assigning each freshly-drawn prompt row its ordinal (see :class:`PromptRef`).
        self._prompt_seq = 0
        self._max_staleness = max_staleness
        self._num_generations = num_generations
        self._groups_per_partition = groups_per_partition
        self._buffer_depth = buffer_depth
        self._submit = submit
        self._poll = poll
        self._collect = collect
        self._assembly_ready = assembly_ready
        self._consume = consume
        self._cancel = cancel
        self._score = score
        self._assemble = assemble
        self._group_complete = group_complete
        self._max_group_regenerations = max_group_regenerations
        # The PRM pass at the group barrier runs whenever the worker carries a wired channel -- its
        # consumer may be a hook here (regenerate / baseline_fn / the deprecated group_complete) OR
        # the training half reading ``sample.prm_process`` downstream (advantage fusion); the
        # channel's own wiring is the signal, probed on the worker once (a Ray proxy shell exposes
        # no fields, so this cannot be a driver-side attribute read).
        self._prm_ready = (score_worker is not None and bool(score_worker.prm_ready()))
        self._group_key = group_key if group_key is not None else _prompt_ordinal_key
        # Snapshot lifecycle, defaulted from snapshot_model: a publish push that saves the next adapter
        # version, a v0 pin, and a prune of paths no pin references. Explicit callables override all three.
        self._snapshot_model = snapshot_model
        self._snapshot_dir = snapshot_dir
        self._snapshot_adapter_name = adapter_name
        self._snapshot_paths: set = set()
        if snapshot_model is not None:
            if initial_adapter_path is None:
                initial_adapter_path = self._save_snapshot_version0
            if prune is None:
                prune = self._default_prune_snapshots
        self._prune = prune if prune is not None else _no_prune
        self._initial_adapter_path = initial_adapter_path if initial_adapter_path is not None else _no_initial_path
        # The optimizer-step budget: either an explicit reached_max callable (dev's builders) or a plain
        # max_steps number compared against the steps commit() has reported so far.
        self._max_steps = max_steps
        self._budget_steps = 0
        self._reached_max = reached_max if reached_max is not None else self._default_reached_max
        self._sleep = sleep if sleep is not None else time.sleep
        self.stats = DriverStats()
        #: Optimizer steps the most recent ``_train_pull`` consume reported (its return channel).
        self._last_pull_steps = 0
        #: Iterator regime: ``consume is None`` means the recipe drives training via ``for batch in driver``
        #: (run() is bridged onto a background thread; see :meth:`__iter__`). ``consume`` provided means the
        #: blocking regime, where run() trains in-process and the driver is NOT iterable.
        self._iterating = consume is None
        #: Iterator-bridge state (only used when ``self._iterating``): the single-slot batch queue the
        #: background run() thread fills and __next__ drains, the ack event that unblocks the bridge's
        #: consume once the recipe has trained the batch, and the step count the recipe reported via
        #: :meth:`commit` (the bridge's consume return, driving the publish cadence).
        self._batch_queue: Optional[queue.Queue] = None
        self._ack_event = threading.Event()
        self._committed_steps = 0
        self._iter_thread: Optional[threading.Thread] = None
        self._run_error: Optional[BaseException] = None
        #: External publication request (see :meth:`request_publish`): a one-shot push callable the run loop
        #: consumes at the next pass boundary. Guarded by ``_publish_lock`` because the request is SET from the
        #: consumer thread (the recipe / UI) and TAKEN on the run-loop thread. ``None`` when no publish is due;
        #: the infer regime never sets it, so its version stays 0 and no weight push ever runs here.
        self._publish_lock = threading.Lock()
        self._pending_push: Optional[Callable[[Any], Optional[str]]] = None
        #: Fold-point snapshots for DriverStats.last_pull_idle_ratio (see the pull boundary in run()).
        self._train_at_pull = 0.0
        self._idle_at_pull = 0.0

    @staticmethod
    def _make_group_pull_rule(num_generations: int, need_rows: int) -> Callable[[List[BufferRecord]], Optional[List[BufferRecord]]]:
        """Pull whole groups only, a lcm(need_rows, num_generations) quantum, group-contiguous.

        Group-relative consumers (GRPO/RFT) normalise each consecutive block of
        num_generations records by POSITION, so the pulled records are re-emitted
        with all siblings of one group adjacent; a partial group stays buffered.
        """
        import math

        def _rule(buffer: List[BufferRecord]) -> Optional[List[BufferRecord]]:
            counts: Dict[Any, int] = {}
            order: List[Any] = []
            for record in buffer:
                key = record.group_key
                if key not in counts:
                    counts[key] = 0
                    order.append(key)
                counts[key] += 1
            complete = [key for key in order if counts[key] >= num_generations]
            groups_per_pull = need_rows // math.gcd(num_generations, need_rows)
            n_pull = groups_per_pull * (len(complete) // groups_per_pull)
            if n_pull < 1:
                return None
            selected = complete[:n_pull]
            selected_set = set(selected)
            by_group: Dict[Any, List[BufferRecord]] = {}
            for record in buffer:
                if record.group_key in selected_set:
                    by_group.setdefault(record.group_key, []).append(record)
            pulled: List[BufferRecord] = []
            for key in selected:
                pulled.extend(by_group[key])
            return pulled

        return _rule

    @staticmethod
    def _make_prefix_pull_rule(need_rows: int) -> Callable[[List[BufferRecord]], Optional[List[BufferRecord]]]:
        """Pull the buffer prefix in whole need_rows multiples (per-sample consumers).

        PPO / distillation score each sample independently, so there is no group
        to wait for: the largest whole-unit prefix is pulled, the remainder stays
        buffered. Order is already completion order, so no reordering is needed.
        """

        def _rule(buffer: List[BufferRecord]) -> Optional[List[BufferRecord]]:
            if len(buffer) < need_rows:
                return None
            return buffer[:(len(buffer) // need_rows) * need_rows]

        return _rule

    def _save_snapshot(self, name: str) -> str:
        """Save one adapter snapshot via snapshot_model and track it for pruning."""
        path = self._snapshot_model.save(name, output_dir=self._snapshot_dir,
                                         adapter_name=self._snapshot_adapter_name, save_optimizer=False)
        self._snapshot_paths.add(path)
        return path

    def _save_snapshot_version0(self) -> str:
        """The version-0 pin: the pre-training adapter snapshot."""
        return self._save_snapshot('v0')

    def _default_publish_push(self, window: Any) -> Optional[str]:
        """The default push: save the NEXT adapter version as the window's snapshot."""
        return self._save_snapshot(f'v{self._current_version() + 1}')

    def _default_prune_snapshots(self) -> None:
        """Drop tracked snapshot paths no live version or in-flight pin references."""
        if not self._snapshot_paths:
            return
        keep = self._adapter_paths_to_keep()
        for path in list(self._snapshot_paths):
            if path not in keep:
                shutil.rmtree(path, ignore_errors=True)
                self._snapshot_paths.discard(path)

    def _default_reached_max(self) -> bool:
        """The step budget: max_steps compared against what commit() has reported."""
        return self._max_steps is not None and self._budget_steps >= self._max_steps

    def _default_submit(self, prompt_ref: Any, trajectory_idx: int, policy: Any) -> Any:
        """Engine-default admit: extract the prompt row and pin the policy onto the rollout."""
        row = prompt_ref.row
        extras = {key: value for key, value in row.items() if key != 'messages'}
        return self._rollout.admit(
            row['messages'], trajectory_idx, prompt_idx=prompt_ref.ordinal,
            prompt_extras=(extras or None),
            adapter_path=getattr(policy, 'adapter_path', None),
            policy_version=getattr(policy, 'version', 0))

    def _default_collect(self, handle: Any) -> Any:
        """Engine-default collect: resolve the sample and stamp its version span here.

        The span (how many publishes the policy advanced between admission and
        collection) needs the version ledger this driver owns, so the stamp
        belongs in the default seam rather than in every recipe closure.
        """
        sample = self._rollout.collect(handle)
        if getattr(handle, 'adapter_path', None) is None:
            # in_place: the episode may have been aborted-and-resumed across publishes. A pinned
            # adapter path (adapter_snapshot) spans nothing; the null ledger keeps version 0.
            sample.version_span = max(0, self._current_version() - sample.policy_version)
        return sample

    # --- lifecycle --------------------------------------------------------------------------------------

    def run(self) -> None:
        """Stream the prompt set: admit per-sample, train whatever is ready, publish on external request.

        Each pass: (1) publish phase -- when an external caller has requested a publication (see
        :meth:`request_publish`; the driver owns neither the cadence nor the weight push), close the OLDEST
        live window (FIFO, as ``on_partition_training_started`` requires), run the caller-supplied push
        through that version/window bracket, bump the tracked version, prune unreferenced artifacts, and
        drop buffered samples whose version lag now exceeds ``max_staleness``; (2) admission -- open a fresh
        window while the staleness gate stays open,
        then admit trajectories one at a time (each prompt expanded into ``num_generations`` of them,
        back-to-back under one pinned policy) until the prompt stream ends or ``buffer_depth`` caps the
        in-flight + ready count; (3) collection -- one non-blocking poll, fetching every completion,
        releasing its pin, and moving it into the ready buffer unless its version lag already exceeds
        ``max_staleness`` (a straggler that aged past the bound while in flight is dropped, not buffered);
        (4) consumption -- when ``assembly_ready`` pulls a
        batch, train it and count the optimizer steps it reports. A pass with in-flight work but no
        completion and no ready batch sleeps briefly instead of spinning.

        Exits when the budget (``reached_max``) is hit, or one pass after the prompt stream is
        exhausted, the in-flight set has drained and ``assembly_ready`` declines the remaining buffer
        (the leftover -- an incomplete group or a sub-batch remainder -- is dropped in the ``finally``,
        counted as ``stats.dropped_incomplete``). The ``finally`` also cancels and unpins whatever is
        still generating and clears every live window, so a budget hit or an error leaks neither a pin
        nor a sampler submission nor a partition.
        """
        # Version 0: pin the initial (pre-training) policy so the first admissions reference a real version
        # and the sampler generates from it, never an unpinned base model (None for in-place; a no-op under
        # the null ledger, where the version is pinned at 0 and nothing publishes).
        self._register_initial_policy(self._initial_adapter_path())
        # A second run() on the same driver must not inherit the previous run's fold snapshots.
        self.stats.last_pull_idle_ratio = 0.0
        self._train_at_pull = 0.0
        self._idle_at_pull = 0.0
        self._budget_steps = 0
        _run_start = time.perf_counter()
        prompts = self._iter_rows()
        # The prompt currently being expanded into trajectories: (prompt_ref, trajectory_idx, group_key),
        # where prompt_ref is the drawn :class:`PromptRef` (row + driver-assigned ordinal). A backpressure cap
        # can interrupt the expansion mid-prompt; the cursor resumes it (carrying the SAME group_key) instead
        # of dropping the remaining trajectories. The group_key is namespaced by a per-expansion ordinal
        # (group_seq) so the SAME prompt re-emitted in a later epoch is a DIFFERENT group: its siblings are
        # admitted under a later policy version and must never be assembled, advantage-scored or stale-dropped
        # together with the earlier epoch's group.
        partial: Optional[Tuple[PromptRef, int, str]] = None
        pending: Optional[PromptRef] = self._draw(prompts)
        group_seq = 0
        # The in-flight set, keyed by sampler handle -- the SINGLE source of truth for what is still
        # generating. A handle is removed the instant it is collected, so the backpressure count, the
        # drain condition and the teardown below all see the CURRENT in-flight set, never a cumulative one.
        handles: Dict[Any, _Inflight] = {}
        ready: List[BufferRecord] = []
        stream_done = pending is None
        exhausted = False
        exhausted_notified = False
        # Group-completion bookkeeping (only load-bearing when a ``group_complete`` seam is injected; the
        # counters are cheap and stay correct regardless). ``group_counts`` is the per-group collected count
        # in the ready buffer (the intra-group barrier reads it); ``group_prompt`` maps a group_key to the
        # prompt it expands (a REGENERATE re-admits that prompt); ``resolved_groups`` fires the seam at most
        # once per group; ``regen_queue`` holds (prompt, group_key) re-admissions a REGENERATE queued;
        # ``regen_counts`` caps them at ``max_group_regenerations``.
        group_counts: Dict[str, int] = {}
        group_prompt: Dict[str, Any] = {}
        resolved_groups: set = set()
        regen_queue: List[Tuple[Any, str]] = []
        regen_counts: Dict[str, int] = {}
        try:
            while True:
                if exhausted or self._reached_max():
                    break

                # (1) Publish phase: run an EXTERNALLY-requested publication (see request_publish). The driver
                #     no longer owns the cadence or the weight push -- it brackets a caller-supplied push with
                #     its own version/window accounting, on this thread, at the pass boundary. A request with
                #     no live window (the stream drained) is dropped: training is done, nothing left to
                #     publish to.
                push = self._take_pending_publish()
                if push is not None:
                    live_windows = self._live_windows()
                    if live_windows:
                        window = live_windows[0]
                        # Produce the resource adapter's scaling decision INSIDE the same bracket
                        # that runs the push: the push (an in-place weight sync the recipe supplies)
                        # reaches the adapter's prepare_publish, which consumes the decision --
                        # produced and consumed in one publication, never split across passes.
                        if self._resource_adapter is not None:
                            decision = self._resource_adapter.decide(
                                step=self.stats.optimizer_steps, publish_boundary=True)
                            self._resource_adapter.register_decision(
                                decision, step=self.stats.optimizer_steps)
                        self._begin_publish(window)
                        _t = time.perf_counter()
                        new_path = push(window)
                        self.stats.publish_time += time.perf_counter() - _t
                        self._finish_publish(window, new_path)
                        self.stats.publishes += 1
                        self._drop_stale(ready, group_counts)
                        # Drop published artifacts no live version, in-flight pin or buffered sample still
                        # references (runs AFTER the stale scan, which released the dropped pins).
                        self._prune()

                # (2) Admission: open a fresh window while the staleness gate is open, then admit
                #     per-sample until the stream ends or the backpressure cap is reached.
                if not stream_done or regen_queue:
                    if not self._open_window():
                        self.stats.staleness_blocked_passes += 1
                    capacity = self._buffer_depth - len(handles) - len(ready)
                    if capacity <= 0:
                        self.stats.capacity_blocked_passes += 1
                    while capacity > 0:
                        if partial is None:
                            if regen_queue:
                                # A group the ``group_complete`` seam rejected (REGENERATE): re-admit its
                                # prompt under the CURRENT policy version, REUSING its group_key (the same
                                # logical group, retried) so its fresh siblings are counted, assembled and
                                # barrier-checked together. No group_seq bump -- it is not a new epoch.
                                prompt_ref, group_key = regen_queue.pop(0)
                                partial = (prompt_ref, 0, group_key)
                            elif pending is None:
                                stream_done = True
                                break
                            else:
                                # Resolve this expansion's group key ONCE (namespaced by the emission ordinal,
                                # see the cursor comment) so every sibling trajectory shares it and a prompt
                                # re-emitted in a later epoch never collides with its earlier group.
                                group_key = f'{self._group_key(pending)}#{group_seq}'
                                group_seq += 1
                                group_prompt[group_key] = pending
                                partial = (pending, 0, group_key)
                                pending = self._draw(prompts)
                        prompt_ref, trajectory_idx, group_key = partial
                        policy = self._acquire_policy()
                        try:
                            _t = time.perf_counter()
                            handle = self._submit(prompt_ref, trajectory_idx, policy)
                            self.stats.submit_time += time.perf_counter() - _t
                        except BaseException:
                            # A failed submit must not leak the pin it just acquired.
                            self._release_pin(policy)
                            raise
                        handles[handle] = (handle, policy, group_key)
                        self.stats.admitted += 1
                        capacity -= 1
                        trajectory_idx += 1
                        partial = ((prompt_ref, trajectory_idx, group_key)
                                   if trajectory_idx < self._num_generations else None)
                    if stream_done and not exhausted_notified:
                        # No further window is opened after this point (a regeneration re-admission
                        # after the stream drained must not re-notify).
                        exhausted_notified = True

                # (3) Collection: one non-blocking poll; every completion is fetched, its pin released,
                #     and -- unless it is already too stale to train -- moved into the ready buffer.
                progressed = False
                fresh: List[BufferRecord] = []
                if handles:
                    # The version cannot change during collection (a publish is phase (1), never here),
                    # so read it once for the whole poll's staleness decision.
                    current = self._current_version()
                    _t = time.perf_counter()
                    completed = self._poll(list(handles))
                    self.stats.poll_time += time.perf_counter() - _t
                    for handle in completed:
                        entry = handles.pop(handle, None)
                        if entry is None:
                            continue  # a handle this driver does not know: ignore, never double-collect
                        _, policy, group_key = entry
                        try:
                            _t = time.perf_counter()
                            sample = self._collect(handle)
                            self.stats.collect_time += time.perf_counter() - _t
                        finally:
                            # The sampler is done with this trajectory, so its version pin is released
                            # HERE (acquire's contract is "pin while one sampler request is using it"): the
                            # pin guards the SAMPLER-side adapter path against pruning, and the collected
                            # sample is now in-process (training never reloads the behaviour adapter). This
                            # is the pin's ONLY release on the collected path -- releasing again when the
                            # record leaves the buffer would double-release and raise. In a finally so a
                            # collect failure leaks no pin either.
                            self._release_pin(policy)
                        self.stats.collected += 1
                        progressed = True
                        # A straggler can stay in flight across MORE than ``max_staleness`` publishes -- the
                        # post-publish stale scan (phase (1)) only sees the BUFFER, never the in-flight set
                        # -- so by the time it completes, its version lag may already exceed the bound. Drop
                        # it HERE instead of buffering it: a record must never ENTER the buffer untrainable,
                        # or -- if its group completes the same pass -- ``assembly_ready`` would pull it and
                        # ``consume`` would train it stale before the next scan could catch it. Its siblings
                        # (admitted back-to-back, same version) are dropped the same way, so the stale group
                        # is dropped WHOLE and never assembled. Counted with the scan's drops.
                        if current - policy.version > self._max_staleness:
                            self.stats.dropped_stale += 1
                            continue
                        # Per-sample scoring seam (PRM/ORM): applied to each trajectory the instant it is
                        # collected, BEFORE it enters the ready buffer, so the group barrier and the
                        # assembler both see a scored sample. A returned value replaces the sample (a seam
                        # that mutates in place and returns None keeps the original). This is per-SAMPLE, not
                        # per-group -- group-relative advantage is the assembler's job, not this seam's.
                        if self._score is not None:
                            _t = time.perf_counter()
                            scored = self._score(sample)
                            self.stats.score_time += time.perf_counter() - _t
                            if scored is not None:
                                sample = scored
                        # The kernel stamps its own group identity onto the sample (the admission expansion
                        # key) so the fixed-format batch is self-describing; positions, not this field,
                        # are the training contract.
                        sample.group_key = group_key
                        record = BufferRecord(sample=sample, version=policy.version, group_key=group_key)
                        ready.append(record)
                        fresh.append(record)
                        group_counts[group_key] = group_counts.get(group_key, 0) + 1
                        self._maybe_group_complete(group_key, ready, group_counts, group_prompt,
                                                   resolved_groups, regen_queue, regen_counts)

                # Per-sample scoring SUBMISSION: the instant trajectories survive the stale gate they
                # go to the score worker NON-BLOCKINGLY -- the same continuous spirit as the sampler's
                # own admission (the sampler and the scoring worker are both continuous services, so
                # scoring runs while generation and training keep running, instead of waiting for a
                # group to complete or for the driver thread to be free). One submission per poll
                # batch keeps the RPC count low; the result ref is reaped onto the samples by
                # _reap_scores below (non-blocking, next pass) or at the prefix pull (blocking tail).
                if self._score_worker is not None and fresh:
                    samples = [record.sample for record in fresh]
                    outcome = (self._score_worker.score_group_prm(samples) if self._prm_ready
                               else self._score_worker.score_group(samples))
                    for index, record in enumerate(fresh):
                        record.score_ref = (outcome, index)
                self._reap_scores(ready, blocking=False)

                # (4) Consumption: train whatever the assembly rule says is ready (exact-global-batch
                #     pulls only; the remainder stays buffered for the next batch).
                records = self._assembly_ready(ready) if ready else None
                if records:
                    if self._pull_rule == 'prefix':
                        # A per-sample consumer pulls a prefix that must leave the trainer with
                        # scored samples: block on the tail of submissions that are already running
                        # (a group-regime driver never blocks here -- its barrier re-checks
                        # non-blockingly and lets a not-yet-scored group wait a pass instead).
                        self._reap_scores(records, blocking=True)
                    _t = time.perf_counter()
                    self._train_pull(records, ready, group_counts)
                    train_now = time.perf_counter()
                    # Fold the bubble ratio at the pull boundary, where the pairing is exact: the
                    # train delta is the PREVIOUS pull's training (already accumulated into the
                    # lifetime counter; this pull's own duration is added only below) and the idle
                    # delta is the starvation bubble spent reaching THIS pull. See
                    # DriverStats.last_pull_idle_ratio.
                    train_delta = self.stats.train_active_time - self._train_at_pull
                    idle_delta = self.stats.idle_time - self._idle_at_pull
                    denominator = train_delta + idle_delta
                    # The FIRST fold has no previous-pull train time, so the ratio would read 1.0
                    # (pure starvation) -- ambiguous, so report 0.0 until the pairing exists.
                    if train_delta > 0 and denominator > 0:
                        self.stats.last_pull_idle_ratio = idle_delta / denominator
                    else:
                        self.stats.last_pull_idle_ratio = 0.0
                    self._train_at_pull = self.stats.train_active_time
                    self._idle_at_pull = self.stats.idle_time
                    self.stats.train_active_time += train_now - _t
                    # Feed the resource adapter's decision window at the same boundary the idle
                    # ratio folds (the commit boundary): one window per pull, purely from stats
                    # that live here -- the adapter itself only decides at publish boundaries.
                    if self._resource_adapter is not None:
                        self._resource_adapter.observe(
                            step=self.stats.optimizer_steps, driver_stats=self.stats,
                            score_time=self.stats.score_time)
                    progressed = True
                elif (stream_done and partial is None and pending is None and not handles
                      and not regen_queue):
                    # The stream drained, nothing is generating and the buffer is not (or no longer)
                    # trainable: this run is over. Break at the top of the next pass; the finally then
                    # drops the unusable remainder and clears every live window. No trailing publish is
                    # needed -- training is done, so there is no further generation to publish to.
                    exhausted = True

                self.stats.ready_high_watermark = max(self.stats.ready_high_watermark, len(ready))
                self.stats.inflight_high_watermark = max(self.stats.inflight_high_watermark, len(handles))
                if not progressed and handles:
                    # Generations in flight, none completed, buffer not (yet) trainable: wait briefly
                    # instead of hot-spinning the poll.
                    _t = time.perf_counter()
                    self._sleep(0.01)
                    self.stats.idle_time += time.perf_counter() - _t
        finally:
            # Budget hit, drained, or an error: cancel and unpin everything STILL generating (uncollected
            # handles -- a collected one already released its pin at collect), drop the unusable buffer
            # remainder, and clear every live window so the control plane ends empty.
            for handle, (_, policy, _) in list(handles.items()):
                self._cancel(handle)
                self._release_pin(policy)
            handles.clear()
            # Buffered records were collected, so their pins are already released; only count the drop.
            self.stats.dropped_incomplete += len(ready)
            ready.clear()
            for window in self._live_windows():
                self._clear_window(window)
            # Drop any publication request that never reached a pass boundary (drain / budget hit / error).
            with self._publish_lock:
                self._pending_push = None
            self.stats.final_version = self._current_version()
            self.stats.total_time = time.perf_counter() - _run_start

    # --- internals ---------------------------------------------------------------------------------------

    def _train_pull(self, records: List[BufferRecord], ready: List[BufferRecord],
                    group_counts: Optional[Dict[str, int]] = None) -> None:
        """Remove one pulled batch from the buffer, train it, and account the steps it reports.

        ``self._last_pull_steps`` carries the consume return back to the caller (0 when the consumer
        reports doing no optimizer step, which then also does not advance the publish cadence). No pin is
        released here: each record's version pin was already released when its trajectory was COLLECTED
        (the pin guards the SAMPLER-side adapter path, and training reads the samples in-process), so a
        record leaving the buffer is pure bookkeeping. A pulled group is complete (``assembly_ready`` pulls
        whole groups for a group consumer), so its ``group_counts`` entry is dropped with it.
        """
        for record in records:
            ready.remove(record)
            if group_counts is not None:
                group_counts.pop(record.group_key, None)
        steps = self._consume_batch(records)
        steps = int(steps) if steps else 1
        self._last_pull_steps = steps
        if steps > 0:
            current = self._current_version()
            self.stats.train_pulls += 1
            self.stats.optimizer_steps += steps
            for record in records:
                self.stats.consumed += 1
                if current - record.version > 0:
                    self.stats.off_policy_consumed += 1

    # --- version / window / pin accounting (the inlined ledger) -----------------------------------------

    def _register_initial_policy(self, adapter_path: Optional[str]) -> None:
        """Pin the version-0 policy so the first admissions reference a real version, never an unpinned base.

        A no-op in the infer regime (the policy already IS the null token). Lazy on purpose: the
        initial adapter path may come from the ``snapshot_model`` save that only runs at run() start.
        """
        if self._run_id is not None:
            self._policy = RolloutPolicy(self._run_id, self._snapshot_adapter_name, 0, adapter_path)

    def _acquire_policy(self) -> RolloutPolicy:
        """Pin the current policy while one sampler request uses it (refcount++ on its adapter path)."""
        if self._run_id is None:
            return _NULL_POLICY
        if self._policy.adapter_path is not None:
            self._policy_pins[self._policy.adapter_path] = self._policy_pins.get(self._policy.adapter_path, 0) + 1
        return self._policy

    def _release_pin(self, policy: RolloutPolicy) -> None:
        """Release a pin from :meth:`_acquire_policy` (refcount--; the pin's ONLY release point)."""
        if policy.adapter_path is None:
            return
        count = self._policy_pins.get(policy.adapter_path, 0)
        if count <= 0:
            raise RuntimeError(f'rollout policy was not acquired: {policy.adapter_path}')
        if count == 1:
            self._policy_pins.pop(policy.adapter_path)
        else:
            self._policy_pins[policy.adapter_path] = count - 1

    def _live_windows(self) -> List[PartitionAdmission]:
        """The live admission windows, oldest step first (the FIFO publish order)."""
        return self._windows

    def _open_window(self) -> bool:
        """Open one fresh admission window if the staleness gate allows (bounded by ``max_staleness``).

        ``target_groups`` is the window's sizing DECLARATION (the groups one publish cycle trains,
        ``groups_per_partition``); membership is not fixed -- the ready buffer fills it in completion
        order, which is why a straggler never blocks a window and the FIFO publish order never waits on
        a specific sample. The infer regime opens no window and always admits (gated only by
        ``buffer_depth``).
        """
        if self._run_id is None:
            return True
        oldest = min((window.step for window in self._windows), default=self._next_step)
        if self._next_step - oldest > self._max_staleness:
            return False
        admission = PartitionAdmission(context_key=self._run_id,
                                        partition_id=f'{self._run_id}:{self._next_step}',
                                        step=self._next_step,
                                        target_groups=self._groups_per_partition,
                                        num_generations=self._num_generations)
        self._next_step += 1
        self._windows.append(admission)
        return True

    def _current_version(self) -> int:
        return self._policy.version

    def _begin_publish(self, window: PartitionAdmission) -> None:
        """Mark the oldest live window as training (the FIFO discipline is enforced, not assumed)."""
        if self._training_step is not None and self._training_step != window.step:
            raise RuntimeError(f'{self._run_id} already trains window step {self._training_step}')
        if not self._windows or self._windows[0] is not window:
            oldest = self._windows[0].step if self._windows else None
            raise RuntimeError(f'window step {window.step} cannot train before the oldest live step {oldest}')
        self._training_step = window.step

    def _finish_publish(self, window: PartitionAdmission, new_path: Optional[str]) -> None:
        """Bump the tracked version to the just-published policy (``new_path``) and retire the window."""
        self._policy = RolloutPolicy(self._run_id, self._snapshot_adapter_name, self._policy.version + 1,
                                     new_path)
        self._windows.remove(window)
        if self._training_step == window.step:
            self._training_step = None

    def _clear_window(self, window: PartitionAdmission) -> None:
        """Retire a still-live window at teardown (no version bump -- the run is ending)."""
        if window in self._windows:
            self._windows.remove(window)
        if self._training_step == window.step:
            self._training_step = None

    def _adapter_paths_to_keep(self) -> set:
        """Paths a live version, an in-flight pin or a buffered sample still references."""
        keep: set = set(self._policy_pins)
        if self._policy.adapter_path is not None:
            keep.add(self._policy.adapter_path)
        return keep

    def _iter_rows(self) -> Iterator[Any]:
        """Flatten ``prompt_stream``'s batches into one row stream spanning ``num_train_epochs`` passes.

        ``prompt_stream`` is a dataloader: it yields a BATCH of prompt rows per step (a list, the shape
        twinkle's ``DataLoader`` produces with an identity collate) and reads ONE pass per ``iter()`` -- it
        does not count epochs itself (``set_epoch`` is caller-driven). The driver runs once and drains its
        prompt stream, so the epoch loop lives here: ``ceil(num_train_epochs)`` passes, each announced to the
        dataloader's ``set_epoch`` (its per-pass reshuffle seed) when it has one, every batch flattened to its
        rows so :meth:`_draw` still admits one trajectory group per row. A stream yielding bare rows (not
        lists) passes through unchanged, so a plain row iterable also works.
        """
        for epoch in range(max(1, math.ceil(self._num_train_epochs))):
            set_epoch = getattr(self._prompt_stream, 'set_epoch', None)
            if set_epoch is not None:
                set_epoch(epoch)
            for batch in self._prompt_stream:
                if isinstance(batch, (list, tuple)):
                    yield from batch
                else:
                    yield batch

    def _draw(self, prompts) -> Optional[PromptRef]:
        """Pull the next prompt ROW from the stream and pair it with a fresh ordinal; ``None`` when exhausted.

        The ordinal is a monotonic per-draw counter (never reused, so the same row re-drawn in a later epoch
        is a distinct :class:`PromptRef` with a distinct group). The driver does not inspect the row -- the
        consumer's ``submit`` seam reads ``ref.row`` to build its trajectory and ``ref.ordinal`` to stamp the
        sample's ``prompt_id``.
        """
        row = next(prompts, None)
        if row is None:
            return None
        ref = PromptRef(self._prompt_seq, row)
        self._prompt_seq += 1
        return ref

    def _drop_stale(self, ready: List[BufferRecord], group_counts: Optional[Dict[str, int]] = None) -> None:
        """Post-publish stale scan: drop every BUFFERED sample whose version lag exceeds the bound.

        This is the SECOND of the two staleness guards -- the first drops a straggler AT COLLECT, before
        it can enter the buffer (see :meth:`run`); this one covers records that were within the bound when
        buffered and then aged past ``max_staleness`` as later publishes bumped the version. Per-record
        but group-sound: every trajectory of a group is admitted back-to-back under one pinned policy, so
        siblings share one version and a stale group is dropped whole across this scan. No pin is released
        here -- a collected record's pin was already released at collect. ``group_counts`` is decremented
        for each dropped record so the group barrier never fires on a group the scan has thinned out.
        """
        current = self._current_version()
        kept: List[BufferRecord] = []
        for record in ready:
            if current - record.version > self._max_staleness:
                self.stats.dropped_stale += 1
                if group_counts is not None:
                    remaining = group_counts.get(record.group_key, 0) - 1
                    if remaining > 0:
                        group_counts[record.group_key] = remaining
                    else:
                        group_counts.pop(record.group_key, None)
            else:
                kept.append(record)
        ready[:] = kept

    def _maybe_group_complete(self, group_key: str, ready: List[BufferRecord], group_counts: Dict[str, int],
                              group_prompt: Dict[str, Any], resolved_groups: set,
                              regen_queue: List[Tuple[Any, str]], regen_counts: Dict[str, int]) -> None:
        """Run the kernel's group-barrier scoring pipeline once a group is fully collected.

        Intra-group barrier (never a batch barrier): fires when ``group_key`` has ``num_generations`` records
        in the ready buffer, at most once per group (``resolved_groups``), while every other group keeps
        streaming and overlapping. No-op when nothing is wired (the infer / PPO / blocking path). The
        pipeline, in order: the scoring pass (``score_worker.score_group`` -> ``sample.orm_reward``, or
        the merged ``score_group_prm`` -> both fields at once when a PRM channel is wired), the
        optional ``regenerate`` verdict, then -- once the group is final (accepted, or a regeneration
        verdict past the cap accepted as-is) -- the optional ``baseline_fn`` write, or -- dev-compat
        only -- the deprecated ``group_complete`` hook doing all three itself. Everything is timed into
        ``stats.score_time`` HERE, not inside any hook. On ``REGENERATE`` (or a ``True`` verdict)
        within the attempt cap the group's records are dropped from the buffer and its prompt
        re-queued for admission under the current policy version (the conservative staleness
        anchor); the group_key is reused so the retried siblings are counted and barrier-checked
        as one group. A verdict past the cap is accepted as-is, matching the fixed-batch
        dynamic-sampling fallback.
        """
        if (self._group_complete is None and self._score_worker is None
                and self._regenerate is None and self._baseline_fn is None):
            return
        if group_key in resolved_groups:
            return
        if group_counts.get(group_key, 0) < self._num_generations:
            return
        group_records = [record for record in ready if record.group_key == group_key]
        # The group must also be SCORED before it can be judged: the scores were submitted the
        # instant each trajectory was collected, so this guard usually passes the same pass; a
        # straggler score simply waits one pass (the barrier re-checks) instead of blocking here.
        if any(record.score_ref is not None for record in group_records):
            return
        samples = [record.sample for record in group_records]
        if self._group_complete is not None:
            decision = self._group_complete(group_key, group_records)
        else:
            if self._regenerate is not None and self._regenerate(group_key, samples):
                decision = GroupDecision.REGENERATE
            else:
                decision = GroupDecision.ACCEPT
        if (decision == GroupDecision.REGENERATE
                and regen_counts.get(group_key, 0) < self._max_group_regenerations):
            for record in group_records:
                ready.remove(record)
            group_counts.pop(group_key, None)
            regen_counts[group_key] = regen_counts.get(group_key, 0) + 1
            resolved_groups.discard(group_key)
            regen_queue.append((group_prompt[group_key], group_key))
        else:
            # The group is FINAL now (plainly accepted, or a REGENERATE verdict past the cap that
            # is accepted as-is): a remax-style baseline must land on every kept sibling here, or
            # this group would train with no baseline while its batch peers carry one.
            if self._group_complete is None and self._baseline_fn is not None:
                baseline = self._baseline_fn(group_key, samples)
                if baseline is not None:
                    for sample in samples:
                        sample.baseline = baseline
            resolved_groups.add(group_key)

    @staticmethod
    def _score_ready(ref: Any) -> bool:
        """A LazyCollect answers its own non-materialising ``ready()``; a plain value already is one."""
        probe = getattr(ref, 'ready', None)
        return bool(probe()) if callable(probe) else True

    @staticmethod
    def _score_deref(ref: Any) -> Any:
        """Collect the score result: a LazyCollect derefs (cached), a plain value passes through."""
        return ref() if getattr(ref, '_is_lazy_collect', False) else ref

    def _reap_scores(self, records: List[BufferRecord], *, blocking: bool) -> None:
        """Write finished score submissions back onto the samples; a pending one keeps waiting.

        The submission happened at collection time (see the run loop), so scoring has been running
        alongside generation and training; this only moves results onto the samples. Non-blocking
        mode (the ready buffer, every pass) skips a not-yet-finished submission -- the group barrier
        re-checks next pass; blocking mode (a prefix pull) waits out the tail, because the pulled
        batch must reach the trainer scored. Timed into ``stats.score_time``, which therefore
        measures the reap wait only -- the submission itself returns immediately.
        """
        if self._score_worker is None:
            return
        _t = time.perf_counter()
        for record in records:
            if record.score_ref is None:
                continue
            outcome, index = record.score_ref
            if not blocking and not self._score_ready(outcome):
                continue
            result = self._score_deref(outcome)
            if self._prm_ready:
                rows, prm_refs = result
                record.sample.prm_process = prm_refs[index]
            else:
                rows = result
            record.sample.orm_reward = rows[index]
            record.score_ref = None
        self.stats.score_time += time.perf_counter() - _t

    def _consume_batch(self, records: List[BufferRecord]) -> Optional[int]:
        """Train one pulled batch: the injected blocking ``consume`` seam, or the iterator bridge.

        In the blocking regime (``consume`` provided) this is the consumer's own score-and-train half. In
        the iterator regime (``consume=None``) the batch is assembled and handed to ``__next__`` through the
        queue, and this call BLOCKS until the recipe trains it and calls :meth:`commit`, returning the real
        optimizer-step count the recipe reported -- so twinkle's contract (publish happens after consume
        returns) makes the version bump wait for training to finish.
        """
        if self._consume is not None:
            return self._consume(records)
        return self._bridge_consume(records)

    def _bridge_consume(self, records: List[BufferRecord]) -> int:
        """Iterator-regime consume: assemble the batch, hand it to ``__next__``, block for the recipe's ack.

        The FIXED output format is the list of scored samples (``group_key`` / ``policy_version`` stamped
        by the kernel, scoring fields written by the hooks); the optional ``assemble`` override replaces
        it with a custom batch object. The batch goes on the single-slot queue (the driver thread blocks on
        the ack right after, so the queue never overflows and generation keeps overlapping on the sampler
        actor), and the step count the recipe reports via :meth:`commit` becomes this consume's return --
        the publish cadence unit.
        """
        batch = self._assemble(records) if self._assemble is not None else [record.sample for record in records]
        self._committed_steps = 0
        self._ack_event.clear()
        self._batch_queue.put(batch)
        self._ack_event.wait()
        return self._committed_steps

    # --- iterator regime (``for batch in driver:``) ------------------------------------------------------

    def __iter__(self) -> 'StreamingDriver':
        """Drive the blocking :meth:`run` on a background thread and yield assembled, scored batches.

        Only valid in the iterator regime (constructed with ``consume=None``); a driver built with a
        blocking ``consume`` is already driven by :meth:`run` and is not iterable. Single-use: starting
        iteration twice raises. The recipe trains each yielded batch and calls :meth:`commit` to release the
        driver thread blocked in the bridge and report the optimizer steps it ran.
        """
        if not self._iterating:
            raise RuntimeError(
                'StreamingDriver was constructed with a blocking consume seam, so it is driven by run() and '
                'is not iterable. To iterate (`for batch in driver:`), construct it with consume=None and '
                'inject the score/assemble/group_complete seams instead.')
        if self._iter_thread is not None:
            raise RuntimeError('StreamingDriver iteration was already started; a driver instance is single-use.')
        self._batch_queue = queue.Queue(maxsize=1)
        self._committed_steps = 0
        self._ack_event.clear()
        self._run_error = None

        def _runner() -> None:
            try:
                self.run()
            except BaseException as exc:  # re-raised on the recipe thread by __next__
                self._run_error = exc
            finally:
                self._batch_queue.put(_ITER_DONE)

        self._iter_thread = threading.Thread(target=_runner, name='streaming-driver', daemon=True)
        self._iter_thread.start()
        return self

    def __next__(self) -> Any:
        """The next assembled batch, blocking until the driver thread produces one or the run ends.

        Ends the iteration (``StopIteration``) when the background run drains; if the run raised, its
        exception is re-raised here on the recipe thread so a failure is never silently swallowed.
        """
        if self._batch_queue is None:
            raise StopIteration
        batch = self._batch_queue.get()
        if batch is _ITER_DONE:
            if self._run_error is not None:
                raise self._run_error
            raise StopIteration
        return batch

    def commit(self, optimizer_steps: int = 1) -> None:
        """Release the driver thread blocked in the bridge, reporting the real optimizer steps just run.

        The recipe calls this after training the batch ``__next__`` handed it. The count becomes the
        bridge's consume return, which drives the publish cadence (twinkle publishes after consume returns,
        so the trained weights -- not stale ones -- are what the next admission window samples from). A falsy
        count is treated as one step, matching :meth:`_train_pull`. The same count also feeds the internal
        ``max_steps`` budget, so a driver constructed with a numeric budget exits on its own.
        """
        self._committed_steps = int(optimizer_steps) if optimizer_steps else 1
        self._budget_steps += self._committed_steps
        self._ack_event.set()

    # --- external publication signal (the driver owns the accounting, NOT the cadence or the push) --------

    @staticmethod
    def no_push(window: Any) -> None:
        """A publish request that pushes nothing: the caller ran the weight update itself.

        The in-place strategy's visible half (abort in-flight + sync weights) runs in the recipe loop;
        the driver's bracket then carries only the version accounting, hence this no-op.
        """

    def request_publish(self, push: Optional[Callable[[Any], Optional[str]]] = None) -> None:
        """Signal that the just-trained weights should be published at the next pass boundary.

        ``push(window) -> Optional[str]`` is the CALLER's weight sync (the same
        :class:`~twinkle_agentic.async_rl.weight_sync.WeightSyncStrategy.publish` signature): it aborts any
        in-flight generation and overwrites the sampler's weights (in-place, returning ``None``), or saves a
        new adapter snapshot (returning its path). Omitting ``push`` selects the default built from
        ``snapshot_model`` (:meth:`_default_publish_push`); :attr:`no_push` is the explicit no-op for a
        caller that already pushed the weights itself. The driver does NOT store it and does NOT decide
        when to call it -- the external controller owns the cadence and the weight push; the driver only
        brackets the push with its own version/window accounting (close the oldest window before, bump the
        version + drop now-stale samples + prune after), executed on the run-loop thread at the top of the
        next pass so the buffer/pins are only ever mutated on that one thread.

        Thread-safe: the request is set from the consumer thread (the recipe / UI) and taken on the run-loop
        thread. A second request before the first is consumed raises (the run loop consumes one per pass, so
        this only happens on a wiring bug). The infer regime never calls this, so its version stays 0.
        """
        if push is None:
            if self._snapshot_model is None:
                raise ValueError('request_publish() without a push needs snapshot_model on the driver -- pass '
                                 'a push callable, use StreamingDriver.no_push when the weights are already '
                                 'synced, or construct with snapshot_model for the default adapter save.')
            push = self._default_publish_push
        if not callable(push):
            raise TypeError(f'request_publish needs a callable push, got {type(push).__name__}')
        with self._publish_lock:
            if self._pending_push is not None:
                raise RuntimeError('a publish is already pending; the run loop consumes one request per pass, '
                                   'so a second request before it is taken is a wiring bug.')
            self._pending_push = push

    def _take_pending_publish(self) -> Optional[Callable[[Any], Optional[str]]]:
        """Take (and clear) the pending publication request; ``None`` when no publish is due this pass."""
        with self._publish_lock:
            push, self._pending_push = self._pending_push, None
            return push
