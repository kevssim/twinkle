# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from .types import PartitionAdmission


class WeightSyncStrategy(ABC):
    """Publishes newly-trained weights and returns their version-pinning handle.

    Owned by the dev-side controller (``swift.dev.recipe.rl.streaming.StreamingPublisher``), NOT by the
    ``StreamingDriver``: the driver was purified into a rollout pump that holds no weight/model control, so the
    controller keeps this strategy and hands its bound :meth:`publish` to ``StreamingDriver.request_publish``
    once per publication. The driver then calls it as ``push(window)`` inside its own version/window bracket
    (close the oldest window before, bump the version + drop stale + prune after) -- the strategy never lives
    on the driver and the driver never decides the cadence.
    """

    @abstractmethod
    def publish(self, admission: PartitionAdmission) -> str | None:
        """Make the trained policy visible to the sampler and return its handle.

        The handle is a disk path when each version is a separate snapshot, or
        ``None`` when the sampler's single live copy is overwritten in place.
        """

    def __call__(self, admission: PartitionAdmission) -> str | None:
        return self.publish(admission)


class AdapterSnapshotSync(WeightSyncStrategy):
    """Save each trained version as a LoRA adapter snapshot on disk.

    ``save_fn`` is the existing disk-save seam (``pipeline._save_adapter``): it
    writes the adapter and returns its non-empty checkpoint path, which the
    control plane then pins by reference count.  Because each version lives at
    its own path, several can be resident at once, so this strategy does not
    bound ``max_staleness``.
    """

    def __init__(self, save_fn: Callable[[PartitionAdmission], str]):
        if not callable(save_fn):
            raise TypeError(f'AdapterSnapshotSync needs a callable save_fn, got {type(save_fn).__name__}')
        self._save_fn = save_fn

    def publish(self, admission: PartitionAdmission) -> str:
        return self._save_fn(admission)


class InPlaceWeightSync(WeightSyncStrategy):
    """Overwrite the sampler's single live copy of the weights; ``publish`` returns ``None``.

    ``sync_fn`` is the in-place weight-sync callback (in dev, the rollout engine's ``sync_weights`` via
    ``CheckpointEngineManager``); ``abort_fn`` is called BEFORE ``sync_fn`` so no generation decodes across
    the overwrite: each in-flight generation is interrupted, returns its partial tokens with a resume
    point, and is re-submitted on the new weights, keeping every segment's logprobs on one consistent
    policy version. ``abort_fn`` may be None only when the sampler is provably idle at the sync point
    (strictly synchronous regime); a streaming regime with a lookahead window needs it, which
    ``dev.config.validate._check_streaming_publication`` enforces. Whether staleness can exceed 1 is a
    property of the deployment (colocated forces idle, disaggregated allows verl-style fully-async).
    """

    def __init__(self, sync_fn: Callable[[], None], abort_fn: Callable[[], Any] | None = None):
        if not callable(sync_fn):
            raise TypeError(f'InPlaceWeightSync needs a callable sync_fn, got {type(sync_fn).__name__}')
        if abort_fn is not None and not callable(abort_fn):
            raise TypeError(f'InPlaceWeightSync abort_fn must be callable or None, got {type(abort_fn).__name__}')
        self._sync_fn = sync_fn
        self._abort_fn = abort_fn

    def publish(self, admission: PartitionAdmission) -> None:
        # Abort every in-flight generation BEFORE the weights are overwritten, so none decodes across the
        # update: each returns its partial tokens + resume point and re-runs on the new weights (see the
        # class docstring). A no-op when nothing is in flight.
        if self._abort_fn is not None:
            self._abort_fn()
        self._sync_fn()
        return None


def build_weight_sync_strategy(
    name: str,
    *,
    save_fn: Callable[[PartitionAdmission], str] | None = None,
    sync_fn: Callable[[], None] | None = None,
    abort_fn: Callable[[], Any] | None = None,
) -> WeightSyncStrategy:
    """Build the weight-sync strategy selected by ``name``.

    ``save_fn`` is required for ``adapter_snapshot`` and ``sync_fn`` for
    ``in_place``.  ``abort_fn`` is the optional abort-on-publish hook, meaningful
    only for ``in_place`` (the sampler's ``abort_all_inflight``): it interrupts
    every in-flight generation before the weights are overwritten so each resumes
    on the new weights instead of decoding across the update.  There is no separate
    ``nccl`` strategy: NCCL is the transport ``in_place`` uses when the sampler is
    disaggregated, chosen by the deployment (``CheckpointEngineManager`` mode), not
    by this name.
    """
    if name == 'adapter_snapshot':
        if save_fn is None:
            raise ValueError("weight_sync_strategy='adapter_snapshot' requires a save_fn")
        return AdapterSnapshotSync(save_fn)
    if name == 'in_place':
        if sync_fn is None:
            raise ValueError("weight_sync_strategy='in_place' requires a sync_fn")
        return InPlaceWeightSync(sync_fn, abort_fn=abort_fn)
    if name == 'nccl':
        raise ValueError(
            "weight_sync_strategy='nccl' is not a separate strategy: NCCL is how 'in_place' transports "
            "weights to a disaggregated sampler (CheckpointEngineManager mode='standalone'). Use 'in_place' "
            "and select colocate (IPC) vs standalone (NCCL) via the deployment setting.")
    raise ValueError(f'unknown weight_sync_strategy: {name!r}')
