"""Control-plane primitives for async RL.

The public names resolve LAZILY (PEP 562 ``__getattr__``): ``RLContext`` / ``RLContextManager`` /
the ``weight_sync`` strategies / ``StreamingDriver`` are dependency-light, and a consumer that only
needs a primitive (dev's streaming recipes drive ``RLContextManager`` + ``build_weight_sync_strategy``
over their own sampler) must not be forced to import anything heavier. Each name is imported from
its defining submodule on first access instead of eagerly here.

The former TransferQueue runtime (pipeline / workers / scheduler / data_plane / native_tq / the
``*_sampler_tq`` backends / generation_submissions) is archived under ``legacy/``: it modelled a
server-side multi-tenant context pipeline, which this architecture retires -- the streaming driver
runs on the client, the server is a stateless execution device, and the two never shared state.
New code must not import from ``legacy``.
"""

from __future__ import annotations

import importlib

#: Maps each public name to the submodule that defines it, so ``__getattr__`` imports only what is
#: accessed, and only when. Kept next to ``__all__`` so the two cannot drift.
_EXPORTS = {
    'AdapterSnapshotSync': 'weight_sync',
    'BufferRecord': 'streaming_driver',
    'ClusterAdapter': 'cluster_adapter',
    'ContextStatus': 'context_manager',
    'DriverStats': 'streaming_driver',
    'GroupDecision': 'streaming_driver',
    'InPlaceWeightSync': 'weight_sync',
    'LoraContextManager': 'context_manager',
    'ModelMemoryEstimate': 'cluster_adapter',
    'PartitionAdmission': 'types',
    'RLContext': 'types',
    'RLContextManager': 'context_manager',
    'RLResourceEstimator': 'cluster_adapter',
    'RLResourcePlan': 'cluster_adapter',
    'RolePlacement': 'cluster_adapter',
    'REF_ONLY_REMOTE_GROUP': 'score_worker',
    'ScoreWorker': 'score_worker',
    'RolloutPolicy': 'types',
    'StreamingDriver': 'streaming_driver',
    'WeightSyncStrategy': 'weight_sync',
    'build_weight_sync_strategy': 'weight_sync',
    'estimate_plan': 'cluster_adapter',
    'flex_role_groups': 'cluster_adapter',
    'elastic_roles': 'cluster_adapter',
    'resource_device_groups': 'cluster_adapter',
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Import and cache ``name`` from its defining submodule on first access (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is not None:
        value = getattr(importlib.import_module(f'.{module_name}', __name__), name)
        globals()[name] = value  # cache, so __getattr__ is not hit again for that name
        return value
    # Not a re-exported name: fall back to importing a submodule of the same name, so
    # ``from twinkle_agentic.async_rl import <submodule>`` keeps working.
    try:
        module = importlib.import_module(f'.{name}', __name__)
    except ImportError as exc:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from exc
    globals()[name] = module
    return module


def __dir__():
    return sorted(__all__)
