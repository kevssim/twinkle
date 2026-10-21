# Copyright (c) ModelScope Contributors. All rights reserved.
"""twinkle infrastructure: driver-side dispatch to Ray workers.

This package re-exports its public surface; the implementation lives in focused
modules:

- ``dispatch``: the ``remote_class`` / ``remote_function`` decorators, argument
  dispatch and result collection.
- ``continuous_work``: the ``enable_continous_work`` ledger, barriers and the
  worker-side async companions, plus actor identity helpers.
- ``ray_tasks``: ``remote_map``, short-lived CPU task fan-out.
- ``collectors``: cross-worker result aggregation for tensor outputs.
- ``_torch_setup``: per-process torch init knobs (seed / determinism / tf32).
- ``_ray``: Ray resource placement and the execute/collect helpers.

Process-wide state (mode, device groups, device mesh, notifier) is defined here
and owned by :func:`initialize`; ``dispatch`` reads it through this module at
call time, so ``infra._mode``-style access and ``initialize`` keep one owner.
"""
import json
import os
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from twinkle.notifier import Notifier
from twinkle.utils import DeviceGroup, DeviceMesh, Platform, requires
from . import _torch_setup
from .collectors import collect_dp_and_flatten, collect_tensor_dict
from .continuous_work import _cw_drop_locked, _cw_unregister, _select_workers, get_worker_id
from .dispatch import (_dispatch_args, _get_device_mesh_param_name, get_device_placement, remote_class,
                       remote_function)
from .ray_tasks import remote_map
from twinkle.utils.lazy_collect import set_lazy_collect
from .worker_pool import with_worker_pool

__all__ = [
    'initialize',
    'get_mode',
    'get_device_mesh',
    'get_device_placement',
    'remote_class',
    'remote_function',
    'remote_map',
    'get_worker_id',
    'collect_dp_and_flatten',
    'collect_tensor_dict',
]

_mode: Optional[Literal['local', 'ray']] = 'local'

if os.environ.get('TWINKLE_MODE', 'local') == 'ray':
    _mode = 'ray'

# Torch process-global init knobs (seed / determinism / tf32), bundled into one dict so a new knob never
# means a new initialize()/create_workers() argument or a new env var. Each only takes effect in the
# process that runs the compute, so the set is published to every Ray worker through the
# EXTRA_TORCH_KWARGS env var and re-applied there; _torch_setup owns the whole mechanism.
_extra_torch_kwargs: Dict[str, Any] = _torch_setup.normalize(None)

_device_group: Optional[List[DeviceGroup]] = None

_device_mesh = None

_notifier: Optional[Any] = None

_name: Optional[str] = None

_TWINKLE_NOTIFIER_ENV = 'TWINKLE_NOTIFIER'


def _maybe_load_worker_notifier() -> None:
    """Lazily reconstruct notifier + name on ray workers from inherited env vars."""
    global _notifier, _name
    if _notifier is not None:
        return
    if _name is None:
        _name = os.environ.get('TWINKLE_NAME') or None
    raw = os.environ.get(_TWINKLE_NOTIFIER_ENV)
    if not raw:
        return

    candidate = Notifier.from_dict(json.loads(raw))
    if candidate is not None:
        _notifier = candidate


def initialize(mode: Literal['local', 'ray'] = 'local',
               nproc_per_node: int = 8,
               ncpu_proc_per_node: int = 8,
               extra_torch_kwargs: Optional[Dict[str, Any]] = None,
               groups: Optional[List[DeviceGroup]] = None,
               global_device_mesh: Optional[DeviceMesh] = None,
               lazy_collect: bool = True,
               name: Optional[str] = None,
               notifier: Optional[Any] = None):
    """Initialize the twinkle infrastructure.

    Args:
        mode: The mode of twinkle works in.
            'local': Run with a single GPU, or torchrun.
            'ray': Run in ray cluster.
        nproc_per_node: The GPU count(number of processes) per node.
        ncpu_proc_per_node: The CPU processes count per node.
        extra_torch_kwargs: Per-process PyTorch init knobs, bundled so a new one needs no new argument
            (see ``_torch_setup``): ``seed`` (seed everything with this), ``full_determinism`` (freeze
            the random, use determinism kernels, default `False`) and ``tf32`` (explicit matmul/conv
            precision; None leaves torch's default). Applied here and re-applied on every Ray worker
            through the EXTRA_TORCH_KWARGS env var, since each is a per-process global.
        groups: The device groups of the training.
        global_device_mesh: The global default device mesh.
        lazy_collect: Lazy collect all outputs in workers, default `True`.
        name: The name of this run.
        notifier: Optional callable (e.g. ``DingNotifier``) invoked with a
            single ``str`` message whenever any ``remote_function``-decorated
            method raises. The original exception is always re-raised; the
            notifier is best-effort and its own failures are swallowed.
    """
    global _mode, _device_group, _extra_torch_kwargs, _device_mesh, _name, _notifier
    assert mode in ('local', 'ray')
    _mode = mode
    _name = name
    set_lazy_collect(lazy_collect)
    _notifier = notifier
    if name is not None:
        os.environ['TWINKLE_NAME'] = name
    os.environ.setdefault('TWINKLE_SESSION_ID', str(os.getpid()))
    if notifier is not None and hasattr(notifier, 'to_dict'):
        os.environ[_TWINKLE_NOTIFIER_ENV] = json.dumps(notifier.to_dict())
    if global_device_mesh is not None:
        _device_mesh = global_device_mesh

    # Seed + set determinism/tf32 in THIS process, and publish the knob set for the Ray workers (which
    # inherit env, not this process's globals). In local/torchrun mode this process does the compute; in
    # ray mode the driver still runs its own light compute (e.g. collect), so both want it applied here.
    _extra_torch_kwargs = _torch_setup.apply(extra_torch_kwargs)
    os.environ[_torch_setup.EXTRA_TORCH_KWARGS_ENV] = _torch_setup.to_env_value(_extra_torch_kwargs)
    if _mode == 'local':
        if groups is not None:
            _device_group = groups
        else:
            _device_group = [
                DeviceGroup(
                    name='default',
                    ranks=list(range(Platform.get_world_size())),
                    device_type=Platform.get_platform().device_prefix(),
                )
            ]

        if _device_mesh is None:
            _device_mesh = DeviceMesh(
                device_type=Platform.device_prefix(),
                mesh=np.arange(Platform.get_world_size()),
                mesh_dim_names=('dp', ))

        assert Platform.get_world_size() == _device_mesh.world_size
    else:
        requires('ray')
        from ._ray import RayHelper
        assert groups is not None
        # groups is needed for ray
        _device_group = groups
        RayHelper.initialize(
            nproc_per_node=nproc_per_node, ncpu_proc_per_node=ncpu_proc_per_node, device_groups=_device_group)


def get_device_mesh() -> Optional[DeviceMesh]:
    """The global default device mesh set by ``initialize`` (None before init).

    A ``@remote_class`` model receives this mesh automatically: its wrapped ``__init__`` injects it
    whenever the caller passes ``device_mesh=None`` (see ``_new_init_body``). A model that builds
    *outside* that wrapper -- one overriding ``__init__`` rather than inheriting the wrapped base --
    resolves its mesh through this accessor to uphold the same invariant: a constructed model holds a
    real mesh, so the InputProcessor / OptimizerGroup it builds get proper DP placement instead of
    silently degrading to single-rank loss normalisation and no metric aggregation.
    """
    return _device_mesh


def get_mode() -> Optional[Literal['local', 'ray']]:
    """The twinkle execution mode (``'local'`` or ``'ray'``) set by :func:`initialize` (None before init).

    Public accessor over the module-private ``_mode``. Callers outside twinkle (e.g. swift/dev) need to
    know whether cross-process isolation is in effect: under ``'ray'`` a ``forward_only`` serializes its
    inputs to a worker process, so a driver-side defensive copy of those inputs is redundant, while under
    ``'local'`` the compute shares this process and the copy is load-bearing. Reading ``_mode`` directly
    would reach into a private global, so this is the sanctioned seam.
    """
    return _mode
