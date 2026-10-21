# Copyright (c) ModelScope Contributors. All rights reserved.
"""Short-lived Ray CPU tasks: the fan-out primitive for stateless parallel jobs.

Distinct from ``remote_class``/``remote_function`` (resident training worker pools
bound to device groups) -- see ``remote_map``'s docstring.
"""
from typing import Any, Callable, List, Optional


def remote_map(func: Callable, args_list: List[tuple], *, runtime_env: Optional[dict] = None) -> List[Any]:
    """Fan a picklable function out over short-lived CPU tasks, returning results in input order.

    Unlike remote_class/remote_function (which drive the resident training worker pool bound to
    device groups), this serves stateless embarrassingly-parallel jobs: each tuple in ``args_list``
    runs ``func(*tuple)`` on its own Ray CPU task. Starts a local head if no cluster is running.

    ``runtime_env`` is the caller's choice when a local head has to be started; the default is a
    plain head, because a CPU task needs neither Ray Serve patches nor server-side env propagation.
    Callers wanting those pass their own runtime env explicitly.
    """
    if not args_list:
        return []
    import ray

    if not ray.is_initialized():
        ray.init(runtime_env=runtime_env, ignore_reinit_error=True)
    remote_fn = ray.remote(func)
    return ray.get([remote_fn.remote(*args) for args in args_list])
