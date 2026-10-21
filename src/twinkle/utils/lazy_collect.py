# Copyright (c) ModelScope Contributors. All rights reserved.
"""Process-wide lazy-collect switch, shared by infra and the component capabilities.

The default ``True`` keeps remote results as lazy collectibles; ``initialize(lazy_collect=False)``
makes every undecorated remote call block until its result is materialised. The global lives here
(a neutral utils module) so component-side code can read it without importing infra; only
``initialize`` writes it, through :func:`set_lazy_collect`.
"""

_lazy_collect = True


def get_lazy_collect() -> bool:
    """Read the process-wide lazy-collect default."""
    return _lazy_collect


def set_lazy_collect(value: bool) -> None:
    """Write the process-wide lazy-collect default (called by ``twinkle.infra.initialize``)."""
    global _lazy_collect
    _lazy_collect = value
