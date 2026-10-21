# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Optional

from twinkle.infra import remote_function


class AsyncSaveMixin:
    """Staging/write separation for checkpoint persistence.

    A save splits into a staging half (collecting state off the training device: gathers, D2H copies,
    optimizer/rng/scheduler state_dicts -- everything that touches the model or a process group) and a
    write half (serializing staged CPU payloads to disk -- pure file I/O, safe off-thread). Callers
    stage synchronously, then hand the write closure to :meth:`submit_async_write`; the checkpoint
    bytes are frozen at stage time, so training steps taken while the write is in flight never leak
    into the saved artifact.

    One write in flight per instance: a new submission first joins the previous write, so a slow disk
    can at most delay the next save, never pile up snapshots. Exceptions raised by the background
    write are re-raised to the next join (or :meth:`wait_async_saves`), never swallowed.
    """

    def _stage(self) -> None:
        """Hook for subclasses needing extra synchronous staging work before a write is submitted."""

    def submit_async_write(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` (pure file I/O over already-staged CPU payloads) on the background writer."""
        executor = self._async_save_executor()
        self._join_async_save()
        self._async_write_future = executor.submit(fn)

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def wait_async_saves(self) -> None:
        """Join and re-raise any failure from the in-flight background write."""
        self._join_async_save()

    def _async_save_executor(self) -> ThreadPoolExecutor:
        # Lazily created so instances that never save asynchronously pay nothing.
        if self._async_write_executor is None:
            self._async_write_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix='async-save')
        return self._async_write_executor

    def _join_async_save(self) -> None:
        future = self._async_write_future
        if future is not None:
            self._async_write_future = None
            future.result()

    def shutdown_async_saves(self) -> None:
        """Join any in-flight write and retire the writer thread (idempotent)."""
        self._join_async_save()
        executor = self._async_write_executor
        if executor is not None:
            self._async_write_executor = None
            executor.shutdown(wait=True)

    def __del__(self):
        # Best-effort: an unjoined in-flight write at teardown loses the checkpoint silently otherwise.
        executor = getattr(self, '_async_write_executor', None)
        if executor is not None:
            self.shutdown_async_saves()

    _async_write_executor: Optional[ThreadPoolExecutor] = None
    _async_write_future: Optional[Future] = None
