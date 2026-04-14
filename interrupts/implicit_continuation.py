"""Implicit continuation: a lock that keeps weight-sync from happening
while a generation call is in-flight.

This is a coarse approximation of PipelineRL's approach — since we don't
control vLLM's internal decode loop from Python, the finest granularity we
can enforce is "one generate() call = one atomic forward pass". Coordinator
acquires the lock before pushing weights, releases after.

Backed by mp.Lock so it works across the trainer/inference process boundary.
"""
import asyncio
import multiprocessing as mp
from interrupts.base import InterruptStrategy


class ImplicitContinuation(InterruptStrategy):
    def __init__(self):
        ctx = mp.get_context()
        self._forward_lock = ctx.Lock()

    async def prepare_for_sync(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._forward_lock.acquire)

    async def resume_after_sync(self):
        self._forward_lock.release()

    # Inference side — wrap each generate() call.
    def begin_generation(self) -> None:
        self._forward_lock.acquire()

    def end_generation(self) -> None:
        try:
            self._forward_lock.release()
        except ValueError:
            pass

    def forward_pass(self):
        """Async context manager for in-process tests."""
        return _ForwardPassContext(self._forward_lock)


class _ForwardPassContext:
    def __init__(self, lock):
        self._lock = lock

    async def __aenter__(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._lock.acquire)
        return self

    async def __aexit__(self, *exc):
        try:
            self._lock.release()
        except ValueError:
            pass
