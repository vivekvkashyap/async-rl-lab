"""Implicit continuation: per-forward-pass lock, never stop generating.

This is the PipelineRL approach. Generation never stops — weights are
swapped between forward passes (~1-10ms gap). A lock ensures weights
aren't modified during a single forward pass.
"""
import asyncio
from interrupts.base import InterruptStrategy


class ImplicitContinuation(InterruptStrategy):
    """Per-forward-pass lock. Weights change between token steps.

    Usage:
      - Weight sync calls prepare_for_sync() / resume_after_sync()
      - Inference worker wraps each forward pass with:
          async with interrupt.forward_pass():
              output = model(input)
    """

    def __init__(self):
        self._forward_lock = asyncio.Lock()

    async def prepare_for_sync(self):
        """Acquire the forward lock — waits for current forward pass to finish."""
        await self._forward_lock.acquire()

    async def resume_after_sync(self):
        """Release the forward lock — next forward pass can proceed."""
        self._forward_lock.release()

    class _ForwardPassContext:
        """Async context manager for forward pass locking."""
        def __init__(self, lock: asyncio.Lock):
            self._lock = lock

        async def __aenter__(self):
            await self._lock.acquire()
            return self

        async def __aexit__(self, *exc):
            self._lock.release()

    def forward_pass(self):
        """Return an async context manager that holds the forward lock.

        Usage:
            async with interrupt.forward_pass():
                output = model(input)
        """
        return self._ForwardPassContext(self._forward_lock)
