"""Soft drain: stop accepting new requests, drain in-flight, then sync.

This is the PRIME-RL / AReaL approach. No new generation requests are
accepted during weight sync, but in-progress sequences finish naturally.
Sync bubble is proportional to the longest in-flight sequence.
"""
import asyncio
from interrupts.base import InterruptStrategy


class SoftDrain(InterruptStrategy):
    """Uses asyncio.Event to coordinate drain.

    prepare_for_sync() clears the accepting flag, waits for in-flight to finish.
    resume_after_sync() sets the flag to resume generation.
    """

    def __init__(self):
        self._accepting = asyncio.Event()
        self._accepting.set()
        self._lock = asyncio.Lock()
        self._in_flight = 0
        self._drain_complete = asyncio.Event()
        self._drain_complete.set()

    async def prepare_for_sync(self):
        """Stop accepting new generation requests, drain in-flight."""
        self._accepting.clear()
        async with self._lock:
            if self._in_flight > 0:
                self._drain_complete.clear()
        await self._drain_complete.wait()

    async def resume_after_sync(self):
        """Resume accepting generation requests."""
        self._accepting.set()

    async def wait_if_paused(self):
        """Called by inference worker before starting a new generation."""
        await self._accepting.wait()

    async def begin_generation(self):
        """Called when inference worker starts generating."""
        async with self._lock:
            self._in_flight += 1

    async def end_generation(self):
        """Called when inference worker finishes generating."""
        async with self._lock:
            self._in_flight -= 1
            if self._in_flight <= 0:
                self._in_flight = 0
                self._drain_complete.set()

    @property
    def is_accepting(self) -> bool:
        return self._accepting.is_set()
