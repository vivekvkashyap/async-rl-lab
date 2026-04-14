"""Soft drain: stop accepting new requests, drain in-flight, then sync.

The PRIME-RL / AReaL approach. Backed by multiprocessing primitives so that
inference workers in a separate process can honor the drain signal.

Flow:
  1. Coordinator calls prepare_for_sync() — clears `accepting` event,
     waits on `drained` event.
  2. Inference worker loop: before each generation, `wait_if_paused()`
     blocks while `accepting` is clear. When `accepting` is set,
     `begin_generation()` increments `in_flight`. `end_generation()`
     decrements and sets `drained` if it reaches zero.
  3. Coordinator pushes weights, then `resume_after_sync()` sets `accepting`.
"""
import asyncio
import multiprocessing as mp
from interrupts.base import InterruptStrategy


class SoftDrain(InterruptStrategy):
    def __init__(self):
        ctx = mp.get_context()
        self._accepting = ctx.Event()
        self._accepting.set()
        self._drained = ctx.Event()
        self._drained.set()
        self._in_flight = ctx.Value("i", 0)

    async def prepare_for_sync(self):
        self._accepting.clear()
        with self._in_flight.get_lock():
            if self._in_flight.value > 0:
                self._drained.clear()
            else:
                self._drained.set()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._drained.wait)

    async def resume_after_sync(self):
        self._accepting.set()

    # Inference side — sync methods called from the vLLM worker process.
    def wait_if_paused(self) -> None:
        self._accepting.wait()

    def begin_generation(self) -> None:
        with self._in_flight.get_lock():
            self._in_flight.value += 1

    def end_generation(self) -> None:
        with self._in_flight.get_lock():
            self._in_flight.value -= 1
            if self._in_flight.value <= 0:
                self._in_flight.value = 0
                self._drained.set()

    @property
    def is_accepting(self) -> bool:
        return self._accepting.is_set()
