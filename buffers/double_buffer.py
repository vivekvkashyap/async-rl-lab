"""Pattern 2: one-step-ahead prefetch via concurrent.futures."""
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from typing import List, Optional, Callable
from core.types import ScoredRollout
from buffers.base import RolloutBuffer


class DoubleBuffer(RolloutBuffer):
    """Double-buffered rollout storage.

    At the start of training step N, submit generation for batch N+1.
    Train on current batch. After training, collect next batch from future.
    """

    def __init__(self):
        self._buffer: List[ScoredRollout] = []
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._prefetch_future: Optional[Future] = None
        self._lock = asyncio.Lock()

    async def put(self, rollout: ScoredRollout) -> None:
        async with self._lock:
            self._buffer.append(rollout)

    async def get(self, batch_size: int) -> List[ScoredRollout]:
        # If we have a prefetched result, use it
        if self._prefetch_future is not None:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._prefetch_future.result)
            self._prefetch_future = None
            if len(result) >= batch_size:
                return result[:batch_size]

        # Otherwise wait for enough rollouts
        while True:
            async with self._lock:
                if len(self._buffer) >= batch_size:
                    batch = self._buffer[:batch_size]
                    self._buffer = self._buffer[batch_size:]
                    return batch
            await asyncio.sleep(0.01)

    def prefetch(self, generate_fn: Callable, *args) -> None:
        """Submit generation for next batch in background thread."""
        self._prefetch_future = self._executor.submit(generate_fn, *args)

    def size(self) -> int:
        return len(self._buffer)

    def shutdown(self):
        self._executor.shutdown(wait=False)
