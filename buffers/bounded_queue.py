"""Pattern 3: multiprocessing.Queue(maxsize=K) with backpressure."""
import asyncio
import multiprocessing as mp
from typing import List
from core.types import ScoredRollout
from buffers.base import RolloutBuffer


class BoundedQueueBuffer(RolloutBuffer):
    """Bounded async queue backed by multiprocessing.Queue.

    When queue is full, put() blocks (backpressure).
    K is the depth parameter from blog Axis 2.
    """

    def __init__(self, maxsize: int = 4):
        self._queue = mp.Queue(maxsize=maxsize)
        self._size = mp.Value("i", 0)

    async def put(self, rollout: ScoredRollout) -> None:
        loop = asyncio.get_event_loop()
        # Run blocking put in executor to avoid blocking the event loop
        await loop.run_in_executor(None, self._queue.put, rollout)
        with self._size.get_lock():
            self._size.value += 1

    async def get(self, batch_size: int) -> List[ScoredRollout]:
        loop = asyncio.get_event_loop()
        batch = []
        while len(batch) < batch_size:
            item = await loop.run_in_executor(None, self._queue.get)
            batch.append(item)
            with self._size.get_lock():
                self._size.value -= 1
        return batch

    def size(self) -> int:
        return self._size.value

    def get_raw_queue(self) -> mp.Queue:
        """Expose raw queue for direct use in worker processes."""
        return self._queue
