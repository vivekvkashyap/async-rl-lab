"""Pattern 3: multiprocessing.Queue(maxsize=K) with backpressure.

The buffer owns the mp.Queue that inference workers push into. When the
queue is full, inference blocks on put — that's the backpressure this
axis is meant to study.
"""
import multiprocessing as mp
from buffers.base import RolloutBuffer


class BoundedQueueBuffer(RolloutBuffer):
    """Bounded mp.Queue. Producer blocks when full (backpressure)."""

    def __init__(self, maxsize: int = 4):
        self._queue = mp.Queue(maxsize=maxsize)
        self.maxsize = maxsize
