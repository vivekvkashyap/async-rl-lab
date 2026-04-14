"""Pattern 1: unbounded shared mp.Queue. No backpressure."""
import multiprocessing as mp
from buffers.base import RolloutBuffer


class SyncBuffer(RolloutBuffer):
    """Unbounded mp.Queue. Producer (inference) never blocks on put."""

    def __init__(self):
        self._queue = mp.Queue(maxsize=0)
