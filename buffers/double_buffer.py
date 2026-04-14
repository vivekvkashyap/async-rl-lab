"""Pattern 2: double-buffered prefetch.

A background drain thread continuously pulls ScoredRollouts from the shared
mp.Queue into an in-process staging list, so the coordinator can fetch a
training batch without waiting on cross-process IPC. This overlaps batch N
training with batch N+1 collection.
"""
import asyncio
import threading
import time
import multiprocessing as mp
from typing import List
from core.types import ScoredRollout
from buffers.base import RolloutBuffer


class DoubleBuffer(RolloutBuffer):
    """Prefetches rollouts into an in-memory list via a background thread."""

    def __init__(self):
        self._queue = mp.Queue(maxsize=0)
        self._staging: List[ScoredRollout] = []
        self._staging_lock = threading.Lock()
        self._stop = threading.Event()
        self._drain_thread = threading.Thread(
            target=self._drain_loop, daemon=True, name="DoubleBuffer-drain"
        )
        self._drain_thread.start()

    def _drain_loop(self):
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except Exception:
                continue
            with self._staging_lock:
                self._staging.append(item)

    async def put(self, rollout: ScoredRollout) -> None:
        """Direct staging insert (test/in-process path)."""
        with self._staging_lock:
            self._staging.append(rollout)

    async def get(self, batch_size: int) -> List[ScoredRollout]:
        while True:
            with self._staging_lock:
                if len(self._staging) >= batch_size:
                    batch = self._staging[:batch_size]
                    self._staging = self._staging[batch_size:]
                    return batch
            await asyncio.sleep(0.01)

    async def collect(self, batch_size: int, timeout: float = 120.0) -> List[ScoredRollout]:
        deadline = time.time() + timeout
        warned = False
        while time.time() < deadline:
            with self._staging_lock:
                if len(self._staging) >= batch_size:
                    batch = self._staging[:batch_size]
                    self._staging = self._staging[batch_size:]
                    return batch
                current = len(self._staging)
            if not warned and (deadline - time.time()) < timeout - 10:
                print(f"[buffer] DoubleBuffer waiting... staged={current}/{batch_size}")
                warned = True
            await asyncio.sleep(0.02)
        with self._staging_lock:
            batch = self._staging
            self._staging = []
            return batch

    def size(self) -> int:
        with self._staging_lock:
            return len(self._staging)

    def shutdown(self):
        self._stop.set()
