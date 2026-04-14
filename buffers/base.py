"""Buffer base class.

The buffer mediates between inference producers (separate processes) and
the trainer consumer. Producers push into `get_producer_queue()` — the buffer
owns this mp.Queue so that `maxsize=` backpressure actually applies to the
cross-process path.

The coordinator calls `collect(batch_size, timeout)` to pull a batch for
training. This is the single entry point used during the training loop.

Backwards-compatible `put/get` are kept for unit tests that drive the buffer
inside a single process.
"""
import asyncio
import time
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import List, Optional
from core.types import ScoredRollout


class RolloutBuffer(ABC):
    """Base class for rollout buffers."""

    # Subclass must set self._queue (mp.Queue) or override get/put/collect.
    _queue: Optional[mp.Queue] = None

    def get_producer_queue(self) -> Optional[mp.Queue]:
        """mp.Queue inference workers push ScoredRollouts into.

        Returns None for buffers using a non-queue transport (e.g. Redis).
        """
        return self._queue

    async def put(self, rollout: ScoredRollout) -> None:
        """In-process insert. Used by unit tests; inference workers should
        push to get_producer_queue() directly."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._queue.put, rollout)

    async def get(self, batch_size: int) -> List[ScoredRollout]:
        """Blocking fetch of exactly batch_size items."""
        loop = asyncio.get_event_loop()
        batch: List[ScoredRollout] = []
        while len(batch) < batch_size:
            item = await loop.run_in_executor(None, self._queue.get)
            batch.append(item)
        return batch

    async def collect(self, batch_size: int, timeout: float = 120.0) -> List[ScoredRollout]:
        """Best-effort collect up to batch_size items before deadline.

        Returns what's available at timeout (possibly fewer than batch_size).
        The coordinator uses this — it prefers a short batch over deadlock.
        """
        loop = asyncio.get_event_loop()
        batch: List[ScoredRollout] = []
        deadline = time.time() + timeout
        empty_retries = 0
        while len(batch) < batch_size and time.time() < deadline:
            remaining = max(0.1, deadline - time.time())
            try:
                item = await loop.run_in_executor(
                    None, lambda: self._queue.get(timeout=min(2.0, remaining))
                )
                batch.append(item)
                empty_retries = 0
            except Exception:
                empty_retries += 1
                if empty_retries == 5:
                    print(f"[buffer] waiting for rollouts... ({len(batch)}/{batch_size} collected)")
        return batch

    def size(self) -> int:
        try:
            return self._queue.qsize()
        except Exception:
            return 0

    def shutdown(self) -> None:
        """Optional cleanup hook."""
        pass
