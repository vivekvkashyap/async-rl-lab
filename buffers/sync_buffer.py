import asyncio
from typing import List
from core.types import ScoredRollout
from buffers.base import RolloutBuffer


class SyncBuffer(RolloutBuffer):
    """No async overlap. Simple list storage, blocking get."""

    def __init__(self):
        self._buffer: List[ScoredRollout] = []

    async def put(self, rollout: ScoredRollout) -> None:
        self._buffer.append(rollout)

    async def get(self, batch_size: int) -> List[ScoredRollout]:
        while len(self._buffer) < batch_size:
            await asyncio.sleep(0.01)
        batch = self._buffer[:batch_size]
        self._buffer = self._buffer[batch_size:]
        return batch

    def size(self) -> int:
        return len(self._buffer)
