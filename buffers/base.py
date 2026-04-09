from abc import ABC, abstractmethod
from typing import List
from core.types import ScoredRollout


class RolloutBuffer(ABC):
    @abstractmethod
    async def put(self, rollout: ScoredRollout) -> None:
        """Add a scored rollout to the buffer. May block if buffer is full."""
        ...

    @abstractmethod
    async def get(self, batch_size: int) -> List[ScoredRollout]:
        """Get a batch of rollouts. Blocks until batch_size rollouts are available."""
        ...

    @abstractmethod
    def size(self) -> int:
        """Current number of rollouts in buffer."""
        ...
