from abc import ABC, abstractmethod
import torch.nn as nn


class WeightSyncer(ABC):
    @abstractmethod
    async def push(self, model: nn.Module, version: int) -> float:
        """Push updated weights from trainer. Returns sync duration in seconds."""
        ...

    @abstractmethod
    async def pull(self, model: nn.Module) -> int:
        """Pull latest weights into model. Returns the version pulled."""
        ...
