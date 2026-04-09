from abc import ABC, abstractmethod
from typing import List
from core.types import ScoredRollout, TrainingBatch


class StalenessManager(ABC):
    @abstractmethod
    def process(self, rollouts: List[ScoredRollout], current_version: int) -> TrainingBatch:
        """Filter and/or reweight rollouts based on staleness.
        Returns a TrainingBatch with optional IS weights.
        """
        ...
