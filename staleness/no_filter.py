from typing import List
from core.types import ScoredRollout, TrainingBatch
from staleness.base import StalenessManager


class NoFilter(StalenessManager):
    """Accept everything, no filtering or reweighting."""

    def process(self, rollouts: List[ScoredRollout], current_version: int) -> TrainingBatch:
        return TrainingBatch(rollouts=rollouts, is_weights=None)
