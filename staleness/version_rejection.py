"""Drop rollouts where current_version - rollout.model_version > max_lag."""
from typing import List
from core.types import ScoredRollout, TrainingBatch
from staleness.base import StalenessManager


class VersionRejection(StalenessManager):
    """Per-sample version rejection.

    Drops any rollout where staleness exceeds max_lag.
    Per-sample, not per-batch — important because asymmetric filtering
    creates intra-batch version spread (blog section 5.1).
    """

    def __init__(self, max_lag: int = 3):
        self.max_lag = max_lag
        self.total_dropped = 0

    def process(self, rollouts: List[ScoredRollout], current_version: int) -> TrainingBatch:
        accepted = []
        dropped = 0
        for r in rollouts:
            staleness = current_version - r.model_version
            if staleness <= self.max_lag:
                accepted.append(r)
            else:
                dropped += 1

        self.total_dropped += dropped
        return TrainingBatch(rollouts=accepted, is_weights=None)
