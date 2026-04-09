from abc import ABC, abstractmethod
from core.types import Rollout, ScoredRollout


class Scorer(ABC):
    @abstractmethod
    async def score(self, rollout: Rollout) -> ScoredRollout:
        """Score a rollout. For GRPO: compute reward. For distillation: compute teacher logprobs."""
        ...
