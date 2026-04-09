"""Tests for staleness manager implementations."""
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import ScoredRollout
from staleness.no_filter import NoFilter


def make_rollout(version: int = 0, reward: float = 1.0) -> ScoredRollout:
    return ScoredRollout(
        prompt="test",
        prompt_ids=torch.tensor([1, 2]),
        completion="answer",
        completion_ids=torch.tensor([3, 4]),
        logprobs=torch.tensor([-1.0, -2.0]),
        model_version=version,
        generated_at=0.0,
        prompt_id="p1",
        ground_truth=42.0,
        reward=reward,
    )


class TestNoFilter:
    def test_passes_all(self):
        nf = NoFilter()
        rollouts = [make_rollout(version=v) for v in range(5)]
        batch = nf.process(rollouts, current_version=10)
        assert len(batch.rollouts) == 5
        assert batch.is_weights is None

    def test_empty(self):
        nf = NoFilter()
        batch = nf.process([], current_version=0)
        assert len(batch.rollouts) == 0


if __name__ == "__main__":
    t = TestNoFilter()
    t.test_passes_all()
    print("test_passes_all PASSED")
    t.test_empty()
    print("test_empty PASSED")
    print("\nAll staleness tests passed!")
