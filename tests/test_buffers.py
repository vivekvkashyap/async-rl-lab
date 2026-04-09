"""Tests for buffer implementations."""
import asyncio
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import ScoredRollout
from buffers.sync_buffer import SyncBuffer


def make_rollout(prompt_id: str = "p1", reward: float = 1.0, version: int = 0) -> ScoredRollout:
    return ScoredRollout(
        prompt="test prompt",
        prompt_ids=torch.tensor([1, 2, 3]),
        completion="test completion",
        completion_ids=torch.tensor([4, 5, 6]),
        logprobs=torch.tensor([-1.0, -2.0, -3.0]),
        model_version=version,
        generated_at=0.0,
        prompt_id=prompt_id,
        ground_truth=42.0,
        reward=reward,
    )


class TestSyncBuffer:
    def test_put_and_get(self):
        buf = SyncBuffer()
        assert buf.size() == 0

        async def run():
            await buf.put(make_rollout())
            await buf.put(make_rollout())
            assert buf.size() == 2
            batch = await buf.get(2)
            assert len(batch) == 2
            assert buf.size() == 0

        asyncio.run(run())

    def test_get_blocks_until_enough(self):
        buf = SyncBuffer()

        async def run():
            # Put 1, try to get 2 — should block until second arrives
            await buf.put(make_rollout())

            async def delayed_put():
                await asyncio.sleep(0.05)
                await buf.put(make_rollout())

            task = asyncio.create_task(delayed_put())
            batch = await buf.get(2)
            assert len(batch) == 2
            await task

        asyncio.run(run())

    def test_maintains_order(self):
        buf = SyncBuffer()

        async def run():
            for i in range(5):
                await buf.put(make_rollout(reward=float(i)))
            batch = await buf.get(3)
            assert [r.reward for r in batch] == [0.0, 1.0, 2.0]
            batch2 = await buf.get(2)
            assert [r.reward for r in batch2] == [3.0, 4.0]

        asyncio.run(run())


if __name__ == "__main__":
    t = TestSyncBuffer()
    t.test_put_and_get()
    print("test_put_and_get PASSED")
    t.test_get_blocks_until_enough()
    print("test_get_blocks_until_enough PASSED")
    t.test_maintains_order()
    print("test_maintains_order PASSED")
    print("\nAll buffer tests passed!")
