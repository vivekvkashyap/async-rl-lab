"""Integration test: trainer with GRPO and IPO algorithms."""
import asyncio
import sys
import os
import tempfile
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import ScoredRollout, TrainingBatch
from core.trainer import GRPOTrainer, compute_grpo_advantages, compute_ipo_advantages
from staleness.no_filter import NoFilter


def make_batch(n: int = 8, group_size: int = 4) -> TrainingBatch:
    """Create a synthetic training batch with known rewards."""
    rollouts = []
    n_prompts = n // group_size
    for p in range(n_prompts):
        for g in range(group_size):
            reward = 1.0 if g == 0 else 0.0
            seq_len = 10
            rollouts.append(ScoredRollout(
                prompt="What is 2+2?",
                prompt_ids=torch.randint(0, 1000, (5,)),
                completion="The answer is 4. #### 4",
                completion_ids=torch.randint(0, 1000, (seq_len,)),
                logprobs=torch.randn(seq_len) * 0.1 - 2.0,
                model_version=0,
                generated_at=0.0,
                prompt_id=f"prompt_{p}",
                ground_truth=4.0,
                reward=reward,
            ))
    return TrainingBatch(rollouts=rollouts)


class TestAdvantages:
    def test_grpo_advantages(self):
        rollouts = []
        for g in range(4):
            rollouts.append(ScoredRollout(
                prompt="q", prompt_ids=torch.tensor([1]),
                completion="a", completion_ids=torch.tensor([2]),
                logprobs=torch.tensor([-1.0]),
                model_version=0, generated_at=0.0,
                prompt_id="p1", ground_truth=1.0,
                reward=float(g),
            ))
        advantages = compute_grpo_advantages(rollouts)
        # Should be normalized: mean~0, std~1
        assert abs(sum(advantages) / len(advantages)) < 1e-6
        assert advantages[3] > advantages[0]

    def test_ipo_advantages(self):
        rollouts = []
        for g in range(4):
            rollouts.append(ScoredRollout(
                prompt="q", prompt_ids=torch.tensor([1]),
                completion="a", completion_ids=torch.tensor([2]),
                logprobs=torch.tensor([-1.0]),
                model_version=0, generated_at=0.0,
                prompt_id="p1", ground_truth=1.0,
                reward=float(g),
            ))
        advantages = compute_ipo_advantages(rollouts)
        # IPO: reward - baseline, no std normalization
        baseline = sum(range(4)) / 4  # 1.5
        assert abs(advantages[0] - (0.0 - baseline)) < 1e-6
        assert abs(advantages[3] - (3.0 - baseline)) < 1e-6
        assert advantages[3] > advantages[0]


class TestTrainer:
    def _get_model_and_tokenizer(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "sshleifer/tiny-gpt2", trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                "sshleifer/tiny-gpt2", trust_remote_code=True
            )
        except Exception:
            return None, None
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model.to("cpu"), tokenizer

    def test_grpo_loss_decreases(self):
        """Train with GRPO on same batch, loss should decrease."""
        torch.manual_seed(42)
        model, tokenizer = self._get_model_and_tokenizer()
        if model is None:
            print("Skipping test_grpo_loss_decreases: tiny-gpt2 not available")
            return

        trainer = GRPOTrainer(
            model=model, tokenizer=tokenizer, device="cpu",
            lr=1e-4, algorithm="grpo", clip_eps=0.2, kl_coeff=0.01,
        )

        batch = make_batch(8, 4)
        losses = []
        for _ in range(5):
            metrics = trainer.train_step(batch)
            losses.append(metrics["training_loss"])

        print(f"GRPO losses: {losses}")
        assert losses[-1] < losses[0], f"GRPO loss should decrease: {losses}"

    def test_ipo_trains(self):
        """Train with IPO, verify it runs and produces expected metrics."""
        torch.manual_seed(42)
        model, tokenizer = self._get_model_and_tokenizer()
        if model is None:
            print("Skipping test_ipo_trains: tiny-gpt2 not available")
            return

        trainer = GRPOTrainer(
            model=model, tokenizer=tokenizer, device="cpu",
            lr=1e-4, algorithm="ipo",
            ipo_mask_low=0.2, ipo_mask_high=0.2,
            adv_tau=1.0, kl_tau=1e-3,
        )

        batch = make_batch(8, 4)
        for _ in range(3):
            metrics = trainer.train_step(batch)

        # IPO-specific metrics should be present
        assert "ipo_masked_frac" in metrics
        assert "ipo_masked_high" in metrics
        assert "ipo_masked_low" in metrics
        assert "mismatch_kl" in metrics
        assert "kl_loss" in metrics
        print(f"IPO metrics: loss={metrics['training_loss']:.4f}, "
              f"masked={metrics['ipo_masked_frac']:.4f}, "
              f"kl={metrics['kl_loss']:.6f}")


if __name__ == "__main__":
    t1 = TestAdvantages()
    t1.test_grpo_advantages()
    print("test_grpo_advantages PASSED")
    t1.test_ipo_advantages()
    print("test_ipo_advantages PASSED")

    t2 = TestTrainer()
    t2.test_grpo_loss_decreases()
    print("test_grpo_loss_decreases PASSED")
    t2.test_ipo_trains()
    print("test_ipo_trains PASSED")
    print("\nAll trainer tests passed!")
