from core.types import Rollout, ScoredRollout
from scorers.base import Scorer
from utils.gsm8k import extract_model_answer


class VerifierScorer(Scorer):
    """Rule-based reward for GSM8K: extract answer and check correctness."""

    async def score(self, rollout: Rollout) -> ScoredRollout:
        model_answer = extract_model_answer(rollout.completion)

        correct = False
        if model_answer is not None and rollout.ground_truth is not None:
            correct = abs(model_answer - rollout.ground_truth) < 1e-3

        reward = 1.0 if correct else 0.0
        return ScoredRollout(
            prompt=rollout.prompt,
            prompt_ids=rollout.prompt_ids,
            completion=rollout.completion,
            completion_ids=rollout.completion_ids,
            logprobs=rollout.logprobs,
            model_version=rollout.model_version,
            generated_at=rollout.generated_at,
            prompt_id=rollout.prompt_id,
            ground_truth=rollout.ground_truth,
            sampling_mask=rollout.sampling_mask,
            reward=reward,
            teacher_logprobs=None,
        )
