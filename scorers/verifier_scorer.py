from core.types import Rollout, ScoredRollout
from scorers.base import Scorer
from utils.gsm8k import extract_model_answer, compute_format_reward


CORRECTNESS_WEIGHT = 1.0
FORMAT_WEIGHT = 0.1


class VerifierScorer(Scorer):
    """Rule-based reward for GSM8K.

    Combines a correctness reward (answer matches ground truth) with a
    format reward (answer is emitted in the '#### <number>' shape the
    prompt asks for). Format reward is a shaping signal weighted well
    below correctness so it never dominates the objective.

        reward = CORRECTNESS_WEIGHT * correct + FORMAT_WEIGHT * format_score

    With the defaults above the reward range is [0.0, 1.1].
    """

    async def score(self, rollout: Rollout) -> ScoredRollout:
        model_answer = extract_model_answer(rollout.completion)

        correct = False
        if model_answer is not None and rollout.ground_truth is not None:
            correct = abs(model_answer - rollout.ground_truth) < 1e-3

        correctness_reward = 1.0 if correct else 0.0
        format_reward = compute_format_reward(rollout.completion)

        reward = CORRECTNESS_WEIGHT * correctness_reward + FORMAT_WEIGHT * format_reward

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
