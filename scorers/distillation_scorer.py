"""Teacher model forward pass scoring for distillation."""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.types import Rollout, ScoredRollout
from scorers.base import Scorer


class TeacherManager:
    """Manages teacher model lifecycle for distillation.

    Supports:
    - External teacher (larger model)
    - Self-distillation (snapshot student every N steps, hot-swap)
    """

    def __init__(self, teacher_model_name: str = None, device: str = "cuda:1",
                 dtype: torch.dtype = torch.bfloat16, snapshot_every: int = 10):
        self.device = device
        self.dtype = dtype
        self.snapshot_every = snapshot_every
        self.model = None
        self.tokenizer = None
        self._step_count = 0

        if teacher_model_name:
            self.model = AutoModelForCausalLM.from_pretrained(
                teacher_model_name, torch_dtype=dtype, trust_remote_code=True
            ).to(device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                teacher_model_name, trust_remote_code=True
            )

    def snapshot_student(self, student_model: nn.Module):
        """Create a frozen copy of the student as teacher (self-distillation)."""
        self._step_count += 1
        if self._step_count % self.snapshot_every != 0:
            return
        self.model = copy.deepcopy(student_model).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def get_logprobs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get per-token logprobs from teacher model."""
        if self.model is None:
            raise RuntimeError("No teacher model loaded")
        input_ids = input_ids.unsqueeze(0).to(self.device)
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits[0, :-1]  # [seq_len-1, vocab_size]
        target_ids = input_ids[0, 1:]  # [seq_len-1]
        log_probs = F.log_softmax(logits, dim=-1)
        token_logprobs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        return token_logprobs.cpu()


class DistillationScorer(Scorer):
    """Score rollouts using teacher model logprobs.

    Returns teacher logprobs as the scoring signal.
    Reward is the mean teacher logprob (higher = teacher agrees more).
    """

    def __init__(self, teacher_manager: TeacherManager):
        self.teacher = teacher_manager

    async def score(self, rollout: Rollout) -> ScoredRollout:
        input_ids = torch.cat([rollout.prompt_ids, rollout.completion_ids])
        teacher_lp = self.teacher.get_logprobs(input_ids)

        # Only keep logprobs for completion tokens
        prompt_len = rollout.prompt_ids.shape[0]
        completion_teacher_lp = teacher_lp[prompt_len - 1:]  # shifted by 1

        # Reward = mean teacher logprob for completion
        reward = completion_teacher_lp.mean().item()

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
            teacher_logprobs=completion_teacher_lp,
        )
