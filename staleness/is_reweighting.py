"""Importance sampling ratio correction for stale rollouts.

Computes IS ratio = pi_theta(a|s) / pi_old(a|s) per token.
pi_old comes from rollout.logprobs (recorded at generation time).
pi_theta comes from a fresh forward pass with the current model.
Per-sample IS weight is the geometric mean of per-token ratios,
clipped to [1-eps, 1+eps] for stability.
"""
import torch
import torch.nn as nn
from typing import List
from core.types import ScoredRollout, TrainingBatch
from staleness.base import StalenessManager


class ISReweighting(StalenessManager):
    """Truncated importance sampling correction for off-policy rollouts."""

    def __init__(self, clip_eps: float = 0.2, model: nn.Module = None,
                 tokenizer=None, device: str = "cuda:0"):
        self.clip_eps = clip_eps
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.is_weight_variances = []

    def set_model(self, model: nn.Module, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def _compute_current_logprobs(self, rollout: ScoredRollout) -> torch.Tensor:
        """Forward pass to get current policy logprobs for completion tokens."""
        input_ids = torch.cat([rollout.prompt_ids, rollout.completion_ids]).unsqueeze(0).to(self.device)
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits[0]  # [seq_len, vocab_size]

        prompt_len = rollout.prompt_ids.shape[0]
        comp_len = rollout.completion_ids.shape[0]

        # logits[t] predicts token at position t+1
        # For completion token at position (prompt_len + t), the predicting logit is at (prompt_len + t - 1)
        current_logprobs = []
        for t in range(comp_len):
            logit_pos = prompt_len + t - 1
            # First completion token is predicted by last prompt logit
            # This is always valid since prompt_len >= 1
            log_probs = torch.log_softmax(logits[logit_pos], dim=-1)
            token_id = rollout.completion_ids[t]
            current_logprobs.append(log_probs[token_id].item())

        return torch.tensor(current_logprobs, dtype=torch.float32)

    def process(self, rollouts: List[ScoredRollout], current_version: int) -> TrainingBatch:
        if not rollouts or self.model is None:
            return TrainingBatch(rollouts=rollouts, is_weights=None)

        is_weights = []
        for r in rollouts:
            if current_version == r.model_version:
                is_weights.append(1.0)
                continue

            current_lp = self._compute_current_logprobs(r)
            old_lp = r.logprobs[:len(current_lp)]

            # Per-token log IS ratio, then geometric mean (= mean of log ratios)
            log_ratio = current_lp - old_lp
            mean_log_ratio = log_ratio.mean()

            # Clip the IS weight for stability
            is_weight = torch.exp(mean_log_ratio).clamp(
                1 - self.clip_eps, 1 + self.clip_eps
            ).item()
            is_weights.append(is_weight)

        is_weights_tensor = torch.tensor(is_weights, dtype=torch.float32)
        self.is_weight_variances.append(is_weights_tensor.var().item())

        return TrainingBatch(rollouts=rollouts, is_weights=is_weights_tensor)
