import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, List, Optional
from collections import defaultdict
from core.types import ScoredRollout, TrainingBatch


def selective_log_softmax(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Compute log_softmax and gather only for target token ids.
    Memory efficient: avoids materializing full vocab-size log_softmax.

    Args:
        logits: [batch, seq_len, vocab_size]
        target_ids: [batch, seq_len]
    Returns:
        log_probs: [batch, seq_len] — logprob of each target token
    """
    log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)
    return log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


def compute_grpo_advantages(rollouts: List[ScoredRollout]) -> List[float]:
    """GRPO: group-relative advantages with std normalization."""
    groups = defaultdict(list)
    for i, r in enumerate(rollouts):
        groups[r.prompt_id].append((i, r.reward or 0.0))

    advantages = [0.0] * len(rollouts)
    for prompt_id, members in groups.items():
        rewards = [r for _, r in members]
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
        std_r = max(std_r, 1e-8)
        for idx, reward in members:
            advantages[idx] = (reward - mean_r) / std_r

    return advantages


def compute_ipo_advantages(rollouts: List[ScoredRollout]) -> List[float]:
    """IPO: reward minus per-problem baseline (no std normalization).

    Following prime-rl's default_advantage_fn:
        advantage = reward - mean(rewards_for_same_prompt)
    """
    groups = defaultdict(list)
    for i, r in enumerate(rollouts):
        groups[r.prompt_id].append((i, r.reward or 0.0))

    advantages = [0.0] * len(rollouts)
    for prompt_id, members in groups.items():
        rewards = [r for _, r in members]
        baseline = sum(rewards) / len(rewards)
        for idx, reward in members:
            advantages[idx] = reward - baseline

    return advantages


class GRPOTrainer:
    """RL trainer supporting GRPO and IPO (INTELLECT Policy Optimization).

    Algorithms:
      - grpo: Standard clipped surrogate loss (PPO-style)
      - ipo: Probability-difference masking + quadratic KL (prime-rl)
             More stable for async training.

    Supports both single-GPU and FSDP multi-GPU training.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda:0",
        lr: float = 1e-6,
        algorithm: str = "ipo",
        # GRPO params
        clip_eps: float = 0.2,
        kl_coeff: float = 0.01,
        # IPO params (prime-rl defaults)
        ipo_mask_low: float = 0.2,
        ipo_mask_high: float = 0.2,
        adv_tau: float = 1.0,
        kl_tau: float = 1e-3,
        # Shared
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.algorithm = algorithm
        self.max_grad_norm = max_grad_norm
        self.version = 0

        # GRPO params
        self.clip_eps = clip_eps
        self.kl_coeff = kl_coeff

        # IPO params
        self.ipo_mask_low = ipo_mask_low
        self.ipo_mask_high = ipo_mask_high
        self.adv_tau = adv_tau
        self.kl_tau = kl_tau

        # Detect FSDP wrapping
        from utils.fsdp import is_fsdp_wrapped
        self.is_fsdp = is_fsdp_wrapped(model)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def prepare_batch(self, batch: TrainingBatch) -> Dict[str, torch.Tensor]:
        """Convert TrainingBatch into padded tensors for training."""
        rollouts = batch.rollouts

        if self.algorithm == "ipo":
            advantages = compute_ipo_advantages(rollouts)
        else:
            advantages = compute_grpo_advantages(rollouts)

        all_input_ids = []
        all_old_logprobs = []
        all_advantages = []
        all_loss_mask = []

        max_len = 0
        for r in rollouts:
            total_len = r.prompt_ids.shape[0] + r.completion_ids.shape[0]
            max_len = max(max_len, total_len)

        for i, r in enumerate(rollouts):
            prompt_len = r.prompt_ids.shape[0]
            comp_len = r.completion_ids.shape[0]
            total_len = prompt_len + comp_len
            pad_len = max_len - total_len

            input_ids = torch.cat([r.prompt_ids, r.completion_ids])
            if pad_len > 0:
                input_ids = F.pad(input_ids, (0, pad_len), value=self.tokenizer.pad_token_id or 0)

            old_lp = torch.zeros(max_len, dtype=torch.float32)
            old_lp[prompt_len:total_len] = r.logprobs[:comp_len]

            mask = torch.zeros(max_len, dtype=torch.float32)
            mask[prompt_len:total_len] = 1.0

            adv = torch.zeros(max_len, dtype=torch.float32)
            adv[prompt_len:total_len] = advantages[i]

            all_input_ids.append(input_ids)
            all_old_logprobs.append(old_lp)
            all_loss_mask.append(mask)
            all_advantages.append(adv)

        result = {
            "input_ids": torch.stack(all_input_ids).to(self.device),
            "old_logprobs": torch.stack(all_old_logprobs).to(self.device),
            "loss_mask": torch.stack(all_loss_mask).to(self.device),
            "advantages": torch.stack(all_advantages).to(self.device),
        }

        if batch.is_weights is not None:
            result["is_weights"] = batch.is_weights.to(self.device)

        return result

    def _compute_loss_grpo(self, current_logprobs, old_logprobs, loss_mask,
                            advantages, is_weights):
        """Standard GRPO: clipped surrogate + approximate KL."""
        log_ratio = current_logprobs - old_logprobs
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2)

        masked_loss = (policy_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        kl = (ratio - log_ratio - 1) * loss_mask
        kl_loss = kl.sum() / loss_mask.sum().clamp(min=1)

        if is_weights is not None:
            w = is_weights.unsqueeze(-1).expand_as(policy_loss)
            masked_loss = (policy_loss * loss_mask * w).sum() / (loss_mask * w).sum().clamp(min=1)

        total_loss = masked_loss + self.kl_coeff * kl_loss

        with torch.no_grad():
            clip_frac = ((ratio - 1).abs() > self.clip_eps).float()
            clip_frac = (clip_frac * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        metrics = {
            "policy_loss": masked_loss.item(),
            "kl_loss": kl_loss.item(),
            "clip_fraction": clip_frac.item(),
        }
        return total_loss, metrics

    def _compute_loss_ipo(self, current_logprobs, old_logprobs, loss_mask,
                           advantages, is_weights):
        """IPO (INTELLECT Policy Optimization): probability-difference masking + quadratic KL.

        Combines DPPO-Binary TV Loss with Kimi-K2.5 KL Loss.
        Unlike GRPO's ratio clipping, IPO masks tokens where the probability
        difference exceeds thresholds. This is more stable for async RL because
        policy updates are not well-predicted by the advantage sign.
        """
        # Probability-difference masking (trust region via TV distance)
        trainer_probs = torch.exp(current_logprobs)
        inference_probs = torch.exp(old_logprobs)
        probs_diff = trainer_probs - inference_probs

        ipo_mask_high = probs_diff > self.ipo_mask_high
        ipo_mask_low = probs_diff < -self.ipo_mask_low
        ipo_invalid_mask = ipo_mask_high | ipo_mask_low

        # Keep mask: trainable tokens that are within trust region
        loss_mask_bool = loss_mask.bool()
        keep_mask = loss_mask_bool & ~ipo_invalid_mask

        # Importance ratio
        log_ratio = current_logprobs - old_logprobs
        importance_ratio = torch.exp(log_ratio)

        # Scale advantages
        scaled_advantages = self.adv_tau * advantages

        # Policy gradient loss (only on kept tokens)
        pg_loss = keep_mask.float() * scaled_advantages * importance_ratio

        # Quadratic KL regularization (on all trainable tokens)
        kl_loss = loss_mask * log_ratio ** 2

        # Apply IS weights if provided
        if is_weights is not None:
            w = is_weights.unsqueeze(-1).expand_as(pg_loss)
            pg_loss = pg_loss * w
            kl_loss = kl_loss * w

        total_loss = (-pg_loss + self.kl_tau * kl_loss).sum() / loss_mask.sum().clamp(min=1)

        # Metrics
        with torch.no_grad():
            mask_frac = ipo_invalid_mask.float()
            mask_frac = (mask_frac * loss_mask).sum() / loss_mask.sum().clamp(min=1)
            mask_high_frac = (ipo_mask_high.float() * loss_mask).sum() / loss_mask.sum().clamp(min=1)
            mask_low_frac = (ipo_mask_low.float() * loss_mask).sum() / loss_mask.sum().clamp(min=1)

            mismatch_kl = (importance_ratio - log_ratio - 1) * loss_mask
            mismatch_kl_mean = mismatch_kl.sum() / loss_mask.sum().clamp(min=1)

            pg_loss_mean = (keep_mask.float() * scaled_advantages * importance_ratio).sum() / keep_mask.float().sum().clamp(min=1)
            kl_loss_mean = (loss_mask * log_ratio ** 2).sum() / loss_mask.sum().clamp(min=1)

        metrics = {
            "policy_loss": -pg_loss_mean.item(),
            "kl_loss": kl_loss_mean.item(),
            "ipo_masked_frac": mask_frac.item(),
            "ipo_masked_high": mask_high_frac.item(),
            "ipo_masked_low": mask_low_frac.item(),
            "mismatch_kl": mismatch_kl_mean.item(),
        }
        return total_loss, metrics

    def train_step(self, batch: TrainingBatch) -> Dict[str, float]:
        """One gradient update step (GRPO or IPO based on self.algorithm).

        When FSDP is active, rank 0 broadcasts a signal and batch data
        to follower ranks so they can participate in forward/backward.
        """
        self.model.train()
        tensors = self.prepare_batch(batch)

        # Broadcast to FSDP follower ranks
        if self.is_fsdp:
            from utils.fsdp import broadcast_batch_tensors
            signal = torch.ones(1, dtype=torch.long, device=self.device)
            dist.broadcast(signal, src=0)
            broadcast_batch_tensors(tensors, self.device)

        input_ids = tensors["input_ids"]
        old_logprobs = tensors["old_logprobs"]
        loss_mask = tensors["loss_mask"]
        advantages = tensors["advantages"]
        is_weights = tensors.get("is_weights")

        # Attention mask: non-pad tokens
        attention_mask = (input_ids != (self.tokenizer.pad_token_id or 0)).long()

        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]
        current_logprobs = selective_log_softmax(logits, target_ids)

        # Align masks with shifted sequence
        old_logprobs = old_logprobs[:, 1:]
        loss_mask = loss_mask[:, 1:]
        advantages = advantages[:, 1:]

        # Compute loss based on algorithm
        if self.algorithm == "ipo":
            total_loss, loss_metrics = self._compute_loss_ipo(
                current_logprobs, old_logprobs, loss_mask, advantages, is_weights
            )
        else:
            total_loss, loss_metrics = self._compute_loss_grpo(
                current_logprobs, old_logprobs, loss_mask, advantages, is_weights
            )

        # Backward + clip grad + step
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.is_fsdp:
            grad_norm = self.model.clip_grad_norm_(self.max_grad_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
        self.optimizer.step()
        self.version += 1

        rewards = [r.reward or 0.0 for r in batch.rollouts]
        return {
            "training_loss": total_loss.item(),
            **loss_metrics,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "reward_mean": sum(rewards) / len(rewards),
            "reward_max": max(rewards),
            "model_version": self.version,
        }
