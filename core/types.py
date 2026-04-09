from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import torch


@dataclass
class Rollout:
    prompt: str
    prompt_ids: torch.Tensor
    completion: str
    completion_ids: torch.Tensor
    logprobs: torch.Tensor  # per-token logprobs from generation policy
    model_version: int  # integer version of policy that generated this
    generated_at: float  # timestamp
    prompt_id: str = ""  # groups rollouts for the same prompt (for GRPO)
    ground_truth: Optional[float] = None  # for verifier scoring (e.g. GSM8K answer)
    sampling_mask: Optional[torch.Tensor] = None  # top-p/top-k mask if recorded


@dataclass
class ScoredRollout(Rollout):
    reward: Optional[float] = None
    teacher_logprobs: Optional[torch.Tensor] = None  # for distillation scorer


@dataclass
class TrainingBatch:
    rollouts: List[ScoredRollout]
    is_weights: Optional[torch.Tensor] = None  # importance sampling weights
