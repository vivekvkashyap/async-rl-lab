"""Version rejection + IS reweighting combined (PRIME-RL approach)."""
import torch
import torch.nn as nn
from typing import List
from core.types import ScoredRollout, TrainingBatch
from staleness.base import StalenessManager
from staleness.version_rejection import VersionRejection
from staleness.is_reweighting import ISReweighting


class HybridStaleness(StalenessManager):
    """First applies version rejection, then IS reweighting on survivors.

    This is what PRIME-RL does: hard-drop samples beyond max_lag,
    then correct remaining stale samples with importance sampling.
    """

    def __init__(self, max_lag: int = 3, is_clip_eps: float = 0.2,
                 model: nn.Module = None, tokenizer=None, device: str = "cuda:0"):
        self.rejector = VersionRejection(max_lag=max_lag)
        self.reweighter = ISReweighting(
            clip_eps=is_clip_eps, model=model,
            tokenizer=tokenizer, device=device,
        )

    def set_model(self, model: nn.Module, tokenizer, device: str):
        """Set the model for IS reweighting."""
        self.reweighter.set_model(model, tokenizer, device)

    def process(self, rollouts: List[ScoredRollout], current_version: int) -> TrainingBatch:
        # Step 1: Hard rejection
        rejected_batch = self.rejector.process(rollouts, current_version)
        surviving = rejected_batch.rollouts

        if not surviving:
            return TrainingBatch(rollouts=[], is_weights=None)

        # Step 2: IS reweighting on survivors
        return self.reweighter.process(surviving, current_version)

    @property
    def total_dropped(self):
        return self.rejector.total_dropped

    @property
    def is_weight_variances(self):
        return self.reweighter.is_weight_variances
