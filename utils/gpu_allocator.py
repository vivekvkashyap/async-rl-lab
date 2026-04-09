"""GPU allocation: maps deployment config to physical GPU IDs.

Users specify num_train_gpus and num_infer_gpus in config.
GPUs are assigned sequentially: training first, then inference.
Inference always uses vLLM, one process per GPU.

Examples:
  2 GPUs, train=1, infer=1  → train=[0], infer=[1]
  4 GPUs, train=1, infer=3  → train=[0], infer=[1,2,3]
  8 GPUs, train=2, infer=6  → train=[0,1], infer=[2,3,4,5,6,7]
"""
import torch
from dataclasses import dataclass
from typing import List


@dataclass
class GPUAllocation:
    train_gpu_ids: List[int]
    infer_gpu_ids: List[int]

    @property
    def train_device(self) -> str:
        if not self.train_gpu_ids:
            return "cpu"
        return f"cuda:{self.train_gpu_ids[0]}"

    @property
    def infer_devices(self) -> List[str]:
        return [f"cuda:{g}" for g in self.infer_gpu_ids]

    def summary(self) -> str:
        total = len(self.train_gpu_ids) + len(self.infer_gpu_ids)
        lines = [f"GPU Allocation ({total} GPUs):"]
        lines.append(f"  Training:  GPUs {self.train_gpu_ids}")
        lines.append(f"  Inference: GPUs {self.infer_gpu_ids} (vLLM)")
        return "\n".join(lines)


def allocate_gpus(config: dict) -> GPUAllocation:
    """Allocate GPUs based on deployment config.

    Falls back gracefully if fewer GPUs are available than requested.
    Requires at least 2 GPUs (1 train + 1 infer).
    """
    deployment = config.get("deployment", {})
    num_train = deployment.get("num_train_gpus", 1)
    num_infer = deployment.get("num_infer_gpus", 1)
    total_requested = num_train + num_infer

    available = torch.cuda.device_count()

    if available < 2:
        raise RuntimeError(
            f"async-rl-lab requires at least 2 GPUs (1 train + 1 infer). "
            f"Found {available}."
        )

    if total_requested > available:
        print(f"WARNING: Requested {total_requested} GPUs ({num_train} train + {num_infer} infer) "
              f"but only {available} available.")
        num_train = min(num_train, max(1, available // 2))
        num_infer = available - num_train
        print(f"  Adjusted to: {num_train} train + {num_infer} infer")

    train_ids = list(range(0, num_train))
    infer_ids = list(range(num_train, num_train + num_infer))

    return GPUAllocation(train_gpu_ids=train_ids, infer_gpu_ids=infer_ids)
