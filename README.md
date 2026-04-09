# async-rl-lab

Modular async RL training framework for experimenting with different design choices in asynchronous reinforcement learning. Built for controlled comparison of buffers, weight sync, staleness handling, interrupts, and scoring strategies.

Based on the [HuggingFace Async RL Training Landscape](https://huggingface.co/blog/async-rl-training-landscape) blog and [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl).

## Architecture

```
GPU 0 (trainer)          GPU 1+ (inference)
+-----------------+      +------------------+
| GRPOTrainer     |      | vLLM Engine      |
| (IPO/GRPO loss) |<---->| (PagedAttention)  |
| Coordinator     | queue| Hot-swap weights  |
| Weight Sync     |      +------------------+
+-----------------+
```

- **Inference**: vLLM with continuous batching, one process per GPU
- **Training**: IPO (INTELLECT Policy Optimization) or GRPO, with optional FSDP for multi-GPU
- **Communication**: `multiprocessing.Queue` for rollouts, filesystem/NCCL for weights
- Inference processes start **before** training model loads to avoid CUDA context conflicts

## Quick Start

```bash
# Install dependencies
uv sync

# Run with default config (1 train GPU + 1 infer GPU, IPO algorithm)
python experiments/run_experiment.py --config configs/experiment.yaml

# Quick test (3 steps)
python experiments/run_experiment.py --max-steps 3

# Multi-GPU FSDP training (e.g., 4 GPUs: 2 train + 2 infer)
python experiments/run_experiment.py --num-train-gpus 2 --num-infer-gpus 2
```

## Training Algorithms

### IPO (default) - INTELLECT Policy Optimization
From [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl). More stable for async RL:
- **Probability-difference masking** instead of ratio clipping (trust region via TV distance)
- **Quadratic KL regularization**: `kl_tau * log(ratio)^2`
- **No std normalization** of advantages (just `reward - baseline`)

### GRPO - Group Relative Policy Optimization
Standard clipped surrogate loss (PPO-style):
- Ratio clipping: `clip(ratio, 1-eps, 1+eps)`
- Approximate KL penalty: `ratio - log(ratio) - 1`
- Std-normalized advantages

Set `algorithm: "ipo"` or `algorithm: "grpo"` in config.

## Swappable Components

| Axis | Options | Config key |
|------|---------|------------|
| **Buffer** | `sync`, `bounded_queue`, `double`, `redis_stream` | `buffer.type` |
| **Weight Sync** | `filesystem`, `nccl_broadcast`, `nccl_bucketed`, `zmq_notify_fs` | `weight_sync.type` |
| **Staleness** | `no_filter`, `version_rejection`, `is_reweighting`, `hybrid` | `staleness.type` |
| **Interrupt** | `batch_sync`, `soft_drain`, `implicit_continuation` | `interrupt.type` |
| **Scorer** | `verifier`, `distillation` | `scorer.type` |

## Config Reference

```yaml
model:
  name: "Qwen/Qwen2.5-0.5B"
  dtype: "bf16"
  inference_dtype: "bf16"

deployment:
  num_train_gpus: 1        # >1 enables FSDP
  num_infer_gpus: 1        # one vLLM process per GPU

training:
  algorithm: "ipo"          # "ipo" or "grpo"
  group_size: 8
  batch_size: 16
  lr: 1.0e-6
  max_steps: 200
  # IPO params
  ipo_mask_low: 0.2
  ipo_mask_high: 0.2
  adv_tau: 1.0
  kl_tau: 1.0e-3
  # GRPO params
  clip_eps: 0.2
  kl_coeff: 0.01

buffer:
  type: "sync"

weight_sync:
  type: "filesystem"
  checkpoint_dir: "/tmp/async-rl-lab/checkpoints"

staleness:
  type: "no_filter"

metrics:
  log_every: 1
  eval_every: 10           # GSM8K eval every N steps (0 to disable)
  eval_samples: 200
```

## Experiments

Pre-built experiment scripts for comparing design choices:

```bash
# Compare buffer strategies
python experiments/buffer_comparison.py --max-steps 50

# Staleness handling tradeoffs
python experiments/staleness_tradeoff.py --max-steps 50

# Dtype mismatch effects (train vs inference precision)
python experiments/dtype_mismatch.py --max-steps 50

# Sampling mask effect (DeepSeek-V3.2 Keep Sampling Mask)
python experiments/sampling_mask.py --max-steps 50

# Distillation vs verifier scoring
python experiments/distillation_vs_grpo.py --max-steps 50
```

Results and plots are saved to `results/`.

## Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
async-rl-lab/
├── core/
│   ├── coordinator.py      # Async training loop orchestration
│   ├── trainer.py           # IPO/GRPO loss computation
│   ├── launcher.py          # Process spawning (inference + training)
│   ├── inference_process.py # vLLM inference worker loop
│   ├── vllm_inference_worker.py  # vLLM engine wrapper
│   └── types.py             # Rollout, ScoredRollout, TrainingBatch
├── buffers/                 # Rollout buffer implementations
├── weight_sync/             # Weight synchronization strategies
├── staleness/               # Off-policy staleness handling
├── interrupts/              # Generation interrupt strategies
├── scorers/                 # Reward/scoring functions
├── utils/
│   ├── factory.py           # Component creation from config
│   ├── fsdp.py              # FSDP distributed training utilities
│   ├── gpu_allocator.py     # GPU assignment logic
│   ├── gsm8k.py             # Dataset loading and answer extraction
│   ├── metrics.py           # Training metrics tracker
│   └── plotting.py          # Publication-quality plots
├── experiments/             # Comparison experiment scripts
├── configs/experiment.yaml  # Default config
└── tests/
```

## Requirements

- 2+ NVIDIA GPUs (1 train + 1 infer minimum)
- Python 3.10+
- PyTorch with CUDA
- vLLM
- transformers, safetensors, datasets
