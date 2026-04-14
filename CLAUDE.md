# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install deps (managed by uv, lockfile committed)
uv sync

# Default run (1 train GPU + 1 infer GPU, IPO algorithm, GSM8K)
python experiments/run_experiment.py --config configs/experiment.yaml

# Smoke test
python experiments/run_experiment.py --max-steps 3

# Multi-GPU FSDP (e.g. 2 train + 2 infer)
python experiments/run_experiment.py --num-train-gpus 2 --num-infer-gpus 2

# Tests
python -m pytest tests/ -v
python -m pytest tests/test_coordinator.py::test_name -v   # single test
```

Requires 2+ NVIDIA GPUs. `mp.set_start_method("spawn")` is set in the entrypoint — required because vLLM + CUDA cannot share fork()'d state.

## Architecture

This is a research framework for comparing **swappable async-RL components** (buffer / weight-sync / staleness / interrupt / scorer strategies). The scaffolding is fixed; experiments vary one axis at a time via config.

### Process topology
Trainer and inference always run in **separate processes**, communicating via `mp.Queue` (rollouts), `mp.Value` (current weight version), and `mp.Event` (shutdown). This is not optional — it's the point of the lab.

- `core/launcher.py::launch` is the single entry point. Order matters: **inference workers are spawned before the training model loads**, otherwise CUDA context conflicts. Each inference worker sets `CUDA_VISIBLE_DEVICES` and `os.setpgrp()` so the launcher can `killpg` the entire vLLM subtree (vLLM spawns its own EngineCore child) on cleanup.
- Single-GPU training runs directly; multi-GPU uses `mp.spawn` → FSDP, where **rank 0 runs the Coordinator** and other ranks run `run_fsdp_follower` which listens for broadcast signals (`-1` = stop, `0` = skip step, `>0` = train).
- `core/coordinator.py::Coordinator._training_loop` is the orchestration heart: collect rollouts from queue → buffer put/get → staleness filter → `trainer.train_step` → `interrupt.prepare_for_sync` → `syncer.push` → `interrupt.resume_after_sync` → bump `version_val` (inference reads this to swap weights). Any change to the async dataflow happens here.

### Swappable axes (factory pattern)
`utils/factory.py` is the **only** place components are instantiated from config. When adding a new variant, register it there; the rest of the code only sees the base-class interface.

| Axis | Base | Implementations |
|---|---|---|
| `buffer.type` | `buffers/base.py` | `sync`, `bounded_queue`, `double`, `redis_stream` |
| `weight_sync.type` | `weight_sync/base.py` | `filesystem`, `nccl_broadcast`, `nccl_bucketed`, `zmq_notify_fs` |
| `staleness.type` | `staleness/base.py` | `no_filter`, `version_rejection`, `is_reweighting`, `hybrid` (IS reweighting needs the training model+tokenizer to recompute logprobs) |
| `interrupt.type` | `interrupts/base.py` | `batch_sync`, `soft_drain`, `implicit_continuation` |
| `scorer.type` | `scorers/base.py` | `verifier`, `distillation` (uses `TeacherManager` snapshots) |

### Training algorithms
`core/trainer.py::GRPOTrainer` implements **both** IPO and GRPO under one class, switched by `training.algorithm`:
- **IPO** (default, from prime-rl): probability-difference masking instead of ratio clipping, quadratic KL `kl_tau * log(ratio)^2`, no std-normalization of advantages. More stable under off-policy data.
- **GRPO**: PPO-style ratio clipping + approximate KL, std-normalized advantages.

When editing loss code, keep both paths working — experiments compare them directly.

### Evaluation caveat
`Coordinator.evaluate` calls `model.generate()` on the training model, which **does not work under FSDP** (needs unsharded params). The loop skips eval when `trainer.is_fsdp`. Don't "fix" this by unsharding — the design intentionally keeps the train path hot.

### Data flow invariants
- `ScoredRollout` carries `model_version` (set by inference at generation time). Staleness logic compares against `trainer.version` — preserve this field through any buffer/filter changes.
- `version_val.value` is the **only** signal inference uses to decide when to hot-swap weights. Update it *after* `syncer.push` completes, not before.
- `rollout_queue` has bounded capacity (`64 * num_infer_gpus`). Back-pressure on inference is intentional, not a bug.

## Config
Single source of truth is `configs/experiment.yaml`. Experiment scripts in `experiments/` build config dicts programmatically and call `run_from_config` — they do not take YAML files. Results + plots go to `results/`, wandb logging is opt-in via `wandb.enabled`.
