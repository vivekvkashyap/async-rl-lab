# Experiments guide

How to run each axis and what to record. Run from the repo root with the
project venv: `.venv/bin/python`.

All multi-process runs need `mp.set_start_method("spawn")`, which the
entrypoints already set. Requires at least 2 GPUs (1 train + 1 infer).

---

## 0. Sanity check (do this first)

Verify the full pipeline is healthy on your hardware before sweeping.

```bash
.venv/bin/python experiments/run_experiment.py --max-steps 3
```

**What to check before moving on:**
- The run finishes without exceptions.
- Per-step print line contains `training_loss`, `reward_mean`, `tokens_per_second`.
- `results/metrics.json` exists and has 3 entries.
- At least one `Hot-swapped weights v1` line in stdout (proves the weight
  sync round-trip works).

If any of those fail, fix before running the sweeps below.

---

## 1. Buffer axis

**Question:** which buffer pattern gives the best overlap between training
and inference without letting rollouts go stale?

```bash
.venv/bin/python experiments/buffer_comparison.py --max-steps 50
```

Sweeps: `sync`, `double`, `bounded_queue(K=2)`, `bounded_queue(K=4)`, `bounded_queue(K=8)`.
Fixes staleness=hybrid, sync=filesystem, interrupt=batch_sync, scorer=verifier.

**Record for each variant:**
| Metric | Aggregation | Why |
|---|---|---|
| `tokens_per_second` | mean (steps 5..end) | throughput — primary |
| `wall_clock_time` | mean | end-to-end step time |
| `gpu_util_train` | mean | are we keeping the trainer fed? |
| `gpu_util_infer` | mean | is inference idling on backpressure? |
| `buffer_depth` | mean | sitting at maxsize → training-bound |
| `batch_staleness_mean` | mean | cost of higher K |
| `reward_mean` | last | quality didn't regress |

**Reading it:** higher K → higher throughput but higher staleness. The
"best" buffer is the one where throughput plateaus just before staleness
starts hurting reward. `double` should beat `sync` on throughput; if it
doesn't, inference is the bottleneck, not collection.

Outputs: `results/buffer_comparison/{reward_comparison,step_time,throughput}.png`
plus per-variant `staleness_<name>.png`.

---

## 2. Staleness axis

**Question:** what's the cost/benefit of filtering vs reweighting stale
rollouts?

```bash
.venv/bin/python experiments/staleness_tradeoff.py --max-steps 50
```

Sweeps: `no_filter`, `version_rejection(max_lag=3)`, `is_reweighting`, `hybrid`.

**Record for each variant:**
| Metric | Aggregation | Why |
|---|---|---|
| `gsm8k_accuracy` | last | primary — quality of learning |
| `reward_mean` | last and full curve | learning trajectory |
| `rollouts_dropped_by_staleness` | sum | hard cost of version_rejection/hybrid |
| `batch_staleness_mean` | mean | post-filter staleness |
| `batch_staleness_precollect_mean` | mean | pre-filter staleness (what was rejected) |
| `training_loss` | std over last 20 steps | stability |
| `ipo_masked_frac` | mean | IPO trust-region violations |
| `mismatch_kl` | mean | train↔inference policy divergence |

**Reading it:** `no_filter` should be fastest but noisiest. `version_rejection`
drops data, losing throughput but stabilizing loss. `is_reweighting` keeps
all data but correction variance can blow up — watch `mismatch_kl`. `hybrid`
should be the Pareto winner on (accuracy, throughput). Compare
`batch_staleness_mean` to `batch_staleness_precollect_mean` to see exactly
what each filter is rejecting.

---

## 3. Weight sync axis

**Question:** how much does each sync mechanism cost, and does it
bottleneck the training loop?

**Raw latency benchmark (no training):**
```bash
.venv/bin/python experiments/weight_sync_benchmark.py --n-iters 100
```

Only `filesystem` has a working benchmark in this repo. `zmq_notify_fs` is
functionally equivalent (filesystem + notification) for the training-side
cost. NCCL variants are documented-out of the main pipeline and the
benchmark script only stubs them. **Record:** `mean`, `p50`, `p99` from
`results/sync_benchmark.json`.

**In-pipeline cost (re-run default with each sync type):**
```bash
# Edit configs/experiment.yaml → weight_sync.type, then:
.venv/bin/python experiments/run_experiment.py --max-steps 50
```

Do this once with `type: filesystem` and once with `type: zmq_notify_fs`.

**Record for each variant:**
| Metric | Aggregation | Why |
|---|---|---|
| `sync_duration` | mean, p99 | primary — bubble cost |
| `wall_clock_time` | mean | overall step time |
| `tokens_per_second` | mean | end-to-end throughput impact |

**Reading it:** `zmq_notify_fs` should beat `filesystem` on the wall-clock
gap between trainer pushing weights and inference picking them up (because
inference reacts to the notification instead of polling `version_val`). The
push side is identical — both write safetensors. If `sync_duration` p99 is
more than 2x the mean, you have a filesystem stall (slow disk, contention)
and should switch to a faster `checkpoint_dir`.

---

## 4. Interrupt axis

**Question:** how should generation be paused (or not) during weight sync?

There's no dedicated sweep script, so re-run the default with each
`interrupt.type`:

```bash
# Edit configs/experiment.yaml → interrupt.type, then:
.venv/bin/python experiments/run_experiment.py --max-steps 50
```

Do this three times: `batch_sync`, `soft_drain`, `implicit_continuation`.
Save each run's `results/metrics.json` under a different folder (set
`metrics.plot_dir` in the yaml) so you can compare afterwards.

**Record for each variant:**
| Metric | Aggregation | Why |
|---|---|---|
| `sync_duration` | mean | time sync side takes |
| `generation_time` | mean | includes any drain wait |
| `tokens_per_second` | mean | net impact |
| `gpu_util_infer` | mean | is inference idling during drain? |
| `reward_mean` | last | quality impact of mid-generation weight swaps |

**Reading it:** `batch_sync` gives the cleanest data but pauses inference
(lower `gpu_util_infer`). `soft_drain` reduces the bubble by not rejecting
in-flight requests. `implicit_continuation` should have the lowest
`sync_duration` and highest `gpu_util_infer` but may show slightly worse
reward if mid-generation weight swaps hurt coherence.

---

## 5. Scorer axis

**Question:** rule-based verifier vs distillation from a teacher — which
reward signal learns faster?

```bash
.venv/bin/python experiments/distillation_vs_grpo.py --max-steps 50
```

**Record for each variant:**
| Metric | Aggregation | Why |
|---|---|---|
| `gsm8k_accuracy` | last | primary — did it learn the task? |
| `reward_mean` | last and full curve | learning trajectory |
| `generation_time` | mean | distillation adds a teacher forward pass |
| `tokens_per_second` | mean | throughput penalty for the teacher |

**Reading it:** verifier reward is sparse (0/1) but on-target for GSM8K.
Distillation gives dense teacher logprobs but the reward is "agree with
teacher", not "get the right answer". Expect verifier to win on
`gsm8k_accuracy` for GSM8K specifically; distillation to have smoother
`reward_mean` curve but slower throughput.

---

## 6. Algorithm axis (IPO vs GRPO)

No dedicated sweep — re-run default with each algorithm:

```bash
# Edit configs/experiment.yaml → training.algorithm, then:
.venv/bin/python experiments/run_experiment.py --max-steps 100
```

Do this twice: `algorithm: "ipo"` and `algorithm: "grpo"`.

**Record for each variant:**
| Metric | Aggregation | Why |
|---|---|---|
| `gsm8k_accuracy` | last | primary |
| `reward_mean` | last and full curve | learning trajectory |
| `training_loss` | full curve | stability |
| `ipo_masked_frac` (IPO) | mean | trust region violations — policy drift |
| `clip_fraction` (GRPO) | mean | ratio clipping saturation — policy drift |
| `kl_loss` | mean | divergence from generation policy |

**Reading it:** IPO is supposed to be more stable under off-policy data.
Look at `training_loss` variance — IPO should be smoother. If
`ipo_masked_frac` climbs above ~0.3, the trust region is being hit too
often (lower the learning rate or tighten staleness). If GRPO's
`clip_fraction` is above ~0.2 consistently, PPO-style clipping is
saturating and the effective learning rate is near zero.

---

## 7. Dtype mismatch axis

**Question:** does using a lower-precision dtype on inference vs training
hurt off-policy correction?

```bash
.venv/bin/python experiments/dtype_mismatch.py --max-steps 50
```

**Record for each variant:**
| Metric | Aggregation | Why |
|---|---|---|
| `mismatch_kl` | mean | primary — directly measures train↔infer divergence |
| `reward_mean` | last and full curve | downstream effect |
| `gsm8k_accuracy` | last | final quality |
| `ipo_masked_frac` | mean | how often trust region is tripped |

**Reading it:** `mismatch_kl` is the purest measurement of the dtype
effect — it's the KL between what inference thought the logprobs were and
what the trainer recomputes. bf16/bf16 should be lowest. fp16/bf16 higher.
If `mismatch_kl > 0.05` consistently, your IS correction is doing real
work and the dtype mismatch matters.

---

## Comparison protocol

For every axis:

1. **Fix everything except the axis under test.** The sweep scripts
   (`buffer_comparison.py`, `staleness_tradeoff.py`, etc.) already do
   this. For axes without a sweep script (weight_sync, interrupt,
   algorithm), edit `configs/experiment.yaml` and run
   `run_experiment.py` once per variant, saving each run to a distinct
   `metrics.plot_dir`.
2. Use the same `--max-steps`, same `seed`, same model across all variants
   in a sweep.
3. For **throughput / timing metrics** (`tokens_per_second`,
   `wall_clock_time`, `sync_duration`, `generation_time`): report the
   **mean over steps 5..end**, skipping warmup.
4. For **quality metrics** (`reward_mean`, `gsm8k_accuracy`): report the
   **last** value. For loss-trajectory comparisons, plot the full curve.
5. Record the **3-tuple per variant**: (primary metric, secondary cost
   metric, `gsm8k_accuracy_last`). That's the minimum to tell a story.
6. Look at `print_comparison_table` output at the end of each sweep script
   — it already formats most of these.

## Healthy-run pattern

Watch the per-step print line for this shape:

```
step=5 | training_loss=0.12 | reward_mean=0.35 | tokens_per_second=420.1 |
  generation_time=3.2 | train_time=0.8 | sync_duration=0.15 |
  batch_staleness_mean=1.20 | buffer_depth=8 |
  gpu_util_train=0.42 | gpu_util_infer=0.88
```

- `generation_time ≈ train_time + sync_duration` → good overlap
- `gpu_util_train > 0.6` after warmup → trainer is fed
- `buffer_depth` stable (not at 0, not at max) → balanced pipeline
- `reward_mean` trending up → the thing is actually learning

**Failure modes:**
- `gpu_util_train` near 0 and `buffer_depth` near 0 → inference-bound
- `buffer_depth` pinned at maxsize → training-bound (raise K or use `double`)
- `batch_staleness_mean` climbing without plateau → sync isn't keeping up
- `sync_duration` p99 >> mean → disk contention, switch checkpoint_dir
