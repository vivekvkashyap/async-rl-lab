import json
import time
import os
from typing import Dict, Any, List, Optional
from collections import defaultdict


class MetricsTracker:
    """Tracks per-step training metrics with optional W&B logging."""

    def __init__(self, log_every: int = 1, output_dir: str = "results/",
                 wandb_config: Optional[Dict[str, Any]] = None):
        self.log_every = log_every
        self.output_dir = output_dir
        self.history: List[Dict[str, Any]] = []
        self._start_time = time.time()
        self._wandb_run = None
        os.makedirs(output_dir, exist_ok=True)

        if wandb_config and wandb_config.get("enabled", False):
            self._init_wandb(wandb_config)

    def _init_wandb(self, wandb_config: Dict[str, Any]):
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=wandb_config.get("project", "async-rl-lab"),
                name=wandb_config.get("run_name"),
                group=wandb_config.get("group"),
                tags=wandb_config.get("tags", []),
                config=wandb_config.get("config", {}),
                reinit="finish_previous",
            )
            print(f"[wandb] Logging to {self._wandb_run.url}")
        except ImportError:
            print("[wandb] wandb not installed, skipping. Install with: pip install wandb")
        except Exception as e:
            print(f"[wandb] Init failed: {e}")

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        metrics["step"] = step
        metrics["wall_clock"] = time.time() - self._start_time
        self.history.append(metrics)
        if step % self.log_every == 0:
            self._print(step, metrics)
        if self._wandb_run is not None:
            self._log_wandb(step, metrics)

    def _print(self, step: int, metrics: Dict[str, Any]) -> None:
        parts = [f"step={step}"]
        for key in ["training_loss", "reward_mean", "reward_max",
                     "gsm8k_accuracy",
                     "tokens_per_second", "generation_time", "train_time",
                     "sync_duration", "wall_clock_time",
                     "batch_staleness_mean", "buffer_depth",
                     "ipo_masked_frac", "mismatch_kl",
                     "num_dropped", "gpu_util_train", "gpu_util_infer"]:
            if key in metrics:
                val = metrics[key]
                parts.append(f"{key}={val:.4f}" if isinstance(val, float) else f"{key}={val}")
        print(" | ".join(parts))

    def summary(self) -> Dict[str, Any]:
        if not self.history:
            return {}
        result = {}
        numeric_keys = set()
        for h in self.history:
            for k, v in h.items():
                if isinstance(v, (int, float)):
                    numeric_keys.add(k)
        for k in numeric_keys:
            vals = [h[k] for h in self.history if k in h]
            if vals:
                result[f"{k}_mean"] = sum(vals) / len(vals)
                result[f"{k}_last"] = vals[-1]
        return result

    def _log_wandb(self, step: int, metrics: Dict[str, Any]):
        import wandb
        # Group metrics into sections for cleaner wandb dashboard
        wb_metrics = {}
        for k, v in metrics.items():
            if not isinstance(v, (int, float)):
                continue
            if k in ("training_loss", "policy_loss", "kl_loss", "grad_norm",
                     "clip_fraction", "ipo_masked_frac", "mismatch_kl",
                     "ipo_masked_high", "ipo_masked_low"):
                wb_metrics[f"train/{k}"] = v
            elif k in ("reward_mean", "reward_max", "gsm8k_accuracy"):
                wb_metrics[f"reward/{k}"] = v
            elif k in ("tokens_per_second", "generation_time", "train_time",
                        "sync_duration", "wall_clock_time"):
                wb_metrics[f"perf/{k}"] = v
            elif k in ("gpu_util_train", "gpu_util_infer", "gpu_mem_train"):
                wb_metrics[f"gpu/{k}"] = v
            elif k in ("batch_staleness_mean", "buffer_depth", "num_dropped"):
                wb_metrics[f"async/{k}"] = v
            else:
                wb_metrics[k] = v
        wandb.log(wb_metrics, step=step)

    def save(self, filename: str = "metrics.json") -> str:
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        return path

    def finish(self):
        """Close wandb run if active."""
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None


def print_comparison_table(metrics_files: List[str], labels: List[str]):
    """Print a comparison table across experiment runs for blog/reporting.

    Shows key metrics side by side for each configuration.
    """
    columns = [
        ("reward_mean", "Reward (last)", "last", ".4f"),
        ("reward_mean", "Reward (mean)", "mean", ".4f"),
        ("training_loss", "Loss (last)", "last", ".4f"),
        ("tokens_per_second", "Throughput (tok/s)", "mean", ".1f"),
        ("wall_clock_time", "Step Time (s)", "mean", ".2f"),
        ("sync_duration", "Sync Time (s)", "mean", ".3f"),
        ("batch_staleness_mean", "Staleness", "mean", ".2f"),
        ("gsm8k_accuracy", "GSM8K Acc", "last", ".4f"),
        ("gpu_util_train", "GPU% Train", "mean", ".1%"),
        ("gpu_util_infer", "GPU% Infer", "mean", ".1%"),
    ]

    # Load all metrics
    summaries = []
    for path in metrics_files:
        with open(path) as f:
            history = json.load(f)
        summary = {}
        for h in history:
            for k, v in h.items():
                if isinstance(v, (int, float)):
                    summary.setdefault(k, []).append(v)
        summaries.append(summary)

    # Build table
    header = f"{'Config':<20}"
    for _, col_name, _, _ in columns:
        header += f" | {col_name:>14}"
    print("\n" + "=" * len(header))
    print("EXPERIMENT COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for label, summary in zip(labels, summaries):
        row = f"{label:<20}"
        for key, _, agg, fmt in columns:
            vals = summary.get(key, [])
            if not vals:
                row += f" | {'N/A':>14}"
            elif agg == "last":
                row += f" | {format(vals[-1], fmt):>14}"
            else:
                row += f" | {format(sum(vals) / len(vals), fmt):>14}"
        print(row)

    print("=" * len(header) + "\n")
