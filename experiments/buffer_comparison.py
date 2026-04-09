#!/usr/bin/env python3
"""Buffer comparison experiment.

Fix: filesystem sync (or nccl_bucketed), hybrid staleness (max_lag=3),
     batch_sync interrupt, verifier scorer.
Vary: sync_buffer, double_buffer, bounded_queue(K=2), bounded_queue(K=4),
      bounded_queue(K=8).
Plot: wall clock time per step, GPU utilization, staleness distribution,
      GSM8K accuracy curve.
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_experiment import run_from_config
from utils.metrics import print_comparison_table
from utils.plotting import (
    plot_training_curves, plot_throughput_comparison, plot_staleness_distribution,
)


BUFFER_CONFIGS = {
    "sync": {"type": "sync"},
    "double": {"type": "double"},
    "queue_K2": {"type": "bounded_queue", "maxsize": 2},
    "queue_K4": {"type": "bounded_queue", "maxsize": 4},
    "queue_K8": {"type": "bounded_queue", "maxsize": 8},
}


def make_config(base_config_path: str, buffer_name: str, buffer_cfg: dict, output_dir: str) -> dict:
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    config["buffer"] = buffer_cfg
    config["staleness"] = {"type": "hybrid", "max_lag": 3, "is_clip_eps": 0.2}
    config["weight_sync"]["type"] = "filesystem"
    config["interrupt"] = {"type": "batch_sync"}
    config["scorer"] = {"type": "verifier"}
    config["metrics"]["plot_dir"] = os.path.join(output_dir, buffer_name)
    os.makedirs(config["metrics"]["plot_dir"], exist_ok=True)

    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output-dir", default="results/buffer_comparison")
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_files = []
    labels = []

    for name, buffer_cfg in BUFFER_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Running experiment: buffer={name}")
        print(f"{'='*60}")

        config = make_config(args.config, name, buffer_cfg, args.output_dir)
        config["training"]["max_steps"] = args.max_steps

        try:
            run_from_config(config)
        except Exception as e:
            print(f"Error running {name}: {e}")
            continue

        metrics_path = os.path.join(config["metrics"]["plot_dir"], "metrics.json")
        if os.path.exists(metrics_path):
            metrics_files.append(metrics_path)
            labels.append(name)

    if metrics_files:
        print_comparison_table(metrics_files, labels)
        print(f"Generating comparison plots...")
        plot_training_curves(metrics_files, labels,
                           key="reward_mean", title="Buffer Comparison: Reward",
                           ylabel="Mean Reward",
                           output_path=os.path.join(args.output_dir, "reward_comparison"))
        plot_training_curves(metrics_files, labels,
                           key="wall_clock_time", title="Buffer Comparison: Step Time",
                           ylabel="Seconds per Step",
                           output_path=os.path.join(args.output_dir, "step_time"))
        plot_throughput_comparison(metrics_files, labels,
                                 output_path=os.path.join(args.output_dir, "throughput"))
        for mf, label in zip(metrics_files, labels):
            plot_staleness_distribution(mf, os.path.join(args.output_dir, f"staleness_{label}"))
        print(f"Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
