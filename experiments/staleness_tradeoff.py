#!/usr/bin/env python3
"""Staleness tradeoff experiment.

Fix: bounded_queue(K=4), nccl_bucketed sync, soft_drain interrupt, verifier scorer.
Vary: no_filter, version_rejection(max_lag=1), version_rejection(max_lag=3),
      is_reweighting, hybrid(max_lag=3).
Plot: GSM8K accuracy over steps, throughput, drop rate, IS weight variance.
"""
import argparse
import asyncio
import copy
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_experiment import run_from_config
from utils.metrics import print_comparison_table
from utils.plotting import plot_training_curves, plot_throughput_comparison, plot_staleness_distribution


STALENESS_CONFIGS = {
    "no_filter": {"type": "no_filter"},
    "reject_lag1": {"type": "version_rejection", "max_lag": 1},
    "reject_lag3": {"type": "version_rejection", "max_lag": 3},
    "is_reweight": {"type": "is_reweighting", "is_clip_eps": 0.2},
    "hybrid_lag3": {"type": "hybrid", "max_lag": 3, "is_clip_eps": 0.2},
}


def make_config(base_config_path: str, staleness_name: str, staleness_cfg: dict, output_dir: str) -> dict:
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    config["staleness"] = staleness_cfg
    config["buffer"] = {"type": "bounded_queue", "maxsize": 4}
    config["weight_sync"]["type"] = "filesystem"  # fallback for single-process
    config["interrupt"] = {"type": "batch_sync"}
    config["scorer"] = {"type": "verifier"}
    config["metrics"]["plot_dir"] = os.path.join(output_dir, staleness_name)
    os.makedirs(config["metrics"]["plot_dir"], exist_ok=True)

    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output-dir", default="results/staleness_tradeoff")
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metrics_files = []
    labels = []

    for name, staleness_cfg in STALENESS_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Running experiment: staleness={name}")
        print(f"{'='*60}")

        config = make_config(args.config, name, staleness_cfg, args.output_dir)
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

    # Generate comparison table and plots
    if metrics_files:
        print_comparison_table(metrics_files, labels)
        print(f"Generating comparison plots...")
        plot_training_curves(metrics_files, labels,
                           key="reward_mean", title="Staleness Tradeoff: Reward",
                           ylabel="Mean Reward",
                           output_path=os.path.join(args.output_dir, "reward_comparison"))
        plot_throughput_comparison(metrics_files, labels,
                                 output_path=os.path.join(args.output_dir, "throughput"))
        print(f"Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
