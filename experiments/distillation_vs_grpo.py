#!/usr/bin/env python3
"""Distillation vs GRPO comparison.

Fix: bounded_queue(K=4), filesystem sync, hybrid staleness, batch_sync.
Compare: verifier_scorer (GRPO) vs distillation_scorer (teacher=Qwen2.5-1.5B).
Plot: GSM8K accuracy curves for both.
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_experiment import run_from_config
from utils.metrics import print_comparison_table
from utils.plotting import plot_training_curves


SCORER_CONFIGS = {
    "grpo_verifier": {
        "type": "verifier",
    },
    "distillation": {
        "type": "distillation",
        "teacher_model": "Qwen/Qwen2.5-1.5B",
        "teacher_snapshot_every": 10,
    },
}


def make_config(base_config_path: str, name: str, scorer_cfg: dict, output_dir: str) -> dict:
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    config["scorer"] = scorer_cfg
    config["buffer"] = {"type": "bounded_queue", "maxsize": 4}
    config["staleness"] = {"type": "hybrid", "max_lag": 3, "is_clip_eps": 0.2}
    config["weight_sync"]["type"] = "filesystem"
    config["interrupt"] = {"type": "batch_sync"}
    config["metrics"]["plot_dir"] = os.path.join(output_dir, name)
    os.makedirs(config["metrics"]["plot_dir"], exist_ok=True)

    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output-dir", default="results/distillation_vs_grpo")
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_files = []
    labels = []

    for name, scorer_cfg in SCORER_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Running experiment: scorer={name}")
        print(f"{'='*60}")

        config = make_config(args.config, name, scorer_cfg, args.output_dir)
        config["training"]["max_steps"] = args.max_steps

        try:
            run_from_config(config)
        except Exception as e:
            print(f"Error running {name}: {e}")
            continue

        metrics_path = os.path.join(config["metrics"]["plot_dir"], "metrics.json")
        if os.path.exists(metrics_path):
            metrics_files.append(metrics_path)
            labels.append(name.replace("_", " ").title())

    if metrics_files:
        print_comparison_table(metrics_files, labels)
        plot_training_curves(metrics_files, labels,
                           key="reward_mean", title="Distillation vs GRPO",
                           ylabel="Mean Reward",
                           output_path=os.path.join(args.output_dir, "comparison"))
        print(f"Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
