#!/usr/bin/env python3
"""Sampling mask experiment (Keep Sampling Mask from DeepSeek-V3.2).

Fix: bounded_queue(K=2), filesystem sync, hybrid staleness, batch_sync.
Run generation with top_p=0.9.
Compare: (a) standard training, (b) training with sampling mask applied.
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


MASK_CONFIGS = {
    "no_mask": {"record_sampling_mask": False},
    "with_mask": {"record_sampling_mask": True},
}


def make_config(base_config_path: str, name: str, mask_cfg: dict, output_dir: str) -> dict:
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    config["buffer"] = {"type": "bounded_queue", "maxsize": 2}
    config["staleness"] = {"type": "hybrid", "max_lag": 3, "is_clip_eps": 0.2}
    config["weight_sync"]["type"] = "filesystem"
    config["interrupt"] = {"type": "batch_sync"}
    config["scorer"] = {"type": "verifier"}
    config["training"]["top_p"] = 0.9
    config["training"]["record_sampling_mask"] = mask_cfg["record_sampling_mask"]
    config["metrics"]["plot_dir"] = os.path.join(output_dir, name)
    os.makedirs(config["metrics"]["plot_dir"], exist_ok=True)

    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output-dir", default="results/sampling_mask")
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_files = []
    labels = []

    for name, mask_cfg in MASK_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Running experiment: {name}")
        print(f"{'='*60}")

        config = make_config(args.config, name, mask_cfg, args.output_dir)
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
                           key="reward_mean", title="Sampling Mask Effect on Convergence",
                           ylabel="Mean Reward",
                           output_path=os.path.join(args.output_dir, "mask_comparison"))
        print(f"Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
