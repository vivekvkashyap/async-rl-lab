#!/usr/bin/env python3
"""Dtype mismatch experiment.

Fix: bounded_queue(K=2), filesystem sync, hybrid staleness, batch_sync interrupt.
Vary: training_dtype x inference_dtype combinations:
  (bf16, bf16), (float32, float32), (float32, bf16), (bf16, float32).
Plot: reward curves over training steps.
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_experiment import run_from_config
from utils.metrics import print_comparison_table
from utils.plotting import plot_reward_curves


DTYPE_COMBOS = {
    "bf16_bf16": ("bf16", "bf16"),
    "fp32_fp32": ("float32", "float32"),
    "fp32_bf16": ("float32", "bf16"),
    "bf16_fp32": ("bf16", "float32"),
}


def make_config(base_config_path: str, name: str, train_dtype: str,
                infer_dtype: str, output_dir: str) -> dict:
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    config["model"]["dtype"] = train_dtype
    config["model"]["inference_dtype"] = infer_dtype
    config["buffer"] = {"type": "bounded_queue", "maxsize": 2}
    config["staleness"] = {"type": "hybrid", "max_lag": 3, "is_clip_eps": 0.2}
    config["weight_sync"]["type"] = "filesystem"
    config["interrupt"] = {"type": "batch_sync"}
    config["scorer"] = {"type": "verifier"}
    config["metrics"]["plot_dir"] = os.path.join(output_dir, name)
    os.makedirs(config["metrics"]["plot_dir"], exist_ok=True)

    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output-dir", default="results/dtype_mismatch")
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_files = []
    labels = []

    for name, (train_dtype, infer_dtype) in DTYPE_COMBOS.items():
        print(f"\n{'='*60}")
        print(f"Running experiment: train={train_dtype}, infer={infer_dtype}")
        print(f"{'='*60}")

        config = make_config(args.config, name, train_dtype, infer_dtype, args.output_dir)
        config["training"]["max_steps"] = args.max_steps

        try:
            run_from_config(config)
        except Exception as e:
            print(f"Error running {name}: {e}")
            continue

        metrics_path = os.path.join(config["metrics"]["plot_dir"], "metrics.json")
        if os.path.exists(metrics_path):
            metrics_files.append(metrics_path)
            labels.append(f"train={train_dtype}\ninfer={infer_dtype}")

    if metrics_files:
        print_comparison_table(metrics_files, labels)
        plot_reward_curves(metrics_files, labels,
                          output_path=os.path.join(args.output_dir, "dtype_reward_comparison"))
        print(f"Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
