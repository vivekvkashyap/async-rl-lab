#!/usr/bin/env python3
"""Run an async GRPO experiment.

Usage:
    python experiments/run_experiment.py --config configs/experiment.yaml
    python experiments/run_experiment.py --config configs/experiment.yaml --max-steps 10
    python experiments/run_experiment.py --num-train-gpus 2 --num-infer-gpus 2
"""
import argparse
import multiprocessing as mp
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_from_config(config: dict):
    """Launch an experiment from a config dict. Used by experiment scripts."""
    deployment = config.get("deployment", {})
    if deployment.get("num_infer_gpus", 1) < 1:
        config.setdefault("deployment", {})["num_infer_gpus"] = 1

    from core.launcher import launch
    launch(config)


def main():
    parser = argparse.ArgumentParser(description="Run async GRPO experiment")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-train-gpus", type=int, default=None)
    parser.add_argument("--num-infer-gpus", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.num_train_gpus is not None:
        config.setdefault("deployment", {})["num_train_gpus"] = args.num_train_gpus
    if args.num_infer_gpus is not None:
        config.setdefault("deployment", {})["num_infer_gpus"] = args.num_infer_gpus

    run_from_config(config)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
