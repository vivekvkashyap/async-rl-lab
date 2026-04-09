#!/usr/bin/env python3
"""Standalone vLLM inference process.

Launched as a separate top-level process (not a subprocess of coordinator).
This avoids CUDA context and multiprocessing conflicts with vLLM's internal
process management.

Usage (called by run_experiment.py, not directly):
    CUDA_VISIBLE_DEVICES=1 python core/inference_process.py \
        --model Qwen/Qwen2.5-0.5B --dtype bfloat16 \
        --queue-name /rollout_queue --worker-id 0 ...
"""
import argparse
import asyncio
import glob as globmod
import os
import random
import sys
import time
import multiprocessing as mp
from multiprocessing import shared_memory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_inference(
    model_name: str,
    dtype: str,
    group_size: int,
    max_completion_length: int,
    temperature: float,
    top_p: float,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    train_data: list,
    queue: mp.Queue,
    version_val: mp.Value,
    stop_event: mp.Event,
    sync_dir: str,
    worker_id: int,
):
    """Main inference loop. Call this after CUDA_VISIBLE_DEVICES is set."""
    from core.vllm_inference_worker import VLLMInferenceWorker
    from scorers.verifier_scorer import VerifierScorer

    config = {
        "group_size": group_size,
        "max_completion_length": max_completion_length,
        "temperature": temperature,
        "top_p": top_p,
    }

    worker = VLLMInferenceWorker(
        model_name=model_name,
        config=config,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )
    scorer = VerifierScorer()

    local_version = 0
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    tag = f"Inference-{worker_id} GPU:{gpu_id}"
    print(f"[{tag}] vLLM ready. Generating rollouts...")

    while not stop_event.is_set():
        # Hot-swap weights if trainer has newer version
        current_trainer_version = version_val.value
        if current_trainer_version > local_version:
            try:
                dirs = sorted(globmod.glob(os.path.join(sync_dir, "v*")))
                if dirs:
                    latest_dir = dirs[-1]
                    v = int(os.path.basename(latest_dir).lstrip("v"))
                    if v > local_version and worker.update_weights(latest_dir):
                        local_version = v
                        print(f"[{tag}] Hot-swapped weights v{local_version}")
            except Exception:
                pass

        # Sample and generate
        prompts = random.sample(train_data, min(2, len(train_data)))
        rollouts = worker.generate_rollouts(prompts, local_version)

        # Score and push
        for r in rollouts:
            scored = asyncio.run(scorer.score(r))
            try:
                queue.put(scored, timeout=5)
            except Exception:
                if stop_event.is_set():
                    break

    print(f"[{tag}] Stopped.")
