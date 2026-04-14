#!/usr/bin/env python3
"""Standalone vLLM inference process.

Launched as a separate top-level process (not a subprocess of coordinator).
This avoids CUDA context and multiprocessing conflicts with vLLM's internal
process management.

The worker pulls new weights from the configured weight_sync path (currently
filesystem — zmq_notify_fs also writes to disk via the same sync_dir), runs
generation, scores rollouts with the configured scorer, then pushes to the
shared rollout queue (or redis stream) supplied by the launcher.
"""
import asyncio
import glob as globmod
import os
import random
import sys
import time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _parse_version_dir(path: str) -> int:
    """Parse 'v000005' → 5. Returns -1 if not a valid version dir."""
    name = os.path.basename(path)
    if not name.startswith("v"):
        return -1
    try:
        return int(name[1:])
    except ValueError:
        return -1


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
    scorer_config: dict,
    interrupt=None,
    redis_url: str = None,
    redis_stream_key: str = "rollouts",
):
    """Main inference loop. Call this after CUDA_VISIBLE_DEVICES is set."""
    from core.vllm_inference_worker import VLLMInferenceWorker
    from utils.factory import create_scorer

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
    scorer = create_scorer({"scorer": scorer_config})

    # Redis path (when the buffer is RedisStreamBuffer, queue is None and
    # we push to a redis stream instead).
    redis_client = None
    if queue is None and redis_url is not None:
        import redis
        from buffers.redis_stream import _rollout_to_dict
        redis_client = redis.from_url(redis_url, decode_responses=True)
    else:
        _rollout_to_dict = None  # noqa

    local_version = 0
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    tag = f"Inference-{worker_id} GPU:{gpu_id}"
    print(f"[{tag}] vLLM ready. Generating rollouts...")

    while not stop_event.is_set():
        # Respect the interrupt strategy (soft_drain pauses, implicit_continuation
        # holds a forward-pass lock).
        if interrupt is not None:
            interrupt.wait_if_paused()
            if stop_event.is_set():
                break

        # Hot-swap weights if trainer has newer version
        current_trainer_version = version_val.value
        if current_trainer_version > local_version:
            try:
                dirs = globmod.glob(os.path.join(sync_dir, "v*"))
                # Filter out in-progress .tmp staging dirs
                committed = sorted(d for d in dirs if not d.endswith(".tmp"))
                if committed:
                    latest_dir = committed[-1]
                    v = _parse_version_dir(latest_dir)
                    if v > local_version and worker.update_weights(latest_dir):
                        local_version = v
                        print(f"[{tag}] Hot-swapped weights v{local_version}")
            except Exception as e:
                print(f"[{tag}] Weight reload error: {e}")

        # Sample and generate
        prompts = random.sample(train_data, min(2, len(train_data)))

        if interrupt is not None:
            interrupt.begin_generation()
        try:
            rollouts = worker.generate_rollouts(prompts, local_version)
        finally:
            if interrupt is not None:
                interrupt.end_generation()

        # Score and push
        for r in rollouts:
            scored = asyncio.run(scorer.score(r))
            if redis_client is not None:
                try:
                    from buffers.redis_stream import _rollout_to_dict
                    redis_client.xadd(redis_stream_key, _rollout_to_dict(scored))
                except Exception as e:
                    if stop_event.is_set():
                        break
                    print(f"[{tag}] redis xadd failed: {e}")
            else:
                try:
                    queue.put(scored, timeout=5)
                except Exception:
                    if stop_event.is_set():
                        break

    print(f"[{tag}] Stopped.")
