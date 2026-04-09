"""Orchestrates process launching for async RL training.

Handles:
  - Spawning vLLM inference workers (one per inference GPU)
  - Single-GPU training (direct, no FSDP)
  - Multi-GPU training (FSDP via mp.spawn)
  - Cleanup of all processes on exit
"""
import asyncio
import multiprocessing as mp
import os
import torch

from utils.factory import (
    seed_everything, get_dtype,
    create_buffer, create_syncer, create_staleness,
    create_interrupt, create_scorer, create_trainer,
)
from utils.gpu_allocator import allocate_gpus


def launch(config: dict):
    """Main entry point. Allocates GPUs, spawns inference + training."""
    seed_everything(config.get("seed", 42))

    gpu_alloc = allocate_gpus(config)
    print(gpu_alloc.summary())

    # Load dataset (shared with inference workers)
    from utils.gsm8k import load_gsm8k
    train_data = load_gsm8k("train")

    # Shared state
    rollout_queue = mp.Queue(maxsize=64 * len(gpu_alloc.infer_gpu_ids))
    version_val = mp.Value("i", 0)
    stop_event = mp.Event()
    sync_dir = config["weight_sync"].get("checkpoint_dir", "/tmp/async-rl-lab/checkpoints")

    model_name = config["model"]["name"]
    dtype_str = config["model"].get("inference_dtype", config["model"].get("dtype", "bf16"))

    # Launch inference FIRST (before loading training model, avoids CUDA conflicts)
    inference_procs = _spawn_inference_workers(
        gpu_alloc.infer_gpu_ids, model_name, dtype_str, config,
        train_data, rollout_queue, version_val, stop_event, sync_dir,
    )

    num_train_gpus = len(gpu_alloc.train_gpu_ids)
    dtype = get_dtype(config["model"].get("dtype", "bf16"))

    try:
        if num_train_gpus > 1:
            _launch_fsdp_training(
                gpu_alloc.train_gpu_ids, config, model_name, dtype,
                rollout_queue, version_val, stop_event, gpu_alloc.infer_gpu_ids,
            )
        else:
            _launch_single_gpu_training(
                gpu_alloc.train_device, config, model_name, dtype,
                rollout_queue, version_val, stop_event, gpu_alloc.infer_gpu_ids,
            )
    finally:
        _cleanup(inference_procs, stop_event)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _spawn_inference_workers(infer_gpu_ids, model_name, dtype_str, config,
                              train_data, queue, version_val, stop_event, sync_dir):
    """Spawn one vLLM inference process per inference GPU."""
    procs = []
    for i, gpu_id in enumerate(infer_gpu_ids):
        proc = mp.Process(
            target=_inference_worker_target,
            args=(gpu_id, i, model_name, dtype_str, config,
                  train_data, queue, version_val, stop_event, sync_dir),
        )
        proc.start()
        procs.append(proc)
        print(f"Launched inference worker {i} on GPU {gpu_id} (PID={proc.pid})")
    return procs


def _inference_worker_target(gpu_id, worker_id, model_name, dtype_str, config,
                              train_data, queue, version_val, stop_event, sync_dir):
    """Target for inference subprocess. Sets CUDA_VISIBLE_DEVICES then runs vLLM."""
    # Create new process group so we can kill all children (vLLM EngineCore) together
    os.setpgrp()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from core.inference_process import run_inference

    dtype_map = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
    run_inference(
        model_name=model_name,
        dtype=dtype_map.get(dtype_str, dtype_str),
        group_size=config["training"]["group_size"],
        max_completion_length=config["training"].get("max_completion_length", 512),
        temperature=config["training"].get("temperature", 0.7),
        top_p=config["training"].get("top_p", 0.9),
        gpu_memory_utilization=config.get("deployment", {}).get("gpu_memory_utilization", 0.85),
        tensor_parallel_size=config.get("deployment", {}).get("tensor_parallel_size", 1),
        train_data=train_data,
        queue=queue,
        version_val=version_val,
        stop_event=stop_event,
        sync_dir=sync_dir,
        worker_id=worker_id,
    )


# ---------------------------------------------------------------------------
# Single-GPU training
# ---------------------------------------------------------------------------

def _launch_single_gpu_training(train_device, config, model_name, dtype,
                                 rollout_queue, version_val, stop_event, infer_gpu_ids):
    """Load model on a single GPU and run the coordinator."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from core.coordinator import Coordinator
    from core.trainer import GRPOTrainer
    from utils.metrics import MetricsTracker

    print(f"\nLoading training model on {train_device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(train_device)

    trainer, coordinator = _build_coordinator(
        model, tokenizer, train_device, config,
        rollout_queue, version_val, stop_event, infer_gpu_ids,
    )
    asyncio.run(coordinator.run())


# ---------------------------------------------------------------------------
# Multi-GPU FSDP training
# ---------------------------------------------------------------------------

def _launch_fsdp_training(train_gpu_ids, config, model_name, dtype,
                           rollout_queue, version_val, stop_event, infer_gpu_ids):
    """Launch FSDP training across multiple GPUs via mp.spawn."""
    num_gpus = len(train_gpu_ids)
    print(f"\nLaunching FSDP training across {num_gpus} GPUs: {train_gpu_ids}")
    mp.spawn(
        _fsdp_rank_entrypoint,
        args=(num_gpus, train_gpu_ids, config, model_name, dtype,
              rollout_queue, version_val, stop_event, infer_gpu_ids),
        nprocs=num_gpus,
        join=True,
    )


def _fsdp_rank_entrypoint(rank, world_size, train_gpu_ids, config, model_name, dtype,
                           rollout_queue, version_val, stop_event, infer_gpu_ids):
    """Entry point for each FSDP rank. Rank 0 runs coordinator, others follow."""
    from utils.fsdp import setup_distributed, cleanup_distributed, wrap_model_fsdp
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gpu_id = train_gpu_ids[rank]
    device = f"cuda:{gpu_id}"
    master_port = config.get("deployment", {}).get("fsdp_master_port", 29500)

    setup_distributed(rank, world_size, master_port)
    seed_everything(config.get("seed", 42) + rank)

    if rank == 0:
        print(f"[FSDP] Initializing {world_size} training ranks on GPUs {train_gpu_ids}")
        print(f"[FSDP rank {rank}] Loading model on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(device)

    cpu_offload = config.get("training", {}).get("fsdp_cpu_offload", False)
    model = wrap_model_fsdp(model, train_gpu_ids, dtype=dtype, cpu_offload=cpu_offload)

    trainer = create_trainer(model, tokenizer, device, config)

    if rank == 0:
        _, coordinator = _build_coordinator(
            model, tokenizer, device, config,
            rollout_queue, version_val, stop_event, infer_gpu_ids,
            trainer=trainer,
        )
        try:
            asyncio.run(coordinator.run())
        finally:
            stop_event.set()
    else:
        from utils.fsdp import run_fsdp_follower
        run_fsdp_follower(trainer, stop_event, config)

    cleanup_distributed()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_wandb_config(config: dict) -> dict:
    """Build wandb config from experiment config. Returns None if disabled."""
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None
    return {
        "enabled": True,
        "project": wandb_cfg.get("project", "async-rl-lab"),
        "run_name": wandb_cfg.get("run_name"),
        "group": wandb_cfg.get("group"),
        "tags": wandb_cfg.get("tags", []),
        "config": config,  # Log full experiment config to wandb
    }


def _build_coordinator(model, tokenizer, device, config,
                        rollout_queue, version_val, stop_event, infer_gpu_ids,
                        trainer=None):
    """Build trainer (if not provided) and coordinator from config."""
    from core.coordinator import Coordinator
    from utils.metrics import MetricsTracker

    if trainer is None:
        trainer = create_trainer(model, tokenizer, device, config)

    coordinator = Coordinator(
        config=config, trainer=trainer,
        buffer=create_buffer(config),
        syncer=create_syncer(config),
        staleness=create_staleness(config, model=model, tokenizer=tokenizer, device=device),
        interrupt=create_interrupt(config),
        scorer=create_scorer(config),
        metrics=MetricsTracker(
            log_every=config.get("metrics", {}).get("log_every", 1),
            output_dir=config.get("metrics", {}).get("plot_dir", "results/"),
            wandb_config=_build_wandb_config(config),
        ),
        rollout_queue=rollout_queue,
        version_val=version_val,
        stop_event=stop_event,
        infer_gpu_ids=infer_gpu_ids,
    )
    return trainer, coordinator


def _cleanup(inference_procs, stop_event):
    """Stop inference processes and their subprocesses (e.g. vLLM EngineCore)."""
    import signal as sig

    stop_event.set()
    for proc in inference_procs:
        proc.join(timeout=15)
        if proc.is_alive():
            # Kill entire process group (catches vLLM's internal EngineCore subprocess)
            try:
                os.killpg(proc.pid, sig.SIGTERM)
            except OSError:
                pass
            proc.join(timeout=5)
            if proc.is_alive():
                try:
                    os.killpg(proc.pid, sig.SIGKILL)
                except OSError:
                    pass
                proc.join(timeout=3)
    print("All processes cleaned up.")
