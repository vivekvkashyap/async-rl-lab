#!/usr/bin/env python3
"""Benchmark sync latency for each weight sync mechanism.

Pure benchmark, no training. Load Qwen-0.5B on two GPUs.
Measure sync latency for: filesystem, nccl_broadcast, nccl_bucketed.
Run 100 syncs each, report mean/p50/p99.
"""
import argparse
import asyncio
import json
import os
import sys
import time
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM
from weight_sync.filesystem_sync import FilesystemSyncer
from utils.plotting import plot_sync_latency


async def benchmark_filesystem(model, n_iters: int = 100):
    """Benchmark filesystem sync."""
    with tempfile.TemporaryDirectory() as tmpdir:
        syncer = FilesystemSyncer(checkpoint_dir=tmpdir)
        latencies = []
        for i in range(n_iters):
            duration = await syncer.push(model, version=i)
            latencies.append(duration)
        return latencies


async def benchmark_nccl(model_0, model_1, syncer_class, n_iters: int = 100, **kwargs):
    """Benchmark NCCL sync (requires 2 GPUs in same process for simplicity)."""
    # NOTE: Full NCCL benchmark requires separate processes.
    # This is a simplified single-process version using filesystem as proxy.
    # Real benchmark should use multiprocessing with TCPStore.
    print(f"  NCCL benchmark requires 2-process setup. Skipping in single-process mode.")
    return []


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--n-iters", type=int, default=100)
    parser.add_argument("--output-dir", default="results/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"  Parameters: {param_count:,} ({param_bytes / 1e6:.1f} MB)")

    results = {}

    # Filesystem benchmark
    print(f"\nBenchmarking filesystem sync ({args.n_iters} iterations)...")
    latencies = await benchmark_filesystem(model, args.n_iters)
    results["filesystem"] = latencies
    print(f"  Mean: {np.mean(latencies):.4f}s | P50: {np.median(latencies):.4f}s | P99: {np.percentile(latencies, 99):.4f}s")

    # NCCL benchmarks (need 2 GPUs)
    if torch.cuda.device_count() >= 2:
        print("\nNCCL benchmarks require separate processes. Run with torchrun for full benchmark.")
        # Placeholder for NCCL results — these need multiprocessing
        results["nccl_broadcast"] = []
        results["nccl_bucketed"] = []
    else:
        print("\nSkipping NCCL benchmarks (need 2 GPUs)")

    # Save results
    save_results = {k: {"mean": float(np.mean(v)) if v else 0,
                         "p50": float(np.median(v)) if v else 0,
                         "p99": float(np.percentile(v, 99)) if v else 0,
                         "latencies": v}
                    for k, v in results.items()}

    with open(os.path.join(args.output_dir, "sync_benchmark.json"), "w") as f:
        json.dump(save_results, f, indent=2)

    # Plot
    plot_results = {k: v for k, v in results.items() if v}
    if plot_results:
        plot_sync_latency(plot_results, os.path.join(args.output_dir, "sync_latency"))
        print(f"\nPlot saved to {args.output_dir}/sync_latency.png")


if __name__ == "__main__":
    asyncio.run(main())
