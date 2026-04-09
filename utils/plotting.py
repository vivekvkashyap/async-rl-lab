"""Publication-quality matplotlib plots for experiments."""
import json
import os
from typing import List, Dict, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Consistent color palette across experiments
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def load_metrics(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def plot_training_curves(
    metrics_files: List[str],
    labels: List[str],
    key: str = "gsm8k_accuracy",
    title: str = "GSM8K Accuracy",
    ylabel: str = "Accuracy",
    output_path: str = "results/accuracy_curves",
    smooth: int = 1,
):
    """Plot overlaid training curves, one line per config."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (path, label) in enumerate(zip(metrics_files, labels)):
        data = load_metrics(path)
        steps = [d["step"] for d in data if key in d]
        values = [d[key] for d in data if key in d]
        if smooth > 1 and len(values) > smooth:
            values = np.convolve(values, np.ones(smooth) / smooth, mode="valid")
            steps = steps[:len(values)]
        ax.plot(steps, values, label=label, color=COLORS[i % len(COLORS)], linewidth=2)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=150)
    plt.savefig(f"{output_path}.svg")
    plt.close()


def plot_throughput_comparison(
    metrics_files: List[str],
    labels: List[str],
    output_path: str = "results/throughput_comparison",
):
    """Bar chart comparing throughput across configs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    throughputs = []
    for path in metrics_files:
        data = load_metrics(path)
        tps = [d.get("tokens_per_second", 0) for d in data if "tokens_per_second" in d]
        throughputs.append(np.mean(tps) if tps else 0)

    bars = ax.bar(range(len(labels)), throughputs, color=COLORS[:len(labels)])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Tokens/second", fontsize=12)
    ax.set_title("Generation Throughput Comparison", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=150)
    plt.savefig(f"{output_path}.svg")
    plt.close()


def plot_staleness_distribution(
    metrics_file: str,
    output_path: str = "results/staleness_distribution",
):
    """Histogram of rollout staleness across training."""
    data = load_metrics(metrics_file)
    staleness = [d["batch_staleness_mean"] for d in data if "batch_staleness_mean" in d]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(staleness, bins=30, color=COLORS[0], alpha=0.7, edgecolor="black")
    ax.set_xlabel("Batch Staleness (mean)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Staleness Distribution", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=150)
    plt.savefig(f"{output_path}.svg")
    plt.close()


def plot_sync_latency(
    results: Dict[str, List[float]],
    output_path: str = "results/sync_latency",
):
    """Bar chart comparing sync latency for different mechanisms."""
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(results.keys())
    means = [np.mean(v) for v in results.values()]
    stds = [np.std(v) for v in results.values()]
    p99s = [np.percentile(v, 99) for v in results.values()]

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, means, width, label="Mean", color=COLORS[0], yerr=stds, capsize=5)
    ax.bar(x + width / 2, p99s, width, label="P99", color=COLORS[1])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Latency (seconds)", fontsize=12)
    ax.set_title("Weight Sync Latency Comparison", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=150)
    plt.savefig(f"{output_path}.svg")
    plt.close()


def plot_reward_curves(
    metrics_files: List[str],
    labels: List[str],
    output_path: str = "results/reward_curves",
):
    """Plot reward curves overlaid."""
    plot_training_curves(
        metrics_files, labels,
        key="reward_mean",
        title="Reward Over Training",
        ylabel="Mean Reward",
        output_path=output_path,
        smooth=5,
    )


def plot_is_weight_variance(
    metrics_files: List[str],
    labels: List[str],
    output_path: str = "results/is_weight_variance",
):
    """Plot IS weight variance over training."""
    plot_training_curves(
        metrics_files, labels,
        key="is_weight_variance",
        title="IS Weight Variance",
        ylabel="Variance",
        output_path=output_path,
    )
