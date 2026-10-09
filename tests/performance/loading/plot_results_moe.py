#!/usr/bin/env python3
"""Plot benchmark results for MoE model (Qwen3-30B-A3B) from JSON log files."""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

MODEL_NAME = "Qwen3-30B-A3B"
MODEL_SIZE_GB = 56.873  # Qwen3-30B-A3B

# Latest result files
FILES = {
    ("Local NVMe (/tmp)", "Cold"):         "logs/tmp/bench_loading_Qwen3-30B-A3B_cold_20260319_044753.json",
    ("Local NVMe (/tmp)", "Warm"):         "logs/tmp/bench_loading_Qwen3-30B-A3B_warm_20260319_044753.json",
    ("Network NVMe (/work/nvme)", "Cold"): "logs/nvme/bench_loading_Qwen3-30B-A3B_cold_20260319_044619.json",
    ("Network NVMe (/work/nvme)", "Warm"): "logs/nvme/bench_loading_Qwen3-30B-A3B_20260319_044619.json",
}

STYLE = {
    "hf":         {"color": "#2196F3", "marker": "s", "label": "HF from_pretrained (mmap)"},
    "stream":     {"color": "#FF5722", "marker": "o", "label": "run:ai streaming cache"},
    "gpu_direct": {"color": "#4CAF50", "marker": "^", "label": "run:ai GPU-direct"},
    "lazy":       {"color": "#9C27B0", "marker": "D", "label": "run:ai lazy (no GPU-direct)"},
}

EXPERIMENTS = ["hf", "stream", "gpu_direct", "lazy"]


def load_data():
    """Parse JSON files into nested dict:
    {(storage, cache, experiment): {"x": [...], "bw": [...], "wall": [...], ...}}
    """
    series = {}
    for (storage, cache), path in FILES.items():
        with open(path) as f:
            data = json.load(f)
        for r in data["results"]:
            if r.get("error"):
                continue
            exp = r["experiment"]
            cfg = r["config"]
            x = cfg.get("concurrency") or cfg.get("workers")
            if x >= 32:
                continue
            wall = r["wall_time_s"]
            bw = MODEL_SIZE_GB / wall

            rss = r.get("peak_rss_mb") or r.get("peak_cpu_mem_mb", 0)
            private = r.get("peak_private_mb", rss)

            # HF mmap correction
            if exp == "hf":
                private = max(private - MODEL_SIZE_GB * 1024, 0)

            gpu_alloc = r.get("peak_gpu_alloc_mb", 0)
            gpu_reserved = r.get("peak_gpu_reserved_mb", 0)

            key = (storage, cache, exp)
            if key not in series:
                series[key] = {"x": [], "bw": [], "wall": [], "rss": [],
                               "private": [], "gpu_alloc": [], "gpu_reserved": []}
            series[key]["x"].append(x)
            series[key]["bw"].append(bw)
            series[key]["wall"].append(wall)
            series[key]["rss"].append(rss / 1024)
            series[key]["private"].append(private / 1024)
            series[key]["gpu_alloc"].append(gpu_alloc / 1024)
            series[key]["gpu_reserved"].append(gpu_reserved / 1024)
    return series


STORAGES = ["Local NVMe (/tmp)", "Network NVMe (/work/nvme)"]
CACHES = ["Cold", "Warm"]


def _setup_axes(ax, storage, cache, xlabel=True):
    ax.set_title(f"{storage} — {cache} cache", fontsize=11)
    if xlabel:
        ax.set_xlabel("Workers / Concurrency", fontsize=10)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)


def plot_bandwidth(series):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    fig.suptitle(f"{MODEL_NAME} (MoE, {MODEL_SIZE_GB:.0f} GB) — Loading Bandwidth",
                 fontsize=14, fontweight="bold")

    for row, storage in enumerate(STORAGES):
        for col, cache in enumerate(CACHES):
            ax = axes[row][col]
            for exp in EXPERIMENTS:
                key = (storage, cache, exp)
                if key not in series:
                    continue
                d = series[key]
                s = STYLE[exp]
                ax.plot(d["x"], d["bw"], marker=s["marker"], color=s["color"],
                        label=s["label"], linewidth=2, markersize=7)
            ax.set_ylabel("Bandwidth (GB/s)", fontsize=10)
            _setup_axes(ax, storage, cache)

    plt.tight_layout()
    out = "logs/bench_loading_moe_bandwidth.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_wall_time(series):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    fig.suptitle(f"{MODEL_NAME} (MoE, {MODEL_SIZE_GB:.0f} GB) — Loading Wall Time",
                 fontsize=14, fontweight="bold")

    for row, storage in enumerate(STORAGES):
        for col, cache in enumerate(CACHES):
            ax = axes[row][col]
            for exp in EXPERIMENTS:
                key = (storage, cache, exp)
                if key not in series:
                    continue
                d = series[key]
                s = STYLE[exp]
                ax.plot(d["x"], d["wall"], marker=s["marker"], color=s["color"],
                        label=s["label"], linewidth=2, markersize=7)
            ax.set_ylabel("Wall Time (s)", fontsize=10)
            _setup_axes(ax, storage, cache)

    plt.tight_layout()
    out = "logs/bench_loading_moe_walltime.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_cpu_memory(series):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    fig.suptitle(f"{MODEL_NAME} (MoE, {MODEL_SIZE_GB:.0f} GB) — Peak CPU Memory\n"
                 "HF corrected for MAP_PRIVATE mmap overhead",
                 fontsize=13, fontweight="bold")

    for row, storage in enumerate(STORAGES):
        for col, cache in enumerate(CACHES):
            ax = axes[row][col]
            for exp in EXPERIMENTS:
                key = (storage, cache, exp)
                if key not in series:
                    continue
                d = series[key]
                s = STYLE[exp]
                ax.plot(d["x"], d["private"], marker=s["marker"], color=s["color"],
                        label=s["label"], linewidth=2, markersize=7)

            ax.axhline(y=MODEL_SIZE_GB, color="gray", linestyle="--",
                       alpha=0.5, linewidth=1)
            ax.text(1.1, MODEL_SIZE_GB + 1, f"model size ({MODEL_SIZE_GB:.0f} GB)",
                    color="gray", fontsize=8, alpha=0.7)

            ax.set_ylabel("Peak Private Memory (GiB)", fontsize=10)
            _setup_axes(ax, storage, cache)

    plt.tight_layout()
    out = "logs/bench_loading_moe_memory.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_gpu_memory(series):
    """Plot peak GPU allocated and reserved memory.
    For each experiment, two lines: allocated (solid) and reserved (dashed),
    using dark/light shades of the same color.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    fig.suptitle(f"{MODEL_NAME} (MoE, {MODEL_SIZE_GB:.0f} GB) — Peak GPU Memory\n"
                 "Allocated (solid) vs Reserved (dashed)",
                 fontsize=13, fontweight="bold")

    for row, storage in enumerate(STORAGES):
        for col, cache in enumerate(CACHES):
            ax = axes[row][col]
            for exp in EXPERIMENTS:
                key = (storage, cache, exp)
                if key not in series:
                    continue
                d = series[key]
                s = STYLE[exp]
                color = s["color"]

                # Allocated: solid, full opacity
                ax.plot(d["x"], d["gpu_alloc"], marker=s["marker"], color=color,
                        label=f'{s["label"]} (alloc)',
                        linewidth=2, markersize=7, linestyle="-")

                # Reserved: dashed, lighter
                ax.plot(d["x"], d["gpu_reserved"], marker=s["marker"], color=color,
                        label=f'{s["label"]} (reserved)',
                        linewidth=2, markersize=5, linestyle="--", alpha=0.5)

            ax.set_ylabel("GPU Memory (GiB)", fontsize=10)
            _setup_axes(ax, storage, cache)

    plt.tight_layout()
    out = "logs/bench_loading_moe_gpu_memory.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    series = load_data()
    plot_bandwidth(series)
    plot_wall_time(series)
    plot_cpu_memory(series)
    plot_gpu_memory(series)
