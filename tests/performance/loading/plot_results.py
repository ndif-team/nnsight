#!/usr/bin/env python3
"""Plot benchmark results from JSON log files."""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

MODEL_SIZE_GB = 61.024  # Qwen3-32B

# Latest result files
FILES = {
    ("Local NVMe (/tmp)", "Cold"):         "logs/tmp/bench_loading_Qwen3-32B_cold_20260318_024952.json",
    ("Local NVMe (/tmp)", "Warm"):         "logs/tmp/bench_loading_Qwen3-32B_warm_20260318_024952.json",
    ("Network NVMe (/work/nvme)", "Cold"): "logs/nvme/bench_loading_Qwen3-32B_cold_20260318_175339.json",
    ("Network NVMe (/work/nvme)", "Warm"): "logs/nvme/bench_loading_Qwen3-32B_warm_20260318_175339.json",
}

STYLE = {
    "hf":                        {"color": "#2196F3", "marker": "s", "label": "HF from_pretrained (mmap)"},
    "runai_stream":              {"color": "#FF5722", "marker": "o", "label": "run:ai streaming cache"},
    "runai_gpu_direct_unpinned": {"color": "#4CAF50", "marker": "^", "label": "run:ai GPU-direct (unpinned)"},
}

EXPERIMENTS = ["hf", "runai_stream", "runai_gpu_direct_unpinned"]


def load_data():
    """Parse JSON files into nested dict:
    {(storage, cache, experiment): {"x": [...], "bw": [...], "wall": [...], "rss": [...], "private": [...]}}
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

            # Handle both old (peak_cpu_mem_mb) and new (peak_rss_mb/peak_private_mb) field names
            rss = r.get("peak_rss_mb") or r.get("peak_cpu_mem_mb", 0)
            private = r.get("peak_private_mb", rss)

            # HF's "private" memory is inflated by MAP_PRIVATE mmap pages
            # from safetensors (~model_size).  These are read-only page cache
            # pages, not real allocations.  Correct by subtracting model size;
            # the residual (~3 GiB) is the true process overhead.
            if exp == "hf":
                private = max(private - MODEL_SIZE_GB * 1024, 0)

            key = (storage, cache, exp)
            if key not in series:
                series[key] = {"x": [], "bw": [], "wall": [], "rss": [], "private": []}
            series[key]["x"].append(x)
            series[key]["bw"].append(bw)
            series[key]["wall"].append(wall)
            series[key]["rss"].append(rss / 1024)        # → GiB
            series[key]["private"].append(private / 1024)  # → GiB
    return series


def _setup_axes(ax, storage, cache, xlabel=True):
    ax.set_title(f"{storage} — {cache} cache", fontsize=11)
    if xlabel:
        ax.set_xlabel("Workers / Concurrency", fontsize=10)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)


def plot_bandwidth(series):
    storages = ["Local NVMe (/tmp)", "Network NVMe (/work/nvme)"]
    caches = ["Cold", "Warm"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    fig.suptitle("Qwen3-32B (61 GB) — Loading Bandwidth",
                 fontsize=14, fontweight="bold")

    for row, storage in enumerate(storages):
        for col, cache in enumerate(caches):
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
    out = "logs/bench_loading_bandwidth.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_wall_time(series):
    storages = ["Local NVMe (/tmp)", "Network NVMe (/work/nvme)"]
    caches = ["Cold", "Warm"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    fig.suptitle("Qwen3-32B (61 GB) — Loading Wall Time",
                 fontsize=14, fontweight="bold")

    for row, storage in enumerate(storages):
        for col, cache in enumerate(caches):
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
    out = "logs/bench_loading_walltime.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_memory(series):
    storages = ["Local NVMe (/tmp)", "Network NVMe (/work/nvme)"]
    caches = ["Cold", "Warm"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    fig.suptitle("Qwen3-32B (61 GB) — Peak CPU Memory\n"
                 "HF corrected for MAP_PRIVATE mmap overhead",
                 fontsize=13, fontweight="bold")

    for row, storage in enumerate(storages):
        for col, cache in enumerate(caches):
            ax = axes[row][col]
            for exp in EXPERIMENTS:
                key = (storage, cache, exp)
                if key not in series:
                    continue
                d = series[key]
                s = STYLE[exp]
                ax.plot(d["x"], d["private"], marker=s["marker"], color=s["color"],
                        label=s["label"], linewidth=2, markersize=7)

            # Reference line for model size
            ax.axhline(y=MODEL_SIZE_GB, color="gray", linestyle="--",
                       alpha=0.5, linewidth=1)
            ax.text(1.1, MODEL_SIZE_GB + 1, "model size (61 GB)",
                    color="gray", fontsize=8, alpha=0.7)

            ax.set_ylabel("Peak Private Memory (GiB)", fontsize=10)
            _setup_axes(ax, storage, cache)

    plt.tight_layout()
    out = "logs/bench_loading_memory.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    series = load_data()
    plot_bandwidth(series)
    plot_wall_time(series)
    plot_memory(series)
