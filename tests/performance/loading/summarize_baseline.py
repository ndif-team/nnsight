#!/usr/bin/env python3
"""Aggregate loading-baseline JSONs into one table.

Reads logs/baseline/<GPU>_<tier>_<model>_<cold|warm>_<stamp>.json (written by
run_baseline.slurm) and prints, per (GPU, storage tier, cache, model), the best
wall time / effective bandwidth of every experiment plus its speedup over HF.

Usage:
    python summarize_baseline.py [--dir logs/baseline] [--csv out.csv] [--all-configs]
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

NAME_RE = re.compile(r"^(?P<gpu>[^_]+)_(?P<tier>nvme|tmp)_(?P<model>.+)_(?P<cache>cold|warm)_(?P<stamp>\d{8}_\d{6})\.json$")

EXP_ORDER = ["hf", "hf_pread", "stream", "gpu_direct", "lazy"]
EXP_LABEL = {
    "hf": "HF from_pretrained (mmap)",
    "hf_pread": "HF from_pretrained (pread)",
    "stream": "run:ai stream (CPU clone)",
    "gpu_direct": "run:ai GPU-direct",
    "lazy": "run:ai GPU-direct lazy",
}


def load(dirname):
    rows = []
    for path in sorted(glob.glob(os.path.join(dirname, "*.json"))):
        m = NAME_RE.match(os.path.basename(path))
        if not m:
            continue
        with open(path) as f:
            data = json.load(f)
        size = data["model_size_gb"]
        env = data.get("environment", {})
        for r in data["results"]:
            if r.get("error"):
                continue
            knob = r["config"].get("workers") if r["experiment"].startswith("hf") else r["config"].get("concurrency")
            rows.append({
                **m.groupdict(),
                "experiment": r["experiment"],
                "knob": knob,
                "repeat": r["config"].get("repeat", 0),
                "wall_s": r["wall_time_s"],
                "bw_gbs": size / r["wall_time_s"] if r["wall_time_s"] else 0.0,
                "peak_rss_mb": r["peak_rss_mb"],
                "peak_private_mb": r["peak_private_mb"],
                "peak_gpu_reserved_mb": r.get("peak_gpu_reserved_mb", 0.0),
                "model_size_gb": size,
                "transformers": env.get("transformers"),
                "gpus": ",".join(env.get("gpus", [])),
                "file": os.path.basename(path),
            })
    return rows


def best_per_experiment(rows):
    best = {}
    for r in rows:
        key = (r["gpu"], r["tier"], r["cache"], r["model"], r["experiment"])
        if key not in best or r["wall_s"] < best[key]["wall_s"]:
            best[key] = r
    return best


def write_markdown(path, groups, gpu_order):
    """One Markdown table per model, rows = (GPU, tier, cache), columns = experiments."""
    by_model = defaultdict(list)
    for group in groups:
        by_model[group[3]].append(group)
    lines = []
    for model in sorted(by_model):
        any_row = next(iter(groups[by_model[model][0]].values()))
        lines.append(f"### {model} ({any_row['model_size_gb']:.1f} GB)\n")
        lines.append("Best wall time per path (s), speedup vs HF mmap in parentheses; " +
                     "GB/s = model size / wall.\n")
        header = "| GPU | storage | cache | " + " | ".join(EXP_LABEL[e] for e in EXP_ORDER) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (3 + len(EXP_ORDER)))
        for group in sorted(by_model[model], key=lambda g: (gpu_order.get(g[0], 9), g[1], g[2])):
            gpu, tier, cache, _ = group
            exps = groups[group]
            hf = exps.get("hf")
            cells = []
            for e in EXP_ORDER:
                r = exps.get(e)
                if r is None:
                    cells.append("-")
                elif e == "hf" or hf is None:
                    cells.append(f"{r['wall_s']:.1f} s, {r['bw_gbs']:.2f} GB/s")
                else:
                    cells.append(f"{r['wall_s']:.1f} s ({hf['wall_s'] / r['wall_s']:.2f}x)")
            tier_name = "/work/nvme (Lustre)" if tier == "nvme" else "/tmp (local NVMe)"
            lines.append(f"| {gpu} | {tier_name} | {cache} | " + " | ".join(cells) + " |")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote markdown tables to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="logs/baseline")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--all-configs", action="store_true", help="print every config, not just the best")
    ap.add_argument("--markdown", default=None, help="also write the best-per-experiment tables as Markdown")
    args = ap.parse_args()

    rows = load(args.dir)
    if not rows:
        print(f"no baseline JSONs in {args.dir}")
        return
    best = best_per_experiment(rows)

    groups = defaultdict(dict)
    for key, r in best.items():
        groups[key[:4]][key[4]] = r

    gpu_order = {"A40": 0, "A100": 1, "H200": 2}
    for group in sorted(groups, key=lambda g: (gpu_order.get(g[0], 9), g[1], g[2], g[3])):
        gpu, tier, cache, model = group
        exps = groups[group]
        hf = exps.get("hf")
        any_row = next(iter(exps.values()))
        print(f"\n== {gpu} | {'network /work/nvme' if tier == 'nvme' else 'node-local /tmp'} | {cache} cache | {model} "
              f"({any_row['model_size_gb']:.1f} GB) | transformers {any_row['transformers']} | {any_row['gpus']}")
        print(f"  {'experiment':<28} {'knob':>5} {'wall (s)':>9} {'GB/s':>6} {'vs HF':>6} {'RSS MB':>8} {'priv MB':>8}")
        for exp in EXP_ORDER:
            r = exps.get(exp)
            if r is None:
                continue
            speed = f"{hf['wall_s'] / r['wall_s']:.2f}x" if hf and r["wall_s"] else "-"
            print(f"  {EXP_LABEL[exp]:<28} {str(r['knob']):>5} {r['wall_s']:>9.2f} {r['bw_gbs']:>6.2f} {speed:>6} "
                  f"{r['peak_rss_mb']:>8.0f} {r['peak_private_mb']:>8.0f}")

    if args.markdown:
        write_markdown(args.markdown, groups, gpu_order)

    if args.all_configs:
        print("\n== every config")
        for r in sorted(rows, key=lambda r: (gpu_order.get(r["gpu"], 9), r["tier"], r["cache"], r["model"], r["experiment"], r["knob"] or 0)):
            print(f"  {r['gpu']:<5} {r['tier']:<4} {r['cache']:<4} {r['model']:<22} {r['experiment']:<10} knob={str(r['knob']):>3} "
                  f"rep={r['repeat']} wall={r['wall_s']:7.2f}s bw={r['bw_gbs']:5.2f} GB/s")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
