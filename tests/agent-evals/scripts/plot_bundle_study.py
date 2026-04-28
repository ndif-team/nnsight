"""Plot results from results/<bundle>.json files into bar charts.

Reads every <bundle>.json the bundle-study script produced, builds two PNGs:

  results/bundle_comparison.png       — overall + by-kind pass rates per bundle
  results/bundle_comparison_diff.png  — pass rate broken out by difficulty

Run after `scripts/run_bundle_study.sh` finishes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).parent.parent
RESULTS = HERE / "results"

# Order to display bundles (smallest → largest, then legacy at the end).
BUNDLE_ORDER = ["minimal", "router", "full", "legacy"]


def load_results(mode: str = "static") -> dict[str, dict]:
    """Load <bundle>.json (static) or <bundle>.browse.json (browse) files."""
    suffix = ".browse" if mode == "browse" else ""
    out = {}
    for name in BUNDLE_ORDER:
        p = RESULTS / f"{name}{suffix}.json"
        if p.exists():
            with open(p) as f:
                out[name] = json.load(f)
        else:
            # Don't warn for legacy in browse mode — it's not defined there.
            if not (mode == "browse" and name == "legacy"):
                print(f"  warning: {p} not found, skipping", file=sys.stderr)
    return out


def fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"


def plot_overview(results: dict[str, dict], outpath: Path) -> None:
    bundles = list(results.keys())
    overall = [results[b]["summary"]["pass_rate"] for b in bundles]
    code = [
        (
            results[b]["by_kind"]["code"]["passed"]
            / results[b]["by_kind"]["code"]["total"]
        )
        if results[b]["by_kind"]["code"]["total"] > 0
        else 0
        for b in bundles
    ]
    mcq = [
        (
            results[b]["by_kind"]["mcq"]["passed"]
            / results[b]["by_kind"]["mcq"]["total"]
        )
        if results[b]["by_kind"]["mcq"]["total"] > 0
        else 0
        for b in bundles
    ]

    x = np.arange(len(bundles))
    w = 0.27

    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - w, overall, w, label="Overall", color="#444")
    b2 = ax.bar(x, code, w, label="Code", color="#1f77b4")
    b3 = ax.bar(x + w, mcq, w, label="MCQ", color="#ff7f0e")

    for bars in (b1, b2, b3):
        for r in bars:
            h = r.get_height()
            ax.annotate(
                f"{h*100:.0f}%",
                xy=(r.get_x() + r.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(bundles)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pass rate")
    model = next(iter(results.values()))["model"]
    ax.set_title(
        f"nnsight agent-eval — pass rate by doc bundle ({model})"
    )
    ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x*100)}%"))
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"  wrote {outpath}")


def plot_by_difficulty(results: dict[str, dict], outpath: Path) -> None:
    bundles = list(results.keys())
    diffs = ["basic", "intermediate", "advanced"]
    rates = {d: [] for d in diffs}
    for b in bundles:
        for d in diffs:
            stats = results[b]["by_difficulty"][d]
            rates[d].append(
                stats["passed"] / stats["total"] if stats["total"] > 0 else 0
            )

    x = np.arange(len(bundles))
    w = 0.27
    colors = {"basic": "#2ca02c", "intermediate": "#1f77b4", "advanced": "#d62728"}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, d in enumerate(diffs):
        offset = (i - 1) * w
        ax.bar(x + offset, rates[d], w, label=d.capitalize(), color=colors[d])

    for i, d in enumerate(diffs):
        offset = (i - 1) * w
        for j, v in enumerate(rates[d]):
            ax.annotate(
                f"{v*100:.0f}%",
                xy=(x[j] + offset, v),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(bundles)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pass rate")
    model = next(iter(results.values()))["model"]
    ax.set_title(f"nnsight agent-eval — pass rate by difficulty × bundle ({model})")
    ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x*100)}%"))
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"  wrote {outpath}")


def print_table(results: dict[str, dict]) -> None:
    print()
    print(f"{'bundle':>10}  {'overall':>9}  {'code':>9}  {'mcq':>9}  {'basic':>9}  {'inter':>9}  {'advan':>9}")
    print(f"{'-'*10}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")
    for b, r in results.items():
        s = r["summary"]
        bk = r["by_kind"]
        bd = r["by_difficulty"]

        def pct(p, t):
            return fmt_pct(p / t) if t else "n/a"

        print(
            f"{b:>10}  {fmt_pct(s['pass_rate']):>9}  "
            f"{pct(bk['code']['passed'], bk['code']['total']):>9}  "
            f"{pct(bk['mcq']['passed'], bk['mcq']['total']):>9}  "
            f"{pct(bd['basic']['passed'], bd['basic']['total']):>9}  "
            f"{pct(bd['intermediate']['passed'], bd['intermediate']['total']):>9}  "
            f"{pct(bd['advanced']['passed'], bd['advanced']['total']):>9}"
        )
    print()


def plot_static_vs_browse(static: dict, browse: dict, outpath: Path) -> None:
    """Two clusters of bars per bundle: static pass-rate, browse pass-rate."""
    bundles = [b for b in BUNDLE_ORDER if b in static or b in browse]
    static_rates = [
        static[b]["summary"]["pass_rate"] if b in static else 0 for b in bundles
    ]
    browse_rates = [
        browse[b]["summary"]["pass_rate"] if b in browse else 0 for b in bundles
    ]

    x = np.arange(len(bundles))
    w = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - w / 2, static_rates, w, label="static (full bundle in prompt)", color="#1f77b4")
    b2 = ax.bar(x + w / 2, browse_rates, w, label="browse (router + Read tool)", color="#ff7f0e")

    for bars, vals in ((b1, static_rates), (b2, browse_rates)):
        for r, v in zip(bars, vals):
            if v == 0:
                continue
            ax.annotate(
                f"{v*100:.0f}%",
                xy=(r.get_x() + r.get_width() / 2, v),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(bundles)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pass rate")
    model = next(
        (r["model"] for d in (static, browse) for r in d.values()),
        "?",
    )
    ax.set_title(f"nnsight agent-eval — static vs browse mode by bundle ({model})")
    ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x*100)}%"))
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"  wrote {outpath}")


def main():
    static = load_results("static")
    browse = load_results("browse")

    if not static and not browse:
        print("No bundle results found in results/. Run scripts/run_bundle_study.sh first.")
        return 1

    if static:
        print("\n=== static mode ===")
        print_table(static)
        plot_overview(static, RESULTS / "bundle_comparison.png")
        plot_by_difficulty(static, RESULTS / "bundle_comparison_diff.png")

    if browse:
        print("\n=== browse mode ===")
        print_table(browse)
        plot_overview(browse, RESULTS / "bundle_comparison_browse.png")
        plot_by_difficulty(browse, RESULTS / "bundle_comparison_browse_diff.png")

    if static and browse:
        plot_static_vs_browse(
            static, browse, RESULTS / "bundle_comparison_static_vs_browse.png"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
