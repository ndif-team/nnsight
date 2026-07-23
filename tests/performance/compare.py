"""Diff two interleave_bench result files side by side.

    python compare.py results/old.json results/new.json

Each row is a benchmark's median, the two runs next to each other, and their
ratio (second / first). Ratio > 1 means the second run is slower.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _median(node) -> float | None:
    return node["us_median"] if isinstance(node, dict) and "us_median" in node else None


def _rows(benchmarks: dict) -> "dict[str, float]":
    """Flatten a benchmarks dict to an ordered {label: median_us}.

    Insertion order is kept — single-shot benchmarks first, then each scaling
    sweep with its steps in numeric order — so the table reads top to bottom the
    way the harness ran, not alphabetized.
    """
    rows: dict[str, float] = {}
    for key, value in benchmarks.items():
        median = _median(value)
        if median is not None:
            rows[key] = median
        elif isinstance(value, dict):  # a scaling sweep: {step: stats}
            for sub, stats in sorted(value.items(), key=lambda kv: int(kv[0])):
                m = _median(stats)
                if m is not None:
                    rows[f"{key}[{sub}]"] = m
    return rows


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    a_path, b_path = Path(sys.argv[1]), Path(sys.argv[2])
    a = json.loads(a_path.read_text())
    b = json.loads(b_path.read_text())

    a_rows, b_rows = _rows(a["benchmarks"]), _rows(b["benchmarks"])

    a_name, b_name = a_path.stem, b_path.stem
    print(f"{a_name}: {a['nnsight_version']}")
    print(f"  {a['nnsight_file']}")
    print(f"{b_name}: {b['nnsight_version']}")
    print(f"  {b['nnsight_file']}")
    print()
    print(f"{'benchmark':32} {a_name:>12} {b_name:>12} {'ratio':>8}")
    print(f"{'':32} {'(us)':>12} {'(us)':>12} {'b/a':>8}")
    print("-" * 68)
    labels = list(a_rows) + [k for k in b_rows if k not in a_rows]
    for label in labels:
        av, bv = a_rows.get(label), b_rows.get(label)
        ratio = f"{bv / av:.2f}x" if av and bv else "—"
        av_s = f"{av:.2f}" if av is not None else "—"
        bv_s = f"{bv:.2f}" if bv is not None else "—"
        print(f"{label:32} {av_s:>12} {bv_s:>12} {ratio:>8}")


if __name__ == "__main__":
    main()
