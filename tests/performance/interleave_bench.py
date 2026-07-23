"""Micro-benchmarks for the nnsight interleaving pipeline.

What does a ``with model.trace(...):`` cost *before* the model does any real work?
This measures the pipeline around the forward pass — capturing the block,
standing up the interleaver, scoping each invoke, serving each intervention,
instrumenting a forward for ``.source`` — in isolation from model compute.

The isolation comes from the model: a tiny stack of small linear layers wrapped
with :class:`nnsight.NNsight`, so a forward is a handful of negligible matmuls and
what's left in the timings is the pipeline. Because it only needs ``nnsight`` and
``torch`` — no HuggingFace model — the same file runs under any nnsight: point
``PYTHONPATH`` at the tree you want, write the JSON, and diff two of them with
``compare.py``.

    PYTHONPATH=/path/to/nnsight/src python interleave_bench.py --tag old
    PYTHONPATH=/path/to/ndif2/nnsight/src python interleave_bench.py --tag new
    python compare.py results/old.json results/new.json

The toy module is defined at file scope on purpose: ``.source`` reads a module's
forward with ``inspect.getsourcelines``, which needs real source on disk.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

import nnsight

# The model is a fixed size; the benchmarks vary how much of the *pipeline* runs
# over it, not how big it is. Small enough that a forward doesn't show up.
N_LAYERS = 32
WIDTH = 16
torch.manual_seed(0)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(WIDTH, WIDTH)
        self.fc2 = nn.Linear(WIDTH, WIDTH)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Block() for _ in range(N_LAYERS)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class NoOp(nn.Module):
    """Returns its input, with no submodules. Capture cost is independent of the
    model, so a trace over this is essentially just the pipeline — the right thing
    to time capture against, where a real forward's variance would drown it out."""

    def forward(self, x):
        return x


# --- measurement -----------------------------------------------------------


def bench(fn: Callable[[], Any], warmup: int = 5, repeats: int = 100) -> dict:
    """Run ``fn`` ``repeats`` times after ``warmup``; return timings in microseconds.

    Reports the median (the headline — robust to a stray slow run), the min (the
    floor, least polluted by scheduling), and the mean.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e6)
    return {
        "us_min": round(min(samples), 2),
        "us_median": round(statistics.median(samples), 2),
        "us_mean": round(statistics.mean(samples), 2),
        "repeats": repeats,
    }


def clear_capture_cache() -> bool:
    """Drop the memoized (parse, compile) of every trace site, so the next trace
    of a site captures cold. Returns whether a known cache was found."""
    try:  # this tree
        from nnsight.tracing.globals import BLOCKS

        BLOCKS.clear()
        return True
    except Exception:  # noqa: BLE001 - older layouts keep it elsewhere
        pass
    try:  # older nnsight
        from nnsight.intervention.tracing.globals import Globals

        Globals.cache.clear()
        return True
    except Exception:  # noqa: BLE001
        return False


def find_source_op(model: Any) -> str | None:
    """Name an operation inside a Block's forward, so a source read can target it.

    The op naming has been stable across versions (the called function's dotted
    path joined with ``_`` plus an occurrence index), but try a few and confirm
    one actually reads rather than hardcoding."""
    x = torch.randn(2, WIDTH)
    for name in ("torch_relu_0", "self_fc1_0", "self_fc2_0", "relu_0", "fc1_0"):
        try:
            with model.trace(x):
                getattr(model.layers[0].source, name).output.save()
            return name
        except Exception:  # noqa: BLE001 - wrong name for this version/model
            continue
    return None


# --- benchmarks ------------------------------------------------------------


def run() -> dict:
    x = torch.randn(2, WIDTH)
    results: dict = {}

    # A forward with no nnsight at all — the floor every trace timing sits above.
    raw = Net()
    results["baseline_forward"] = bench(lambda: raw(x))

    # The model the scaling benchmarks trace: real layers, so invokes/interventions
    # /source have something to bind to. Its forward is constant across each sweep,
    # so it cancels in the slope.
    model = nnsight.NNsight(Net())

    # Capture, isolated: trace a no-op model, so the timing is the pipeline with no
    # real forward to add variance. Warm reuses the memoized (parse, compile); cold
    # clears it each time, so the gap is what capturing the block costs — parsing
    # the source and compiling it, paid once per site per process.
    noop = nnsight.NNsight(NoOp())

    def capture_warm():
        with noop.trace(x):
            pass

    results["capture_warm"] = bench(capture_warm)

    can_clear = clear_capture_cache()

    def capture_cold():
        clear_capture_cache()
        with noop.trace(x):
            pass

    results["capture_cold"] = bench(capture_cold) if can_clear else None
    results["capture_cache_clearable"] = can_clear

    # Scale with invokes: one real input invoke, then k-1 empty ones (each sees the
    # whole batch, contributes no rows). Empty invokes skip input batching — whose
    # extension points differ across versions — so this isolates the per-invoke
    # pipeline cost itself: a worker built, its block captured, registered on the
    # interleaver. The forward runs once, on the single input.
    invoke = {}
    for k in (1, 2, 4, 8, 16, 32):

        def run_invokes(k=k):
            with model.trace() as tracer:
                with tracer.invoke(x):
                    pass
                for _ in range(k - 1):
                    with tracer.invoke():
                        pass

        invoke[k] = bench(run_invokes)
    results["invoke_scaling"] = invoke

    # Scale with interventions: one trace, n saved layer outputs. The slope is the
    # per-intervention cost of parking a worker and serving it a value.
    intervention = {}
    for n in (0, 1, 2, 4, 8, 16, 32):

        def run_interventions(n=n):
            with model.trace(x):
                for i in range(n):
                    model.layers[i].output.save()

        intervention[n] = bench(run_interventions)
    results["intervention_scaling"] = intervention

    # Source: reading one operation inside a forward vs reading the module's output.
    # The gap is what instrumenting the forward's internals costs, in steady state.
    op = find_source_op(model)
    results["source_op"] = op
    if op is not None:

        def read_output():
            with model.trace(x):
                model.layers[0].output.save()

        def read_source():
            with model.trace(x):
                getattr(model.layers[0].source, op).output.save()

        results["read_output_warm"] = bench(read_output)
        results["read_source_warm"] = bench(read_source)

        # First source access on a fresh model compiles the instrumented forward —
        # a one-time cost, so build a new model each time and time that first read.
        def source_first():
            fresh = nnsight.NNsight(Net())
            with fresh.trace(x):
                getattr(fresh.layers[0].source, op).output.save()

        results["source_first_access"] = bench(source_first, warmup=2, repeats=20)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="run", help="names the results/<tag>.json file")
    args = parser.parse_args()

    results = {
        "nnsight_file": nnsight.__file__,
        "nnsight_version": getattr(nnsight, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "config": {"n_layers": N_LAYERS, "width": WIDTH},
        "benchmarks": run(),
    }

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.tag}.json"
    out_path.write_text(json.dumps(results, indent=2))

    print(f"nnsight: {results['nnsight_file']}")
    print(f"version: {results['nnsight_version']}")
    print(f"wrote:   {out_path}")
    print()
    _print_summary(results["benchmarks"])


def _print_summary(b: dict) -> None:
    def line(label, stats):
        print(f"  {label:28} {stats['us_median']:>10.2f} us  (min {stats['us_min']:.2f})")

    print("single-shot (median of many):")
    line("baseline forward", b["baseline_forward"])
    line("capture (warm)", b["capture_warm"])
    if b.get("capture_cold"):
        line("capture (cold)", b["capture_cold"])
    if b.get("read_output_warm"):
        line("read a module output", b["read_output_warm"])
        line(f"read a source op ({b['source_op']})", b["read_source_warm"])
        line("first source access", b["source_first_access"])
    print("\ninvoke scaling (median us):")
    for k, s in b["invoke_scaling"].items():
        print(f"  {k:>3} invokes {s['us_median']:>10.2f}")
    print("\nintervention scaling (median us):")
    for n, s in b["intervention_scaling"].items():
        print(f"  {n:>3} saves   {s['us_median']:>10.2f}")


if __name__ == "__main__":
    main()
