# Interleaving pipeline benchmarks

What a `with model.trace(...):` costs *around* the forward pass — capturing the
block, standing up the interleaver, scoping invokes, serving interventions,
instrumenting a forward for `.source`. Not the model's compute; the pipeline.

## How it isolates the pipeline

`interleave_bench.py` wraps a tiny stack of small linear layers with
`nnsight.NNsight`, so a forward is a handful of negligible matmuls and what's left
in the timings is the pipeline. It needs only `nnsight` and `torch` — no
HuggingFace model — so the same file runs under any nnsight tree.

Every measurement is a median over many repeats after warmup, reported in
microseconds, with the min alongside (the least scheduling-polluted floor).

## Running

Point `PYTHONPATH` at whichever tree you want, then diff two runs:

```bash
cd nnsight/tests/performance
PYTHONPATH=/path/to/nnsight/src            python interleave_bench.py --tag old
PYTHONPATH=/path/to/ndif2/nnsight/src      python interleave_bench.py --tag new
python compare.py results/old.json results/new.json
```

Each run writes `results/<tag>.json` and prints a summary. `compare.py` prints the
two side by side with a ratio (`new / old`; > 1 means new is slower).

## What each benchmark measures

| Benchmark | Isolates |
|---|---|
| `baseline_forward` | the raw module forward, untraced — the floor every trace sits above |
| `capture_warm` | an empty trace once its block is captured: interleave setup + teardown + forward |
| `capture_cold` | the same with the capture cache cleared each time — the gap to warm is what **parsing + compiling the block** costs (paid once per site per process) |
| `invoke_scaling[k]` | one input invoke + `k-1` empty ones; the slope is the **per-invoke** cost (a worker built, its block captured, registered) |
| `intervention_scaling[n]` | a trace saving `n` layer outputs; the slope is the **per-intervention** cost (parking a worker and serving it a value) |
| `read_output_warm` vs `read_source_warm` | reading a module's output vs one operation *inside* its forward — the gap is steady-state `.source` overhead |
| `source_first_access` | the first `.source` on a fresh model — **compiling the instrumented forward**, a one-time cost |

## Reading it

The absolute numbers include the constant `baseline_forward`, so the signal is in
the **deltas**: cold minus warm is capture; the slope of a scaling sweep is the
marginal cost; source minus output is instrumentation. Ratios in `compare.py` are
what to watch across versions.

Numbers are machine- and load-dependent, and µs-scale rows carry real run-to-run
variance — take a ratio near 1.0 as "the same," and re-run before trusting a small
difference. `capture_cache_clearable` in the JSON says whether the cold-capture row
is real for that tree (the cache lives in a known place per version); if a future
layout moves it, that row is omitted rather than silently wrong.
