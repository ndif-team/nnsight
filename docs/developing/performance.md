---
title: Performance
one_liner: Where the interleaving pipeline's overhead lives, what is cached, and how to measure it.
tags: [internals, dev]
related: [docs/developing/tracing-pipeline.md, docs/developing/interleaver-internals.md, docs/developing/testing.md]
sources: [tests/performance/interleave_bench.py, tests/performance/compare.py, tests/performance/README.md, src/nnsight/tracing/globals.py, src/nnsight/intervention/source.py]
---

# Performance

## What this covers

What a `with model.trace(...):` block costs *around* the forward pass — capturing
the block, standing up the interleaver, scoping invokes, serving interventions,
instrumenting a forward for `.source` — and how to measure it. The pipeline cost is
**constant in model size**: it doesn't scale with parameter count, so for real
models where the forward takes milliseconds-to-seconds it is negligible. It only
matters for tight loops over tiny models. The benchmark harness under
`tests/performance/` is the source of truth; there are no fixed numbers to quote
here because they're machine- and version-dependent — take **ratios**, not
absolutes.

## The benchmark harness

`tests/performance/interleave_bench.py` wraps a tiny stack of small linear layers
with `nnsight.NNsight`, so a forward is a handful of negligible matmuls and what's
left in the timings is the pipeline. It needs only `nnsight` and `torch` (no
HuggingFace model), so the same file runs under any nnsight tree — that's the point:
you diff two trees.

```bash
cd tests/performance
PYTHONPATH=/path/to/old/nnsight/src python interleave_bench.py --tag old
PYTHONPATH=/path/to/new/nnsight/src python interleave_bench.py --tag new
python compare.py results/old.json results/new.json    # ratio new/old; >1 = slower
```

Each run writes `results/<tag>.json` and prints a summary. Every measurement is a
median over many repeats after warmup, in microseconds, with the min alongside (the
least scheduling-polluted floor).

## Where overhead lives, and the benchmark that isolates it

The absolute numbers include the constant `baseline_forward` (the raw module
forward, untraced), so the signal is in the **deltas**.

| Cost | Benchmark | What it is |
|---|---|---|
| Block capture, once per site per process | `capture_cold` − `capture_warm` | `Tracer` reads the `with` block's source and compiles it; the result is memoized by code location in `BLOCKS` (`tracing/globals.py`), so a warm site skips it |
| Interleaver setup/teardown, per trace | `capture_warm` | an empty trace: building the run, starting and parking the workers, tearing them down |
| Per invoke | slope of `invoke_scaling[k]` (one input invoke + `k-1` empty ones) | each `tracer.invoke(...)` builds a worker greenlet and registers it |
| Per intervention | slope of `intervention_scaling[n]` (saving `n` layer outputs) | each `.output`/`.save()` parks a worker and serves it one value through `handle` |
| `.source`, first access | `source_first_access` | compiling the instrumented forward, cached in `FORWARD_CACHE` (`source.py`); milliseconds against everything else's microseconds |
| `.source`, steady state | `read_source_warm` − `read_output_warm` | reading one op vs the module's output; not separable at this scale |

Rows at µs scale carry real run-to-run variance. Treat a ratio near 1.0 as "the
same" and re-run before trusting a small difference.

Each trace pays the setup cost regardless of what it does inside, so the biggest
user-side win is looping inside one trace rather than tracing in a loop — see
[invoke-and-batching.md](../usage/invoke-and-batching.md#comparing-prompts-one-trace-not-a-loop-of-traces).

## When you benchmark

- **Warm up** 2-3 iterations before measuring — the first trace at a site pays the
  full capture + compile.
- **Run from a real file.** Block capture reads the block's source
  ([tracing-pipeline.md](./tracing-pipeline.md#source-extraction)).
- **Bind loop variables.** `def fn(p=prompt): with model.trace(p): ...` — a closure
  over `prompt` captures the last value.
- **GPU:** `torch.cuda.synchronize()` before and after each measurement.

## Related

- `tests/performance/README.md` — harness documentation.
