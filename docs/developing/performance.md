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
PYTHONPATH=/path/to/old/nnsight/src   python interleave_bench.py --tag old
PYTHONPATH=/path/to/ndif2/nnsight/src python interleave_bench.py --tag new
python compare.py results/old.json results/new.json    # ratio new/old; >1 = slower
```

Each run writes `results/<tag>.json` and prints a summary. Every measurement is a
median over many repeats after warmup, in microseconds, with the min alongside (the
least scheduling-polluted floor).

### What each benchmark isolates

| Benchmark | Isolates |
|---|---|
| `baseline_forward` | the raw module forward, untraced — the floor every trace sits above |
| `capture_warm` | an empty trace once its block is captured: interleave setup + teardown |
| `capture_cold` | the same with the capture cache cleared each time — **cold minus warm is what parsing + compiling the block costs** (paid once per site per process) |
| `invoke_scaling[k]` | one input invoke + `k-1` empty ones; the slope is the **per-invoke** cost |
| `intervention_scaling[n]` | a trace saving `n` layer outputs; the slope is the **per-intervention** cost |
| `read_output_warm` vs `read_source_warm` | reading a module's output vs one op inside its forward — the gap is steady-state `.source` overhead |
| `source_first_access` | the first `.source` on a fresh model — **compiling the instrumented forward**, a one-time cost |

The absolute numbers include the constant `baseline_forward`, so the signal is in
the **deltas**: cold − warm is capture; the slope of a scaling sweep is the marginal
cost; source − output is instrumentation.

### A sample run (this repo, CPU, illustrative only)

```
single-shot (median of many):
  baseline forward                 500 us
  capture (warm)                    36 us      # empty trace over a no-op, cache warm
  capture (cold)                    97 us      # +61 us to parse + compile the block
  read a module output            1433 us
  read a source op (torch_relu_0) 1448 us      # +~15 us for .source steady state
  first source access             4899 us      # one-time: compiling the instrumented forward

intervention scaling (median us):  0->1408, 1->1430, 8->1521, 32->1854
invoke scaling (median us):        1->1434, 8->2041, 32->4034
```

Numbers are machine- and load-dependent; µs-scale rows carry real run-to-run
variance. Treat a ratio near 1.0 as "the same" and re-run before trusting a small
difference. Do not cite these figures as authoritative — regenerate them.

## Where overhead lives

- **Block capture (once per site per process).** `Tracer` reads the `with` block's
  own source and compiles it. The compiled `(source, node)` is memoized keyed on the
  block's code location in `BLOCKS` (`src/nnsight/tracing/globals.py`); a warm site
  skips the parse/compile entirely. This is the cold − warm gap.
- **Interleaver setup/teardown (per trace).** Building the `Interleaver`, scoping
  each invoke, starting/parking the intervention greenlets (`Mediator`s), and tearing
  them down. This dominates the warm empty-trace cost.
- **Per invoke.** Each `tracer.invoke(...)` builds a worker greenlet, captures its
  block, and registers it — the slope of `invoke_scaling`.
- **Per intervention.** Each `.output`/`.save()` parks a worker and serves it one
  value through `handle` — the slope of `intervention_scaling` (tens of µs each).
- **`.source`.** First access compiles the instrumented forward (`source_first_access`,
  a one-time cost cached in `_FORWARD_CACHE`, `source.py`); steady-state reads add a
  small constant over reading the module's output.

## The biggest win: consolidate traces

Each trace pays the setup cost regardless of how much it does inside. **Loop inside
one trace, not multiple traces in a loop.**

```python
import nnsight

# Bad — one trace per layer, pays setup N times
hiddens = []
for layer in model.transformer.h:
    with model.trace(prompt):
        h = layer.output.save()
    hiddens.append(h)               # append after the trace, one saved value each

# Good — one trace, setup paid once
with model.trace(prompt):
    hiddens = nnsight.save([layer.output for layer in model.transformer.h])
```

## When you benchmark

- **Warm up** 2-3 iterations before measuring — the first trace at a site pays the
  full capture + compile.
- **Define trace-using functions at module level in a real file.** Block capture
  reads the block's source via `inspect`, so it does not work from `python -c "..."`
  or a heredoc. (`interleave_bench.py` defines its toy modules at file scope for
  exactly this reason.)
- **Bind loop variables.** `def fn(p=prompt): with model.trace(p): ...` — a closure
  over `prompt` captures the last value.
- **GPU:** `torch.cuda.synchronize()` before and after each measurement.

## Key files

- `tests/performance/interleave_bench.py` — the benchmark (self-contained)
- `tests/performance/compare.py` — side-by-side ratio of two runs
- `tests/performance/README.md` — harness documentation
- `src/nnsight/tracing/globals.py` — `BLOCKS`, the block capture cache
- `src/nnsight/intervention/source.py` — `_FORWARD_CACHE`, the instrumented-forward cache

## Related

- [tracing-pipeline.md](./tracing-pipeline.md) — what block capture produces
- [interleaver-internals.md](./interleaver-internals.md) — mediator/greenlet overhead
- [testing.md](./testing.md) — running the suite
