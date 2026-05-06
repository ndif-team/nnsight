---
title: Performance
one_liner: Where NNsight overhead lives, what is cached, and how to keep traces fast.
tags: [internals, dev]
related: [docs/developing/tracing-pipeline.md, docs/developing/backends.md]
sources: [src/nnsight/intervention/tracing/base.py:204, src/nnsight/intervention/backends/base.py:18, src/nnsight/intervention/tracing/globals.py:53, 0.6.0.md]
---

# Performance

## What this covers

Where the per-trace cost goes, what is cached automatically, what config flags trade safety for speed, and the single biggest optimization users get wrong: consolidating multiple traces into one. Most numbers here come from the v0.6 release notes (`0.6.0.md`) and the benchmark in `tests/performance/`.

## Architecture / How it works

### Where overhead lives

A `with model.trace(...)` block has a fixed setup cost on **the first call at a given site**. From the v0.6 release notes (`0.6.0.md` lines 350–356):

| Scenario | v0.5.15 | v0.6.0 | Speedup |
|----------|---------|--------|---------|
| Empty trace (no saves) | 1196 us | 308 us | 3.9x |
| 1 `.save()` | 1370 us | 474 us | 2.9x |
| 12 `.save()`s | 1697 us | 716 us | 2.4x |

(12-layer MLP benchmark, CPU, from `tests/performance/benchmark_interventions.py`.)

After the first call, the per-trace setup drops dramatically because of caching (see below). The marginal per-`.save()` cost is roughly 30–80 us depending on what's being saved.

The overhead is **constant** in model size — it doesn't scale with parameter count. For real models where the forward pass takes seconds, NNsight overhead becomes negligible.

### What `Tracer.capture` does (per trace, first time)

`Tracer.capture` (`src/nnsight/intervention/tracing/base.py:204`) is the bulk of trace setup:

1. Find the user's frame. From `Tracer.__enter__` this is `get_entered_frame()` — walks past any `__enter__` chain (subclass `super().__enter__()` calls, user-defined CM wrappers). For the two direct-capture call sites (`Tensor.backward` patcher, `Envoy.__getattr__` fallback) `capture(frame=None)` falls back to `get_non_nnsight_frame()`, which walks the stack until the next frame's `__name__` is not `nnsight*`.
2. Compute `cache_key = hash((co_filename, start_line, co_name, co_firstlineno))`.
3. Look up `Globals.cache.get(cache_key)` — if present, reuse source, AST node, and filename. Done.
4. On miss: extract source via `inspect.getsourcelines(frame)`, normalize indentation, parse the AST to find the `with` block, compile, and store in cache.

Source extraction touches `linecache`, AST parsing, and column normalization. This is what dominates the 308 us empty-trace cost on the first call.

### What the code-object cache does (per trace, first time)

`Backend.__call__` (`src/nnsight/intervention/backends/base.py:62`) checks `Globals.cache.code_cache` keyed by `(cache_key, type(tracer).__name__)`. On miss it runs `compile(source_code, tracer.info.filename, "exec")` and stores the resulting code object. On a hit it skips compilation entirely.

The tracer-type qualifier matters: the same `with` block can be wrapped differently depending on tracer subclass (`InterleavingTracer` vs `Invoker` vs `ScanningTracer`), and each gets its own compiled code.

PR #652 (the v0.6 caching push) reports a **13.2x speedup** on cache hits at the same trace site.

### What `TracingCache` stores

`TracingCache` (`src/nnsight/intervention/tracing/globals.py:53`):

- `cache: dict[cache_key -> (source_lines, start_line, ast_node, filename)]` — populated by `Tracer.capture` on first hit, drained by `get`.
- `code_cache: dict[(cache_key, tracer_type_name) -> code_object]` — populated by `Backend.__call__` on first compile, drained by `get_code`.

Both are process-wide singletons. `Globals.cache.clear()` resets them. The `TRACE_CACHING` config flag from earlier versions is **deprecated** — caching is now always enabled.

### Per-intervention cost

Per `.output` / `.input` / `.save()` access: roughly 30–80 us. Comes from:

- Mediator state machine ticks (one event per access).
- Hook firing in the worker thread.
- `narrow` / `swap` calls (mostly free with `needs_batching=False`).
- `Globals.saves.add(id(obj))` for `.save()`.

`.save()` via the pymount C extension adds the C-call overhead of dispatching through Python's `object` type. `nnsight.save(obj)` skips the C dispatch.

### Big optimization: consolidate traces

Each trace pays the per-trace setup cost (~0.3 ms) regardless of how much it does inside. **Loop inside one trace, not multiple traces in a loop.**

Bad — 12 traces, ~6 ms total:

```python
for layer in model.transformer.h:
    with model.trace(prompt):
        hiddens.append(layer.output.save())
```

Good — 1 trace, ~0.7 ms:

```python
with model.trace(prompt):
    hiddens = []
    for layer in model.transformer.h:
        hiddens.append(layer.output.save())
```

The second version pays trace setup once and amortizes 12 saves to ~30 us each.

### Config flags that affect performance

`CONFIG.APP.PYMOUNT` — when `True` (default), mounts `Object.save` once at first trace via `Globals.enter` (`src/nnsight/intervention/tracing/globals.py:101`). Setting `False` skips the C extension entirely; you must use `nnsight.save(obj)` instead of `obj.save()`. The performance difference is now negligible because pymount is mounted once and never unmounted (v0.6 change). Earlier versions called `PyType_Modified` on every trace enter/exit, which invalidated all CPython type caches and was expensive.

`CONFIG.APP.CROSS_INVOKER` — when `True` (default), variables defined in one invoke are propagated to subsequent invokes via `PyFrame_LocalsToFast`. v0.6 batched this into a single call per invoke transition (instead of one per variable), so the cost is small but nonzero. Setting `False` saves a small amount of time when invokes don't need to share state.

(`CONFIG.APP.CROSS_INVOKER` is set to `False` automatically inside `VLLM` because vLLM uses worker-side global grafting instead — see `src/nnsight/modeling/vllm/vllm.py:36`.)

### v0.6 specific optimizations (beyond caching)

From `0.6.0.md` lines 359–365:

- **Persistent pymount.** `.save()` and `.stop()` mounted once at import. Earlier versions mounted/unmounted per trace, triggering `PyType_Modified` and invalidating type caches.
- **Removed `torch._dynamo.disable` wrappers.** Hook functions previously had `@torch._dynamo.disable`, which added a `set_eval_frame` C call per forward. The hooks use closures and thread synchronization that dynamo can't compile anyway, so removal is safe.
- **Batched `PyFrame_LocalsToFast`.** Cross-invoker variable sharing now uses one batched call instead of one-per-variable.
- **Filtered globals copy.** When starting a worker thread, NNsight previously copied the entire `__globals__` dict. v0.6 only copies names actually referenced by the intervention's bytecode (`co_names`), reducing copy overhead.

### When you're benchmarking

- Always **warm up** at least 2–3 iterations before measuring. The first trace at a call site pays the full cost (capture + compile); subsequent calls hit the cache.
- Define trace-using functions at **module level**, not inside loops or `-c` strings. NNsight extracts source via `inspect.getsourcelines`, which needs the function's source to be readable from a real file.
- Closures in benchmark loops. `def fn(): with model.trace(prompt): ...` defined inside `for prompt in prompts:` will close over the **last** value of `prompt`. Bind it explicitly: `def fn(p=prompt): with model.trace(p): ...`.
- For GPU timing, `torch.cuda.synchronize()` before and after each measurement.

## Key files / classes

- `src/nnsight/intervention/tracing/base.py:204` — `Tracer.capture` (source extraction + cache lookup)
- `src/nnsight/intervention/backends/base.py:62` — code-object cache key + lookup
- `src/nnsight/intervention/tracing/globals.py:53` — `TracingCache`
- `src/nnsight/intervention/tracing/globals.py:101` — `Globals.enter` (pymount lifecycle)
- `src/nnsight/intervention/interleaver.py` — mediator-loop overhead lives here
- `src/nnsight/intervention/envoy.py` — `eproperty` access (per-`.output` cost)
- `tests/performance/benchmark_interventions.py` — the canonical benchmark
- `tests/performance/README.md` — benchmark documentation
- `0.6.0.md` lines 345–376 — v0.6 performance numbers and rationale
- PR #652 — 13.2x cache speedup write-up (referenced in 0.6.0.md, sections on caching)

## Extension points

- **Profiling a custom backend or runtime.** Wrap the relevant section in `time.perf_counter()` and call from a script that runs the same trace 10–20 times to amortize. Use `cProfile` for function-level breakdown.
- **Adding a new cache layer.** `TracingCache.add` / `add_code` are public. If you have a new tracer subclass with its own expensive setup, follow the same pattern.
- **Reducing intervention cost.** If an intervention does heavy work (e.g. SAE encode), consider pre-computing inside `model.edit()` so the work happens once at edit time and not per trace.

## Related

- [tracing-pipeline.md](./tracing-pipeline.md) — what `Tracer.capture` produces
- [backends.md](./backends.md) — the code-object cache lives in the base backend
- [testing.md](./testing.md) — how to run the benchmark suite
