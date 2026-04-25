---
title: Deferred Execution
one_liner: A trace block is captured by source, compiled into a function, and run in a worker thread that blocks on Envoy property accesses until the model's forward provides the value via a PyTorch hook.
tags: [concept, mental-model, tracing]
related: [docs/concepts/threading-and-mediators.md, docs/concepts/interleaver-and-hooks.md, docs/concepts/envoy-and-eproperty.md]
sources: [src/nnsight/intervention/tracing/base.py:204, src/nnsight/intervention/tracing/base.py:593, src/nnsight/intervention/tracing/tracer.py:335, src/nnsight/intervention/tracing/invoker.py:41, src/nnsight/intervention/backends/base.py:37]
---

# Deferred Execution

## What this is for

The body of `with model.trace(...)` does not run inline. nnsight extracts the with-block source, parses it via AST, compiles it into a function, and runs that function in a worker thread. When the worker thread accesses an `Envoy` property like `.output`, it blocks on a queue until the model's forward pass provides the value via a PyTorch hook. This is the foundation everything else in nnsight builds on.

## When to use / when not to use

- This is the only way to access intermediate activations in nnsight. There is no inline alternative.
- If your code needs the *real* tensor values inline (and you are willing to forgo nnsight's hook-driven interception), call the model directly — outside any trace context.
- If you only want shape information without running the model, use `model.scan(...)` (see [docs/concepts/envoy-and-eproperty.md](envoy-and-eproperty.md)).

## Canonical pattern

```python
import nnsight

model = nnsight.LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

with model.trace("Hello"):
    # This block is extracted, compiled, and run in a worker thread.
    # Accessing .output blocks the worker until the layer-0 forward hook fires.
    hs = model.transformer.h[0].output[0].save()

# After exiting the with-block, hs is a real torch.Tensor.
print(hs.shape)
```

## Step-by-step: what happens inside one trace call

1. `model.trace("Hello")` constructs an `InterleavingTracer`. No code runs yet — the with-block has not been entered.
2. `__enter__` (in `Tracer.__enter__`, `tracing/base.py:593`) calls `capture()`:
   - `capture()` walks the call stack to find the user's frame, reads the source via `inspect.getsourcelines`, and parses the AST to locate the with-block.
   - The captured source, frame, and AST node are stored on `self.info` (`Tracer.Info`).
   - A cache is consulted at `Globals.cache.get(cache_key)` keyed by `(filename, line, function, first_line)`. On cache hit the parse step is skipped.
3. Still inside `__enter__`, nnsight installs `sys.settrace(skip_traced_code)`. The trace function fires the moment Python tries to execute the first line of the with-body and immediately raises `ExitTracingException`.
4. Python unwinds the with-block. `__exit__` catches `ExitTracingException` and calls `self.backend(self)` (typically `ExecutionBackend`).
5. The backend (`backends/base.py:37`) runs the four-step compile pipeline:
   - Indent every captured source line by 4 spaces.
   - Call `tracer.compile()` to wrap the body in `def __nnsight_tracer_<id>__(__nnsight_tracing_info__, tracer): ...`. The compiled function pulls variables from the user frame (`tracer.pull()`), runs the captured body, and pushes results back (`tracer.push()` or `mediator.end()`).
   - Compile to a code object (`compile(source, "<nnsight ...>", "exec")`) and cache it in `Globals.cache`.
   - `exec` the code object against `frame.f_globals + frame.f_locals` to produce the callable.
6. `InterleavingTracer.execute()` runs the compiled function once on the *main* thread to register `Mediator`s — one per `tracer.invoke(...)` plus the implicit invoke when arguments are passed to `.trace(...)` (`tracing/tracer.py:351`). Invokers compile a *separate* function via `Invoker.compile` (`tracing/invoker.py:41`) wrapped in try/catch that ends in `mediator.end()` or `mediator.exception(e)`.
7. `Envoy.interleave(...)` enters the `Interleaver` context, which starts each `Mediator`'s worker thread. Each worker runs its compiled intervention function. When the worker accesses `model.transformer.h[0].output`, the `eproperty` descriptor calls `interleaver.current.request(requester)`, which puts a VALUE event on the queue and blocks.
8. The model's forward pass runs on the main thread. A one-shot PyTorch forward hook (registered lazily by `requires_output`) sees the layer-0 output, calls `mediator.handle(...)`, delivers the value to the queue, and the worker thread unblocks.
9. When the worker function returns, the wrapper hits `mediator.end()` which sends an END event. The `Interleaver.__exit__` reaps the worker. `Globals.saves` controls which user-side variables are pushed back into the original frame.

## Caching

nnsight caches three things keyed on call site identity:

- Source lines + parsed AST (`Globals.cache.cache`).
- Compiled code objects (`Globals.cache.code_cache`), keyed by `(cache_key, type(tracer).__name__)` so the same call site cached separately for `InterleavingTracer` vs `Invoker` vs `IteratorTracer`.

Repeating the same trace in a loop only pays the AST + compile cost on the first iteration.

## Variations

### Trace with implicit invoke

When you pass input directly to `.trace(...)`, an implicit `Invoker` is created (`tracing/tracer.py:357`). The block becomes the body of that invoker.

```python
with model.trace("Hello"):
    out = model.lm_head.output.save()
```

### Trace with explicit invokes

When you use explicit `tracer.invoke(...)`, the outer `with model.trace()` only registers mediators; the inner `with tracer.invoke(...)` blocks each become a `Mediator` worker.

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        out = model.lm_head.output.save()
```

## Gotchas

- The body of a trace **never runs in your file's namespace directly**. It runs as a freshly compiled function. Variables flow in via `pull()` and out via `push()`. The pymount `.save()` mechanism is what marks values to survive the push filter.
- `sys.settrace` is global. nnsight installs it for the duration of one with-enter; if another tool (debugger, profiler) installs its own trace function inside the with-block, behavior is undefined.
- Source extraction uses `inspect.getsourcelines`. It only works when the source is reachable — file on disk, IPython cell, `<nnsight-console>`, or a `python -c` command. Functions defined inside `exec` or `eval` cannot be traced.
- The cache key includes the line number. Editing the source file invalidates the cache automatically only because `linecache.checkcache` is *suppressed* during capture; the cache key changes naturally on reload.

## Related

- [Threading and Mediators](threading-and-mediators.md) — what the worker thread does after capture.
- [Interleaver and Hooks](interleaver-and-hooks.md) — the PyTorch-side machinery that delivers values.
- Source: `src/nnsight/intervention/tracing/base.py` (`Tracer`), `src/nnsight/intervention/tracing/tracer.py` (`InterleavingTracer`), `src/nnsight/intervention/tracing/invoker.py` (`Invoker`), `src/nnsight/intervention/backends/base.py` (`Backend.__call__`), `src/nnsight/intervention/tracing/globals.py` (`Globals`, `TracingCache`).
