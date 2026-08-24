---
title: Deferred Execution
one_liner: A trace block is captured by source, compiled into a standalone function, and run interleaved with the model — parking in a greenlet on each Envoy property access until the model produces the value via a PyTorch hook.
tags: [concept, mental-model, tracing]
related: [docs/concepts/threading-and-mediators.md, docs/concepts/interleaver-and-controller.md, docs/concepts/envoy.md]
sources: [src/nnsight/tracing/tracer.py:214, src/nnsight/tracing/tracer.py:270, src/nnsight/tracing/tracer.py:343, src/nnsight/intervention/tracer.py:223, src/nnsight/intervention/envoy.py:612]
---

# Deferred Execution

## What this is for

The body of `with model.trace(...)` does not run inline. nnsight reads the with-block's source, parses it, compiles the body into a standalone function, and runs *that* interleaved with the model's forward pass. When the body accesses an `Envoy` property like `.output`, it parks (a greenlet switch) until the model reaches that module and a PyTorch hook hands the value over. This is the foundation everything else builds on.

## When to use / when not to use

- This is the only way to read or edit intermediate activations. There is no inline alternative.
- If you want the *real* output tensors without interception, call the model directly outside any trace.
- If you only want activation shapes without running the model for real, use `model.scan(...)` — it runs the forward under `FakeTensorMode` (see [ScanningTracer](#variations)).

## Canonical pattern

```python
import torch
from nnsight import NNsight

net = torch.nn.Sequential(torch.nn.Linear(5, 10), torch.nn.Linear(10, 2))
model = NNsight(net)

with model.trace(torch.rand(1, 5)):
    # This block is captured, compiled, and run interleaved with the forward.
    # Accessing .output parks the worker until layer 0 produces its output.
    hidden = model[0].output.save()

print(hidden.shape)   # torch.Size([1, 10])  — a real tensor after the block
```

Verified output:

```
hidden.shape: (1, 10)
type: Tensor
```

## Step-by-step: one trace call

The plumbing lives in the base `Tracer` (`tracing/tracer.py:214`); `InterleavingTracer` (`intervention/tracer.py:48`) only overrides how the captured body is run.

1. `model.trace(x)` constructs an `InterleavingTracer`, remembering `fn` (`"__call__"` by default) and the forward args. Nothing runs yet.
2. `__enter__` (`tracing/tracer.py:343`) calls `capture()`:
   - `capture()` looks two frames up to the user's frame, reads the file's source (`Tracer.source`, cached in `SOURCES`), and finds the `with` AST node at `frame.f_lineno`.
   - It then runs `build()` (wrap the block *body* in a module) and `compile()` (compile to a code object under the original filename/line so tracebacks read right).
   - The `(node, compiled)` pair is memoized per call site in `BLOCKS`, keyed by `(filename, lineno, co_name, co_firstlineno)`. Re-entering the same `with` (e.g. a loop) parses and compiles only once.
3. Still in `__enter__`, if the block has real code (`skippable`), `skip_context` arms a per-frame trace hook that raises `ExitTracingException` the instant the body would run — so the body never executes inline.
4. Python unwinds the `with`. `__exit__` (`tracing/tracer.py:449`) catches `ExitTracingException`, increments the trace-depth, and calls `self.backend(self)` — which calls `tracer.execute(code)`.
5. `InterleavingTracer.execute` (`intervention/tracer.py:223`) sets up a `Batcher`, then:
   - **direct input** (`trace(x)`): builds one `Mediator` for the whole block and gives it the input's batch group.
   - **invoke mode** (`trace()`): execs the body now to collect `tracer.invoke(...)` sub-blocks, each of which registers its own `Mediator` (see [Batching and Invokers](batching-and-invokers.md)).
6. `Envoy.interleave` (`envoy.py:612`) enters the shared `Interleaver` context. `Interleaver.__enter__` `start()`s every mediator's greenlet worker, running each up to its first park. Then `fn(*combined_input)` runs the model on the main greenlet.
7. As the model reaches each module, its hook calls `Interleaver.handle(location, value)`, which offers the value to any worker parked on that location — serving reads, applying swaps — and returns the (possibly edited) value back into the forward.
8. When the model returns, `interleave` calls `handle("result", result)` so anything parked on `tracer.result` is served, then `check_dangling_mediators()` surfaces any worker still parked on a location the model never reached.
9. Back in `execute`, `push_result` writes each worker's saved values back into the user's frame; `interleave`/`execute` `cancel()` the interleaver so the next run starts clean.

## Saving: what survives the block

By default nothing a trace computes leaves it. `save(value)` (`tracing/tracer.py:161`, exposed as `nnsight.save` and `.save()`) marks a value **by identity** to survive past the *outermost* trace.

- Inner (nested) traces push *all* their locals up to the enclosing block, so values flow freely between nested traces.
- The outermost trace filters to just the saved values (`push_result`). Depth is tracked around the backend call.
- `save()` **raises if called outside a trace** — it is not a silent no-op:

```
save() was called outside a trace. `.save()` / nnsight.save(x) marks a value to
return from the enclosing `with model.trace(...):` block, so it only works inside
one — move the save into the trace block.
```

Save the value you bind: `h = model.layer1.output.save()`. A value built from a saved one is not itself saved — `(x.save() * 2)` returns `x`, not the product; write `(x * 2).save()`.

## Caching (compile cache)

nnsight memoizes per call site:

- Source text of a file (`SOURCES`, read once, never re-validated).
- The parsed AST node + compiled code object (`BLOCKS`, keyed by call-site identity). A site that is *not* a `with` block caches `(None, None)`, so the `traceable` fallback doesn't re-parse every call.

Repeating the same trace in a loop pays the parse+compile cost only on the first iteration.

## Variations

### Trace with implicit invoke (direct input)

Passing input to `trace(...)` makes the whole block one implicit invoke over that input.

```python
with model.trace("Hello"):
    out = model.lm_head.output.save()
```

### Trace with explicit invokes

`trace()` with no input runs the block to collect `tracer.invoke(...)` sub-blocks; each becomes its own worker on its own batch slice.

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        out = model.lm_head.output.save()
```

### Scan (shapes only)

`model.scan(x)` uses `ScanningTracer` (`intervention/tracer.py:299`), which runs the same interleave under `FakeTensorMode`: operations propagate only shape/dtype/device, no real compute, no dispatch. Read `.shape`/`.dtype` inside the block; a fake tensor is invalid once the scan exits.

## Gotchas

- **The body never runs in your file's namespace directly.** It runs as a compiled function against a `Scope` (a snapshot of your locals + the live frame + globals; see `tracing/util.py:32`). Assignments flow back via `push_result`, filtered to saved values at the outermost trace.
- **The body must start on its own line.** A one-liner like `with model.trace(x): y = model.output.save()` is refused — nnsight intercepts the body at the start of a line and can't skip a body written on the `with` line (it would run twice).
- **Source must be reachable.** Capture reads the source via `linecache`. It works for files, IPython/Jupyter cells, and `python -c "..."` programs, but **not** from a heredoc or `python -c` that defines the trace inside an `exec`. Write your snippet to a real `.py` file.
- **The source cache is not re-validated.** A file edited mid-run is traced as first seen; the cache key includes the line number, so normal edits invalidate naturally on reload.

## Related

- [Threading and Mediators](threading-and-mediators.md) — what the greenlet worker does after capture, and the event protocol.
- [Interleaver and Controller](interleaver-and-controller.md) — the PyTorch-side machinery that delivers values.
- Source: `src/nnsight/tracing/tracer.py` (`Tracer`, `save`, `Scope` handling), `src/nnsight/intervention/tracer.py` (`InterleavingTracer`, `ScanningTracer`, `Invoker`), `src/nnsight/intervention/envoy.py` (`Envoy.interleave`).
