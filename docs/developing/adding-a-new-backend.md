---
title: Adding a New Backend
one_liner: Recipe for subclassing Backend to customize how compiled traced code is executed.
tags: [internals, dev]
related: [docs/developing/backends.md, docs/developing/tracing-pipeline.md]
sources: [src/nnsight/intervention/backends/base.py:18, src/nnsight/intervention/backends/execution.py:13]
---

# Adding a New Backend

## What this covers

A backend decides what happens to the function NNsight compiled from your traced source. The default `ExecutionBackend` runs it locally; `RemoteBackend` ships it to NDIF; `EditingBackend` stores it for later replay; `LocalSimulationBackend` round-trips it through serialization to validate. This page is the recipe for adding your own.

If you only need a slightly different runtime (e.g. a new inference engine), you usually want a new model class with a custom `_prepare_input` / `_batch` / `interleave` instead — see [adding-a-new-runtime.md](./adding-a-new-runtime.md). Use a custom backend when you want to change **what is done with the compiled function**, not what model it runs against.

## When to use a new backend vs another mechanism

| Goal | Right tool |
|------|------------|
| Run on a new inference engine (vLLM-style) | New model class + custom batcher (see [adding-a-new-runtime.md](./adding-a-new-runtime.md)) |
| Send compiled code to a remote service | New backend (probably extending `RemoteBackend`) |
| Add cross-cutting instrumentation (logging, profiling, audit trail) | New backend |
| Persistent transformations applied to every trace | `EditingBackend` pattern (`src/nnsight/intervention/backends/editing.py:11`) |
| Async streaming wrapper around an existing path | New backend with a dual-call pattern, like `AsyncVLLMBackend` (`src/nnsight/modeling/vllm/async_backend.py:19`) |
| Validate serialization without leaving the box | Use existing `LocalSimulationBackend` (`src/nnsight/intervention/backends/local_simulation.py:39`) |

## Architecture / How it works

### `Backend.__call__(tracer)` contract

The base class (`src/nnsight/intervention/backends/base.py:18`) does the compile work. Calling `super().__call__(tracer)` yields the compiled function. Your subclass decides what to do with it.

The compile pipeline:

1. Indents `tracer.info.source` to live inside a function body.
2. `tracer.compile()` adds a function header and the `pull` / `push` calls.
3. Looks up `Globals.cache.code_cache` keyed by `(tracer.info.cache_key, type(tracer).__name__)`. On miss, `compile(...)` runs and the result is cached.
4. `exec`s the code object against `frame.f_globals + frame.f_locals` and pulls the function out of the resulting namespace.

### Minimal recipe

```python
from nnsight.intervention.backends.base import Backend
from nnsight.intervention.tracing.globals import Globals
from nnsight.intervention.tracing.util import wrap_exception


class LoggingBackend(Backend):
    """Logs every compiled function then runs it locally."""

    def __init__(self, log_path: str = "trace.log"):
        self.log_path = log_path

    def __call__(self, tracer):
        # 1. Compile the traced source.
        fn = super().__call__(tracer)

        # 2. Log the source for audit.
        with open(self.log_path, "a") as f:
            f.write("".join(tracer.info.source))
            f.write("\n# ---\n")

        # 3. Run it. Mirror ExecutionBackend's pattern: enter Globals, execute,
        # wrap exceptions with the tracer's info so the user sees a clean traceback.
        try:
            Globals.enter()
            return tracer.execute(fn)
        except Exception as e:
            raise wrap_exception(e, tracer.info) from None
        finally:
            Globals.exit()
```

Use it like any other backend:

```python
import nnsight
model = nnsight.LanguageModel("openai-community/gpt2", dispatch=True)

with model.trace("Hello", backend=LoggingBackend("/tmp/nn.log")):
    out = model.transformer.h[0].output.save()
```

### What you must call

If you want execution to behave like `ExecutionBackend`:

- Wrap `tracer.execute(fn)` between `Globals.enter()` / `Globals.exit()`. Forgetting this leaves the trace stack imbalanced.
- Wrap caught exceptions with `wrap_exception(e, tracer.info)` so the user's traceback points at their source rather than NNsight internals.

If you need to skip local execution (like `RemoteBackend`):

- Don't call `tracer.execute(fn)`. Instead, package the function or its source for whatever transport you have. `Backend.__call__` already returned a callable; for remote use, you typically serialize `tracer` itself plus the compiled function. See `RemoteBackend.request` (`src/nnsight/intervention/backends/remote.py:514`) for the reference.

### Async backends

If your backend needs to be awaited, follow the `AsyncVLLMBackend` dual-call pattern (`src/nnsight/modeling/vllm/async_backend.py:19`):

- `__call__(tracer)` runs at `__exit__` time, sets up state, may submit a request.
- A second `__aiter__` (or `__call__()` with no args) is invoked by user code via `tracer.backend()` and returns an async generator or awaitable.

The tracer cooperates by setting `tracer.asynchronous = True` when entering via `async with`; in that branch `Tracer.__exit__` (`src/nnsight/intervention/tracing/base.py:691`) returns the result of `self.backend(self)` directly to the user instead of suppressing it.

## Key files / classes

- `src/nnsight/intervention/backends/base.py:18` — `Backend.__call__` (compile step you'll inherit)
- `src/nnsight/intervention/backends/execution.py:13` — minimal example of executing locally
- `src/nnsight/intervention/backends/editing.py:11` — example that does **not** execute, just stores the compiled function as a default mediator
- `src/nnsight/intervention/backends/local_simulation.py:39` — example that wraps the compiled function with serialize-and-deserialize before executing
- `src/nnsight/intervention/tracing/globals.py:101` — `Globals.enter` / `Globals.exit` (call these around execution)
- `src/nnsight/intervention/tracing/util.py` — `wrap_exception` (use to clean up tracebacks)

## Lifecycle of your backend (during one trace)

1. User passes `backend=YourBackend()` to `model.trace(...)`.
2. The tracer stores it on `self.backend`.
3. Source is captured at `__enter__`. The traced code never runs normally — `ExitTracingException` is raised instead.
4. `Tracer.__exit__` calls `self.backend(self)` — your `__call__(tracer)`.
5. Your code typically calls `super().__call__(tracer)` to get the compiled `fn`.
6. You decide what to do: execute, serialize and submit, store on the model, etc.
7. Return value: synchronous backends usually return `None` and rely on `tracer.push` to propagate variables back; async backends return something awaitable.

## Extension points

- **Mix in remote-style transport.** Subclass `RemoteBackend` (`src/nnsight/intervention/backends/remote.py:421`) instead of `Backend` to inherit serialization, status-display, and result-download infrastructure.
- **Persist edits.** Mirror `EditingBackend`: build a `Mediator(fn, info)` and append to `model._default_mediators`. Every subsequent trace will pick it up via `InterleavingTracer.compile` (`src/nnsight/intervention/tracing/tracer.py:344`).
- **Custom error handling.** Override `wrap_exception` behavior or inspect `tracer.info.source` in your except block to add user-facing context.

## Related

- [backends.md](./backends.md) — full reference for the existing backends
- [tracing-pipeline.md](./tracing-pipeline.md) — what tracer state your backend sees
- [serialization.md](./serialization.md) — useful if your backend ships code over a wire
