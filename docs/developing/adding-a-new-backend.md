---
title: Adding a New Backend
one_liner: Recipe for subclassing Backend to change what running a captured trace does.
tags: [internals, dev]
related: [docs/developing/backends.md, docs/developing/tracing-pipeline.md]
sources: [src/nnsight/tracing/backend.py, src/nnsight/tracing/tracer.py, src/nnsight/intervention/backends/remote.py, src/nnsight/intervention/backends/local.py]
---

# Adding a New Backend

## What this covers

A backend decides what happens to a captured trace on `__exit__`. The default runs
it locally; `RemoteBackend` ships it to NDIF; `LocalSimulationBackend` round-trips
it through serialization to validate. This page is the recipe for your own.

If you need a different *runtime* (a new inference engine), you usually want a new
model class with a custom `_batch_size`/`_batch`/`interleave` instead — see
`docs/developing/adding-a-new-runtime.md`. Use a custom backend when you want to
change **what is done with the captured block**, not what model it runs against.

## The contract

`Backend` (`src/nnsight/tracing/backend.py:9`) is a callable taking the tracer:

```python
class Backend:
    def __call__(self, tracer):
        tracer.execute(tracer.info.code)
```

When your `__call__` runs, the block is **already captured and compiled**:

- `tracer.info.code` — the compiled block body (a `CodeType`).
- `tracer.info.frame` — the caller's live frame (its globals/locals, and its
  `co_filename`/lineno for tracebacks).
- `tracer.node` — the block's AST node (or `None` after deserialization).

You decide whether to run it (`tracer.execute(code)`), transform it, ship it, or
store it. `tracer.execute` for an `InterleavingTracer` is what sets up the
interleaver and runs the model; the base `Tracer.execute` just `exec`s the body and
pushes results back.

There is no compile step to inherit and no `Globals.enter/exit` to balance — those
concepts from older nnsight don't exist here. Trace-scope depth and traceback
cleanup are handled by `Tracer.__exit__` (`tracing/tracer.py:449`) around your
backend call, so you don't manage them.

## Minimal recipe: run locally, but do something first

```python
from nnsight.tracing.backend import Backend


class LoggingBackend(Backend):
    """Log every traced block's source, then run it locally."""

    def __init__(self, log_path="trace.log"):
        self.log_path = log_path

    def __call__(self, tracer):
        # The block is already compiled; log where it came from.
        code = tracer.info.code
        with open(self.log_path, "a") as f:
            f.write(f"{code.co_filename}:{code.co_firstlineno} {code.co_name}\n")

        # Run it locally, exactly as the default backend would.
        tracer.execute(code)
```

Use it like any backend:

```python
import nnsight
model = nnsight.TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("Hello", backend=LoggingBackend("/tmp/nn.log")):
    out = model.transformer.h[0].output.save()
```

To log the block's *source* text (not just its location), read it from the source
cache the tracer already populated: `nnsight.tracing.globals.SOURCES[code.co_filename]`.

## Skipping local execution (like RemoteBackend)

Don't call `tracer.execute`. Serialize the tracer and ship it however you like; the
compiled code and captured scope travel via the source-reduced interventions payload
(see `docs/developing/serialization.md`). The reference is
`RemoteBackend.request` (`src/nnsight/intervention/backends/remote.py:304`):

```python
class MyRemoteBackend(Backend):
    def __init__(self, model_key):
        self.model_key = model_key

    def __call__(self, tracer):
        from nnsight.schema.request import RequestModel
        from nnsight.ndif import pull_env

        pull_env()                                  # register local modules by-value
        blob = RequestModel.serialize(tracer, compress=False)
        result = my_transport.send(self.model_key, blob)   # your wire
        # push saved values back into the caller's frame, like the local path
        from nnsight.tracing.tracer import save
        from nnsight.tracing.util import push
        for value in result.values():
            save(value)
        push(tracer.info.frame, result)
```

The easiest route to a remote-style backend is to **subclass `RemoteBackend`** and
override just the transport (`send`/`request`/`poll`), reusing its serialize,
status-display, and download/decompress helpers.

## Serialize-and-run locally (like LocalSimulationBackend)

If you want to validate the serialization path without a server, don't write it
from scratch — use `LocalSimulationBackend`
(`src/nnsight/intervention/backends/local.py:37`) via `model.trace(...,
remote="local")`. To model a tighter or different server environment, subclass it
and adjust `_SERVER_MODULES` / `_hide_local_modules`.

## Async backends

Follow `AsyncRemoteBackend` (`remote.py:339`). The pattern is a **dual-call**
object:

1. `__call__(tracer)` runs at `__exit__` time: it fires the request synchronously
   and returns `self` (an awaitable/async-iterable) instead of blocking.
2. The user awaits it (`await backend`) or async-iterates it (`async for update in
   backend`) *after* the `with` block. Implement `__await__`/`__aiter__` to consume
   the status stream and return the saved values.

Because the trace has exited by the time the result lands, an async backend returns
its saves from the await / iterator rather than pushing them into a frame.

## Wiring it up

- Pass it explicitly: `model.trace(..., backend=MyBackend())`.
- Or, for a remote-style backend keyed off `remote=`, add a branch in your model's
  `Remotable._remote_backend` (`src/nnsight/modeling/mixins/remotable.py:52`).

## Key files / classes

- `src/nnsight/tracing/backend.py:9` — `Backend`. The one-method contract.
- `src/nnsight/tracing/tracer.py:431` — `Tracer.execute`. What `tracer.execute` does by default.
- `src/nnsight/tracing/tracer.py:449` — `Tracer.__exit__`. Calls your backend; owns depth + traceback cleanup.
- `src/nnsight/intervention/tracer.py:223` — `InterleavingTracer.execute`. The local run you'd usually delegate to.
- `src/nnsight/intervention/backends/remote.py:39` — `RemoteBackend`. Subclass this for remote transports.
- `src/nnsight/intervention/backends/local.py:37` — `LocalSimulationBackend`. Serialize-and-run example.

## Lifecycle of your backend (one trace)

1. User passes `backend=YourBackend()` to `model.trace(...)`.
2. `Tracer.__init__` stores it on `self.backend`.
3. `__enter__` captures + compiles the block; the body never runs inline
   (`ExitTracingException`).
4. `__exit__` calls `self.backend(self)` — your `__call__(tracer)`.
5. You execute, ship, or store `tracer.info.code`.
6. `__exit__` filters/pushes saved values (for local paths) and cleans the
   traceback on any error.

## Related

- `docs/developing/backends.md` — full reference for the existing backends.
- `docs/developing/tracing-pipeline.md` — what tracer state your backend sees.
- `docs/developing/serialization.md` — if your backend ships code over a wire.
