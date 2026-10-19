---
title: Backends
one_liner: A Backend decides what running a captured trace means — execute locally, ship to NDIF (blocking/non-blocking/async), or serialize-and-run for a local dry run.
tags: [internals, dev]
related: [docs/developing/tracing-pipeline.md, docs/developing/serialization.md]
sources: [src/nnsight/tracing/backend.py, src/nnsight/intervention/backends/remote.py, src/nnsight/intervention/backends/local.py, src/nnsight/intervention/backends/display.py, src/nnsight/modeling/mixins/remotable.py]
---

# Backends

## What this covers

A `Backend` decides what happens to a captured, compiled `with` block on
`__exit__`. This page covers the `Backend` base, the three concrete backends
(`RemoteBackend`, `AsyncRemoteBackend`, `LocalSimulationBackend`), how a model turns
a `remote=` kwarg into one, and how to add your own.

## Architecture

### The Backend base

`Backend` (`src/nnsight/tracing/backend.py`) is a one-liner:

```python
class Backend:
    def __call__(self, tracer):
        tracer.execute(tracer.info.code)
```

By the time a backend runs, the block is already captured and compiled (during
`Tracer.capture`), so the backend's only job is to decide what "run it" means. The
default just executes it locally — which, for `InterleavingTracer`, means
`InterleavingTracer.execute` sets up the interleaver and runs the model. It is step
4 of the [trace lifecycle](./architecture-overview.md#lifecycle--sequence). A tracer
gets the default `Backend` unless one is passed to `trace(backend=...)`. Editing is
not a backend concern (`EditingTracer.execute` stores the block as a replayed
`Mediator`), and traceback cleanup stays in `Tracer.__exit__`.

### How a backend is selected

`Tracer.__init__` defaults `self.backend` to `Backend()` (`tracing/tracer.py`).
The model layer intercepts the `remote=` kwarg in `Remotable.trace`
(`src/nnsight/modeling/mixins/remotable.py`):

```python
def trace(self, *inputs, backend=None, remote=False, blocking=True,
          job_id=None, verbose=False, **kwargs):
    if backend is None and remote:
        backend = self._remote_backend(remote, blocking, job_id, verbose)
    return super().trace(*inputs, backend=backend, **kwargs)
```

`_remote_backend` (`remotable.py`) maps:

| call | backend |
|---|---|
| (nothing) | default `Backend` → local execution |
| `remote=True` | `RemoteBackend(model.to_model_key(), ...)` |
| `remote="local"` | `LocalSimulationBackend(model)` |
| `remote="https://host"` | `RemoteBackend(..., host="https://host")` |
| `backend=AsyncRemoteBackend(...)` | async remote (passed explicitly) |

`session()` accepts the same `remote=` and puts the backend on the *session* scope,
so all inner traces run as one remote job (`remotable.py`).

### Model identity: to_model_key

A remote backend never serializes the model — it names it. `to_model_key()`
(`remotable.py`) returns `"{import.path.ClassName}:{model_specific_key}"`. The
import-path part comes from `_remoteable_class()` (`remotable.py`), which a
deprecated alias overrides to return the canonical class, so a model wrapped as
`LanguageModel` and one wrapped as `TransformersModel` produce the *same* key. The
suffix is JSON: `HuggingFaceModel` puts the canonical repo id and revision in it,
plus `"trust_remote_code": true` when the model was built with that option (a
checkpoint whose code may run is a different model from the same checkpoint
whose code may not, and a server has to see the request to decide whether to
honor it; the field is omitted when unset, so an ordinary checkpoint's key is the
same one older clients mint). `TransformersModel` adds the resolved pipeline
`task`, so two tasks over one checkpoint (which can load different architecture
classes) are two deployments and the server rebuilds the pipeline from the key
rather than re-inferring it. The server reconstructs with `from_model_key`, and
the key wins over any kwarg it overlaps with: nothing passed at rebuild can make a
key that does not say `trust_remote_code` load as if it did. `TestModelKey` in
`tests/test_serialization.py` asserts this.

### RemoteBackend — NDIF over one websocket

`RemoteBackend` (`src/nnsight/intervention/backends/remote.py`) serializes the
trace, sends it, streams status updates back, and returns the saved values. Two
modes share one serialize+send step:

- **Blocking** (`blocking=True`, default) — `request` (`remote.py`) opens a
  websocket to `/subscribe`, takes the session id, POSTs the payload to `/request`,
  then loops on `connection.recv()` until a `COMPLETED` status arrives. The result
  blob arrives inline as a binary websocket frame or as a presigned URL to download,
  depending on its size, and is `torch.load`ed. The saved values are
  marked (`save`) and pushed back into the caller's frame, just like a local trace
  (`_push`, `remote.py`).
- **Non-blocking** (`blocking=False`) — `submit` (`remote.py`) POSTs without a
  websocket (the server saves each status to the object store) and stores the
  server-assigned `job_id`; `poll` (`remote.py`) GETs `/response/{job_id}` and
  returns the saves dict on `COMPLETED`, `None` while still running. Because the
  trace has long exited by the time you poll, the result is *returned*, not pushed
  into a frame. Construct with a `job_id` to make a poll-only backend for an
  existing job.

Serialization is done by `RequestModel.serialize` (`schema/request.py`), after
`pull_env()` registers local (non-installed) modules for by-value pickling so their
source ships with the request (`_serialize`, `remote.py`). Status rendering,
compression, and result download are shared helpers (`note`/`handle`/
`download_result`/`finalize`). `StatusDisplay` (`backends/display.py`) animates the
progress display when remote logging or `verbose` is on.

Config comes from `CONFIG.API` / `CONFIG.APP` (`schema/config.py`): `HOST`,
`APIKEY`, `COMPRESS`, `APP.DEBUG`, `APP.REMOTE_LOGGING` (env: `NDIF_API_KEY`,
`NDIF_HOST`).

### AsyncRemoteBackend — the same request, awaited

`AsyncRemoteBackend` (`remote.py`) subclasses `RemoteBackend`. Its `__call__`
(`remote.py`) fires the request the way the blocking parent does — subscribe,
take the session id, POST — then returns `self` **without** consuming any updates
(the connect/serialize/POST are synchronous; only the waiting is async). Two ways to
consume it:

```python
backend = AsyncRemoteBackend(model.to_model_key())
with model.trace(prompt, backend=backend):
    out = model.output.save()

result = await backend                 # resolve(): wait for COMPLETED, return saves dict
# or
async for update in backend:           # stream(): raw ResponseModel updates...
    if isinstance(update, dict):
        result = update                 # ...and the saves dict, yielded last
    else:
        print(update.status)
```

`await backend` (`resolve`, `remote.py`) renders the display and raises on a
server error, like the blocking parent. `async for` (`stream`, `remote.py`)
hands you each raw `ResponseModel` and does *not* touch the display or raise. The
blocking `recv` runs through `asyncio.to_thread` (`receive`, `remote.py`) to keep
the event loop free. The caller's frame is gone by the time the result lands, so
saves come out of the await / the iterator's final item, not a frame push.


### LocalSimulationBackend — a serverless dry run

`LocalSimulationBackend` (`src/nnsight/intervention/backends/local.py`) powers
`model.trace(..., remote="local")`. It serializes the trace exactly as
`RemoteBackend` would, then deserializes it **with local (non-installed) modules
hidden** — mimicking a server where the user's own source files don't exist — and
runs the deserialized block locally against the real model:

```python
pull_env()                                     # register local modules by-value
blob = RequestModel.serialize(tracer, compress=False)
hidden = self._hide_local_modules()            # pop non-installed modules & paths
try:
    restored = RequestModel.deserialize(blob, persistent, compress=False)
finally:
    self._restore(hidden)
tracer.info.code = restored.info.code
tracer.execute(tracer.info.code)               # run in-process; push back into the frame
```

`_hide_local_modules` (`local.py`) removes any `sys.path` entry that isn't
site-packages/stdlib/nnsight's own `src`, and any module whose root isn't in
`_SERVER_MODULES` (`local.py` — `torch`, `numpy`, `transformers`, `accelerate`,
`diffusers`, `einops`, `peft`, `nnsight`) or the stdlib. If the block references a
local function/class that wasn't shipped by value, the deserialize raises
`ModuleNotFoundError` exactly as a real server would — so a passing `remote="local"`
run is strong evidence a real remote run will work, without a live server. Results
land back in the caller's frame like an ordinary local trace.
`_SERVER_MODULES` is the source of truth for what NDIF supports without an
`ndif.register(...)` call — edit it if the server environment changes.

## Adding a backend

Subclass `Backend` when you want to change **what is done with the captured
block** — log it, ship it, store it. A different inference engine is a new model
class instead ([adding-a-new-runtime.md](./adding-a-new-runtime.md)).

When your `__call__(tracer)` runs, the block is already compiled:
`tracer.info.code` is the body (a `CodeType`), `tracer.info.frame` the caller's
live frame, and `tracer.node` the block's AST node (`None` after deserialization).

**Run locally, but do something first:**

```python
from nnsight.tracing.backend import Backend

class LoggingBackend(Backend):
    def __init__(self, log_path="trace.log"):
        self.log_path = log_path

    def __call__(self, tracer):
        code = tracer.info.code
        with open(self.log_path, "a") as f:
            f.write(f"{code.co_filename}:{code.co_firstlineno} {code.co_name}\n")
        tracer.execute(code)                    # what the default backend does

with model.trace("Hello", backend=LoggingBackend("nn.log")):
    out = model.transformer.h[0].output.save()
```

The block's source text is in the cache the tracer already filled:
`nnsight.tracing.globals.SOURCES[code.co_filename]`.

**Skip local execution and ship it** (the shape of `RemoteBackend.request`):

```python
class MyRemoteBackend(Backend):
    def __init__(self, model_key):
        self.model_key = model_key

    def __call__(self, tracer):
        from nnsight.schema.request import RequestModel
        from nnsight.ndif import pull_env
        from nnsight.tracing.tracer import save
        from nnsight.tracing.util import push

        pull_env()                                         # register local modules by value
        blob = RequestModel.serialize(tracer, compress=False)
        result = my_transport.send(self.model_key, blob)   # your wire
        for value in result.values():                      # push saves back like a local run
            save(value)
        push(tracer.info.frame, result)
```

Usually it is easier to subclass `RemoteBackend` and override only the transport
(`send`/`request`/`poll`), keeping its serialize, status display and download
helpers. An async backend follows the shape of
[`AsyncRemoteBackend`](#asyncremotebackend--the-same-request-awaited). To model a different
server environment for dry runs, subclass `LocalSimulationBackend` and adjust
`_SERVER_MODULES` / `_hide_local_modules`.

Wire it up with `model.trace(..., backend=MyBackend())`, or for a `remote=`-keyed
backend, add a branch in your model's `Remotable._remote_backend`.

## Related

- [serialization.md](./serialization.md) — how the payload is (de)serialized.
