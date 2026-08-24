---
title: Backends
one_liner: A Backend decides what running a captured trace means — execute locally, ship to NDIF (blocking/non-blocking/async), or serialize-and-run for a local dry run.
tags: [internals, dev]
related: [docs/developing/tracing-pipeline.md, docs/developing/adding-a-new-backend.md, docs/developing/serialization.md]
sources: [src/nnsight/tracing/backend.py, src/nnsight/intervention/backends/remote.py, src/nnsight/intervention/backends/local.py, src/nnsight/intervention/backends/display.py, src/nnsight/modeling/mixins/remotable.py]
---

# Backends

## What this covers

A `Backend` is the bridge between the tracing system (which captures and compiles a
`with` block) and what actually happens to it on `__exit__`. This doc walks the
`Backend` base and the three concrete backends: `RemoteBackend`,
`AsyncRemoteBackend`, and `LocalSimulationBackend`, plus how a model turns a
`remote=` kwarg into one.

## Architecture

### The Backend base

`Backend` (`src/nnsight/tracing/backend.py:9`) is a one-liner:

```python
class Backend:
    def __call__(self, tracer):
        tracer.execute(tracer.info.code)
```

By the time a backend runs, the block is already captured and compiled (during
`Tracer.capture`), so the backend's only job is to decide what "run it" means. The
default just executes it locally — which, for `InterleavingTracer`, means
`InterleavingTracer.execute` sets up the interleaver and runs the model. A tracer
gets the default `Backend` unless one is passed to `trace(backend=...)`.

There is **no** `ExecutionBackend`, `EditingBackend`, or `Backend.__call__` compile
pipeline as in older nnsight. Local execution *is* the default `Backend`; editing is
done by `EditingTracer.execute` storing the block (not by a backend); exception
cleanup happens in `Tracer.__exit__` via `clean_traceback` (`tracing/util.py:139`),
not a backend `wrap_exception`.

### How a backend is selected

`Tracer.__init__` defaults `self.backend` to `Backend()` (`tracing/tracer.py:248`).
The model layer intercepts the `remote=` kwarg in `Remotable.trace`
(`src/nnsight/modeling/mixins/remotable.py:19`):

```python
def trace(self, *inputs, backend=None, remote=False, blocking=True,
          job_id=None, verbose=False, **kwargs):
    if backend is None and remote:
        backend = self._remote_backend(remote, blocking, job_id, verbose)
    return super().trace(*inputs, backend=backend, **kwargs)
```

`_remote_backend` (`remotable.py:52`) maps:

| call | backend |
|---|---|
| (nothing) | default `Backend` → local execution |
| `remote=True` | `RemoteBackend(model.to_model_key(), ...)` |
| `remote="local"` | `LocalSimulationBackend(model)` |
| `remote="https://host"` | `RemoteBackend(..., host="https://host")` |
| `backend=AsyncRemoteBackend(...)` | async remote (passed explicitly) |

`session()` accepts the same `remote=` and puts the backend on the *session* scope,
so all inner traces run as one remote job (`remotable.py:35`).

### Model identity: to_model_key

A remote backend never serializes the model — it names it. `to_model_key()`
(`remotable.py:125`) returns `"{import.path.ClassName}:{model_specific_key}"`. The
import-path part comes from `_remoteable_class()` (`remotable.py:115`), which a
deprecated alias overrides to return the canonical class, so a model wrapped as
`LanguageModel` and one wrapped as `TransformersModel` produce the *same* key the
server knows it by. The server reconstructs with `from_model_key`.

### RemoteBackend — NDIF over one websocket

`RemoteBackend` (`src/nnsight/intervention/backends/remote.py:39`) serializes the
trace, sends it, streams status updates back, and returns the saved values. Two
modes share one serialize+send step:

- **Blocking** (`blocking=True`, default) — `request` (`remote.py:304`) opens a
  websocket to `/subscribe`, takes the session id, POSTs the payload to `/request`,
  then loops on `connection.recv()` until a `COMPLETED` status arrives, downloading
  the result blob from a presigned URL and `torch.load`ing it. The saved values are
  marked (`save`) and pushed back into the caller's frame, just like a local trace
  (`_push`, `remote.py:112`).
- **Non-blocking** (`blocking=False`) — `submit` (`remote.py:263`) POSTs without a
  websocket (the server saves each status to the object store) and stores the
  server-assigned `job_id`; `poll` (`remote.py:280`) GETs `/response/{job_id}` and
  returns the saves dict on `COMPLETED`, `None` while still running. Because the
  trace has long exited by the time you poll, the result is *returned*, not pushed
  into a frame. Construct with a `job_id` to make a poll-only backend for an
  existing job.

Serialization is done by `RequestModel.serialize` (`schema/request.py`), after
`pull_env()` registers local (non-installed) modules for by-value pickling so their
source ships with the request (`_serialize`, `remote.py:249`). Status rendering,
compression, and result download are shared helpers (`note`/`handle`/
`download_result`/`finalize`). `StatusDisplay` (`backends/display.py`) animates the
progress display when remote logging or `verbose` is on.

Config comes from `CONFIG.API` / `CONFIG.APP` (`schema/config.py`): `HOST`,
`APIKEY`, `COMPRESS`, `APP.DEBUG`, `APP.REMOTE_LOGGING` (env: `NDIF_API_KEY`,
`NDIF_HOST`).

### AsyncRemoteBackend — the same request, awaited

`AsyncRemoteBackend` (`remote.py:339`) subclasses `RemoteBackend`. Its `__call__`
(`remote.py:377`) fires the request the way the blocking parent does — subscribe,
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

`await backend` (`resolve`, `remote.py:398`) renders the display and raises on a
server error, like the blocking parent. `async for` (`stream`, `remote.py:417`)
hands you each raw `ResponseModel` and does *not* touch the display or raise. The
blocking `recv` runs through `asyncio.to_thread` (`receive`, `remote.py:437`) to keep
the event loop free. The caller's frame is gone by the time the result lands, so
saves come out of the await / the iterator's final item, not a frame push.


### LocalSimulationBackend — a serverless dry run

`LocalSimulationBackend` (`src/nnsight/intervention/backends/local.py:37`) powers
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

`_hide_local_modules` (`local.py:81`) removes any `sys.path` entry that isn't
site-packages/stdlib/nnsight's own `src`, and any module whose root isn't in
`_SERVER_MODULES` (`local.py:25` — `torch`, `numpy`, `transformers`, `accelerate`,
`diffusers`, `einops`, `peft`, `nnsight`) or the stdlib. If the block references a
local function/class that wasn't shipped by value, the deserialize raises
`ModuleNotFoundError` exactly as a real server would — so a passing `remote="local"`
run is strong evidence a real remote run will work, without a live server. Results
land back in the caller's frame like an ordinary local trace.

`_SERVER_MODULES` is the source of truth for what NDIF supports without an
`ndif.register(...)` call — edit it if the server environment changes.

## Key files / classes

- `src/nnsight/tracing/backend.py:9` — `Backend`. `tracer.execute(tracer.info.code)`.
- `src/nnsight/intervention/backends/remote.py:39` — `RemoteBackend`. Blocking + non-blocking.
- `:304` — `RemoteBackend.request` (blocking websocket loop).
- `:263`/`:280` — `submit`/`poll` (non-blocking).
- `:339` — `AsyncRemoteBackend`; `:398` — `resolve`; `:417` — `stream`.
- `src/nnsight/intervention/backends/local.py:37` — `LocalSimulationBackend`; `:25` — `_SERVER_MODULES`.
- `src/nnsight/intervention/backends/display.py` — `StatusDisplay`.
- `src/nnsight/modeling/mixins/remotable.py:19` — `Remotable.trace` (backend selection); `:52` — `_remote_backend`; `:125` — `to_model_key`.

## Lifecycle (one trace, default backend)

1. User exits `with model.trace(...)`. `Tracer.__exit__` calls `self.backend(self)`.
2. Default `Backend.__call__` calls `tracer.execute(tracer.info.code)`.
3. `InterleavingTracer.execute` builds the batcher, makes the workers, and calls
   `Envoy.interleave`, which runs the model interleaved with the workers.
4. `push_result` writes saved values back; `__exit__`'s outermost `dec()` filters to
   `.save()`-ed values.

For `remote=True`, step 2 is replaced by `RemoteBackend.__call__` serializing and
sending the trace; the block never runs locally.

## Extension points

- **A new transport.** Subclass `Backend` (or `RemoteBackend` to reuse serialize/
  send/download). See `docs/developing/adding-a-new-backend.md`.
- **A stricter local simulation.** Trim `_SERVER_MODULES` or the path filters in
  `LocalSimulationBackend._hide_local_modules`.
- **Persistent transforms.** Not a backend concern here — use `model.edit()`
  (`EditingTracer`), which stores a block as a replayed `Mediator`.

## Related

- `docs/developing/tracing-pipeline.md` — what the backend receives (`tracer.info`).
- `docs/developing/adding-a-new-backend.md` — recipe for writing your own.
- `docs/developing/serialization.md` — how the payload is (de)serialized.
