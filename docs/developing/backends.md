---
title: Backends
one_liner: Backend classes compile traced source into a callable and execute it; subclasses dispatch local, remote, edit, or simulated runs.
tags: [internals, dev]
related: [docs/developing/tracing-pipeline.md, docs/developing/serialization.md, docs/developing/adding-a-new-backend.md]
sources: [src/nnsight/intervention/backends/base.py:1, src/nnsight/intervention/backends/execution.py:1, src/nnsight/intervention/backends/remote.py:421, src/nnsight/intervention/backends/editing.py:1, src/nnsight/intervention/backends/local_simulation.py:1]
---

# Backends

## What this covers

A `Backend` is the bridge between the tracing system (which captures source) and execution (which runs the compiled function). The base class compiles the captured source into a Python function with caching; subclasses decide what to do with that function — run it locally, ship it off to NDIF, attach it as a persistent edit on the model, or simulate a remote round-trip in-process. This document walks through `Backend.__call__`, the code-object cache, and each subclass.

## Architecture / How it works

### `Backend.__call__(tracer)` — the compile step

The base class (`src/nnsight/intervention/backends/base.py:18`) implements the compile pipeline shared by every backend:

1. **Indent.** Each line of `tracer.info.source` is prefixed with `    ` to live inside a function body (`src/nnsight/intervention/backends/base.py:52`).
2. **Wrap.** `tracer.compile()` prepends a `def __nnsight_tracer_<key>__(...)` signature plus a `tracer.pull()` call, and appends a `tracer.push()` (or `get_frame()`) closer. Different tracer subclasses produce different wrappers — see `Tracer.compile` at `src/nnsight/intervention/tracing/base.py:441` and `InterleavingTracer.compile` at `src/nnsight/intervention/tracing/tracer.py:335`.
3. **Compile or cache-hit.** A `cache_key = (tracer.info.cache_key, type(tracer).__name__)` is built. `Globals.cache.get_code(cache_key)` (`src/nnsight/intervention/tracing/globals.py:71`) is checked first; if absent, `compile(source, filename, "exec")` runs and the result is stored via `add_code`. The tracer-type suffix is critical because a `with` block at the same line can be compiled differently depending on which tracer wraps it.
4. **Exec.** The code object is exec'd against `{**frame.f_globals, **frame.f_locals}` into a fresh `local_namespace`, and the resulting function is returned.

The base class never executes the function — that is left to subclasses.

### Code-object caching strategy

Caching happens at two layers:

- **AST + source cache** keyed by `(filename, start_line, function_name, co_firstlineno)` — populated by `Tracer.capture()` at `src/nnsight/intervention/tracing/base.py:229`. Reused across all tracer types.
- **Compiled code cache** keyed by `(cache_key, tracer_type_name)` — populated by `Backend.__call__` at `src/nnsight/intervention/backends/base.py:65`.

Both live on the singleton `Globals.cache` (`src/nnsight/intervention/tracing/globals.py:53`). `TracingCache` exposes `get`, `add`, `get_code`, `add_code`, and `clear`. The `clear()` method is occasionally needed when source files change at runtime — see [testing.md](./testing.md).

### `ExecutionBackend` — local, synchronous

`ExecutionBackend` (`src/nnsight/intervention/backends/execution.py:13`) is the default. It calls `super().__call__(tracer)` to get the compiled function, then enters the `Globals` context (which mounts pymount and increments the trace stack), calls `tracer.execute(fn)`, and exits. Exceptions inside the traced function are wrapped via `wrap_exception(e, tracer.info)` (`src/nnsight/intervention/tracing/util.py`) so the user sees a clean traceback pointing at their source rather than NNsight internals.

### `RemoteBackend` — NDIF submission

`RemoteBackend` (`src/nnsight/intervention/backends/remote.py:421`) does not call `tracer.execute(fn)` locally. Instead its `request(tracer)` (`src/nnsight/intervention/backends/remote.py:514`) calls the base `Backend.__call__` to compile and produce the intervention function (named `interventions` in this context), wraps it in a `RequestModel(interventions=interventions, tracer=tracer)`, and serializes via the source-based pickler (`src/nnsight/intervention/serialization.py:979`). The serialized payload is POSTed over HTTPS, and a Socket.IO WebSocket carries status updates back.

Three execution modes share the same compile step:

- **Blocking.** `blocking_request(tracer)` (`src/nnsight/intervention/backends/remote.py:869`) opens a WebSocket, submits via HTTP, and loops on `sio.receive()` until a `COMPLETED` status arrives, downloading and `torch.load`ing the result.
- **Non-blocking.** `non_blocking_request(tracer)` (`src/nnsight/intervention/backends/remote.py:990`) submits on first call and polls on subsequent calls. The user can poll `backend.job_status` and `backend.job_id` between calls.
- **Async.** `async_request(tracer)` (`src/nnsight/intervention/backends/remote.py:930`) is the asyncio-friendly version of blocking, used when a tracer enters via `async with`.

A `LocalTracer` (`src/nnsight/intervention/backends/remote.py:1020`) is registered for `STREAM` responses so the server can ship a function back for the client to execute, with results streamed back via the same WebSocket. This enables hybrid local/remote execution (e.g. data-dependent control flow that needs the user's environment).

### `EditingBackend` — persistent edits

`EditingBackend` (`src/nnsight/intervention/backends/editing.py:11`) supports `model.edit()` contexts. It creates an invoker (so the edit code is wrapped as if it were inside `tracer.invoke()`), calls `super().__call__` to compile, builds a `Mediator(fn, invoker.info)`, and appends it to `model._default_mediators`. From that point on, every subsequent trace on the model picks up these mediators in `InterleavingTracer.compile` (`src/nnsight/intervention/tracing/tracer.py:344`), so the edit applies to every forward pass.

`model.clear_edits()` empties `_default_mediators`. Edits can be in-place (mutating the live model) or non-inplace (returning a separately tracked envoy) — both go through `EditingBackend`.

### `LocalSimulationBackend` — round-trip serialization in-process

`LocalSimulationBackend` (`src/nnsight/intervention/backends/local_simulation.py:39`) is what powers `model.trace(..., remote='local')`. It serializes and deserializes the request **in the user's process**, while temporarily blocking the user modules from `sys.modules` and `sys.path` so the deserialization sees only what NDIF would see. This catches missing imports or unpickleable closures before paying for a real NDIF submission.

The blocking step (`src/nnsight/intervention/backends/local_simulation.py:86`):

- Removes any `sys.path` entry that is not site-packages, stdlib, or nnsight's own `src/`.
- Removes any module from `sys.modules` whose root is outside the `SERVER_MODULES` allowlist (`src/nnsight/intervention/backends/local_simulation.py:19`) and not a stdlib module.

After deserialization, `_restore_modules` (`src/nnsight/intervention/backends/local_simulation.py:123`) puts everything back. The restored function is then executed locally via `tracer.execute(restored.interventions)` so the user gets results identical to a real remote run.

The `SERVER_MODULES` set is the source of truth for what NDIF currently supports without an `ndif.register(...)` call. Edit it if the server's environment changes.

### How tracers select backends

Backend selection is driven by user kwargs on `model.trace(...)`:

- No backend specified: `ExecutionBackend()` (default in `Tracer.__init__` at `src/nnsight/intervention/tracing/base.py:195`).
- `remote=True`: `RemoteBackend(model_key, ...)` constructed in the model's mixin (`RemoteableMixin`).
- `remote='local'`: `LocalSimulationBackend(model)`.
- `model.edit()`: forces `EditingBackend()`.
- vLLM async path: `VLLM.trace` (`src/nnsight/modeling/vllm/vllm.py:445`) injects `AsyncVLLMBackend(self)`.

Tracers carry the backend on `self.backend` and call `self.backend(self)` from `__exit__` (`src/nnsight/intervention/tracing/base.py:691`).

## Key files / classes

- `src/nnsight/intervention/backends/base.py:18` — `Backend.__call__` (compile pipeline + cache lookup)
- `src/nnsight/intervention/backends/execution.py:13` — `ExecutionBackend` (local sync)
- `src/nnsight/intervention/backends/remote.py:421` — `RemoteBackend` (NDIF: blocking, non-blocking, async)
- `src/nnsight/intervention/backends/remote.py:1020` — `LocalTracer` (streamed-back local execution)
- `src/nnsight/intervention/backends/editing.py:11` — `EditingBackend` (persistent edits via `_default_mediators`)
- `src/nnsight/intervention/backends/local_simulation.py:39` — `LocalSimulationBackend` (in-process NDIF simulation)
- `src/nnsight/intervention/backends/local_simulation.py:19` — `SERVER_MODULES` allowlist
- `src/nnsight/intervention/tracing/globals.py:53` — `TracingCache` (source cache + code cache)
- `src/nnsight/modeling/vllm/async_backend.py:19` — `AsyncVLLMBackend` (dual-call backend; see [vllm-integration.md](./vllm-integration.md))

## Lifecycle (one trace, default backend)

1. User enters `with model.trace(...)`. `Tracer.capture()` extracts source and stores `Tracer.Info`.
2. User exits the block. `Tracer.__exit__` calls `self.backend(self)`.
3. `ExecutionBackend.__call__` calls `Backend.__call__(tracer)`:
   - `tracer.compile()` prepends/appends function wrapper.
   - `Globals.cache.get_code(cache_key)` is consulted; on miss, `compile(...)` runs and the result is cached.
   - `exec(code_obj, frame_globals_and_locals, local_namespace)` produces the callable.
4. `Globals.enter()` is called.
5. `tracer.execute(fn)` runs — this is where `InterleavingTracer.execute` (`src/nnsight/intervention/tracing/tracer.py:412`) sets up the interleaver and calls `model.interleave(self.fn, *args, **kwargs)`.
6. `Globals.exit()` is called from the backend's `finally`.

## Extension points

- **Custom backend.** Subclass `Backend`, override `__call__(tracer)`. See [adding-a-new-backend.md](./adding-a-new-backend.md). You almost always want to call `super().__call__(tracer)` first to get the compiled function.
- **Persistent transforms (other than edits).** Mirror `EditingBackend`'s pattern — wrap the compiled function in a `Mediator` and append to `model._default_mediators`.
- **Stricter local simulation.** Trim `SERVER_MODULES` (`src/nnsight/intervention/backends/local_simulation.py:19`) or add path filters to `_block_user_modules` to model a tighter NDIF environment.
- **Hybrid local/remote.** Use `LocalTracer.register(callback)` to receive functions streamed back from the server. The default `RemoteBackend` already wires this up for blocking and async modes.

## Related

- [tracing-pipeline.md](./tracing-pipeline.md) — what `tracer.capture` and `tracer.compile` produce before backends run
- [serialization.md](./serialization.md) — how `RemoteBackend` and `LocalSimulationBackend` serialize the payload
- [vllm-integration.md](./vllm-integration.md) — `AsyncVLLMBackend`'s dual-call pattern
- [adding-a-new-backend.md](./adding-a-new-backend.md) — recipe for writing your own
