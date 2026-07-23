---
title: Serialization Internals
one_liner: Source-based (not bytecode) serialization for cross-Python-version remote execution.
tags: [internals, dev]
related: [docs/developing/backends.md, docs/developing/vllm-integration.md]
sources: [src/nnsight/intervention/serialization.py, src/nnsight/schema/request.py, src/nnsight/schema/response.py, src/nnsight/tracing/util.py, src/nnsight/intervention/backends/remote.py, src/nnsight/intervention/backends/local.py]
---

# Serialization Internals

## What this covers

NDIF runs code submitted from any Python 3.10+ client regardless of the server's
Python version. Standard `cloudpickle` serializes functions by their `__code__`
bytecode, and bytecode is version-specific — a function pickled on 3.10 can fail
to load on 3.11. NNsight instead serializes code by its **source text**, then
recompiles it on the far side. This covers the source-based pickler, block
reduction, the persistent-object reference protocol (model/tokenizer passed by
ID), the `RequestModel` envelope, `SerializedFrame`, `Mediator.__getstate__` (how
an edit rides to the server), and the `remote="local"` dry-run backend.

## Quick check (verified round-trips)

`dumps`/`loads` are the drop-in pickle-style API in
`src/nnsight/intervention/serialization.py`:

```python
from nnsight.intervention.serialization import dumps, loads

# a lambda closing over a local
m = 3
loads(dumps(lambda x: x * m))(5)          # -> 15

# a recursive module-level function
def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)
loads(dumps(fib))(10)                       # -> 55
```

Persistent objects are referenced by ID instead of serialized. Any object with
`_persistent_id` in its `__dict__` is written as that ID; the unpickler resolves
it against a `persistent_objects` map:

```python
from nnsight.intervention.serialization import dumps, loads, UnknownPersistentIdError

class Tok: pass
t = Tok(); t._persistent_id = "Tokenizer"
blob = dumps({"tok": t, "n": 5})

loads(blob, persistent_objects={"Tokenizer": t})["tok"] is t   # -> True
loads(blob)                                                    # -> UnknownPersistentIdError: Tokenizer
```

The full round-trip suite is `tests/test_serialization.py` (run it with
`CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_serialization.py`).

## Architecture

### Why source instead of bytecode

Module docstring (`serialization.py`):

> Standard cloudpickle serializes code as *bytecode*, which is tied to the exact
> Python version — a payload pickled on 3.10 can fail to load on 3.11. This module
> serializes code by its **source** instead, so it reconstructs on any Python that
> can parse the syntax (client and server need not match).

Two layered mechanisms: `code_reduce` (capture code as text plus only the names it
references) and persistent ids (`_persistent_id`, reference heavy objects instead
of pickling them).

### `code_reduce` and `reduce_block`

- `code_reduce(source, globals, locals)` (`serialization.py:147`) is the core
  reducer. It returns `(source, used_globals, used_locals)` — the code as text
  plus **only** the globals/locals that source actually references, found via
  `_referenced_names(source)` (`:38`, which compiles the source and walks
  `co_names` recursively). Names are looked up by subscript, so a `Scope` globals
  resolves through its fallback chain and module globals (`torch`, etc.) stay out
  of the payload.
- `reduce_block(node, globals, locals)` (`serialization.py:176`) reduces a
  captured `with`-block AST node. It unparses `node.body` to source, padding with
  leading blank lines so each statement keeps its original line number (correct
  remote tracebacks), then calls `code_reduce`. This is shared by the request
  payload (the traced block) and edit-mediator serialization.

### The pickler / unpickler

- `CustomCloudPickler(cloudpickle.CloudPickler)` (`serialization.py:253`) —
  overrides `_dynamic_function_reduce(func)` (`:256`) to serialize a dynamic
  `def`/`lambda` by source: `_lambda_source(func)` (`:56`) for lambdas, otherwise
  `textwrap.dedent(inspect.getsource(func))`. Closure cell contents are captured
  via cloudpickle's `_get_cell_contents` and folded into the globals. It returns
  the 6-tuple reduce form `(_load_function, args, globals, None, None, _fill)`. On
  `(OSError, TypeError)` — C functions, dataclass methods, REPL functions with no
  recoverable source — it falls back to cloudpickle's bytecode path.
- `persistent_id(obj)` (`serialization.py:300`) returns
  `obj.__dict__["_persistent_id"]` when present (else `None`).
- `CustomCloudUnpickler(pickle.Unpickler)` (`serialization.py:312`) overrides
  `persistent_load(pid)` (`:317`) to return `self.persistent_objects[pid]`, raising
  `UnknownPersistentIdError` (`:308`) if the ID is absent.

### Reconstruction

- `_load_function(...)` (`serialization.py:191`) rebuilds a function from a
  `code_reduce` tuple. It compiles the (blank-padded) source and **lifts out the
  function's code object** rather than executing the `def` — so defaults and
  annotations aren't re-evaluated against incomplete globals. It compiles under a
  synthetic filename `f"[{name}:{line}] {origin_filename}"` and registers the
  source in `linecache`, so remote tracebacks and `inspect.getsource` work.
- `_fill(func, globals)` (`serialization.py:243`) populates the function's globals
  **after** pickle has memoized it, so a self-reference resolves via the memo
  instead of recursing during unpickling. This is how recursive and mutually
  recursive functions reconstruct.

### `_lambda_source` (lambda extraction)

`inspect.getsource` on a lambda returns the whole enclosing line, ambiguous when
several lambdas share it (`f, g = lambda x: x*2, lambda x: x+1`).
`_lambda_source` (`serialization.py:56`) AST-parses the line and disambiguates by
matching parameter names (`co_varnames[:co_argcount]`), then by source-span
enclosure using `func.__code__.co_positions()`. It handles nested lambdas, lambda
defaults, and multi-line lambdas (wrapping unparseable fragments in a `def`).
`tests/test_serialization.py::TestLambda` is the reference for the edge cases.

### Top-level API

`dump`/`dumps` (`serialization.py:323`, `:328`) and `load`/`loads` (`:335`, `:349`).
`load`/`loads` take `persistent_objects=None` and an `unpickler=` override so a
server can substitute its own unpickler. `DEFAULT_PROTOCOL = 4`.

## The request envelope — `schema/request.py`

`RequestModel(BaseModel)` (`request.py:15`) is the JSON routing envelope. The
model itself is never carried — it is identified by `model_key`. Fields:

| Field | Meaning |
|-------|---------|
| `model_key: str` | identifies the server-side model (see `to_model_key()`) |
| `session_id: str` | session grouping |
| `compress: bool` | zstd-compress the payload and ask the server to compress its result |
| `env: dict` | per-request environment (e.g. `{"peft": <adapter repo id>}`) |

- `serialize(tracer, compress=False)` (`request.py:37`, classmethod) builds the
  binary execution payload: `variables = dict(tracer.info.frame.f_locals)`,
  `interventions = reduce_block(tracer.node, glbls, variables)`, then
  `dumps((tracer, interventions))`, zstd level 6 if `compress`.
- `deserialize(blob, persistent_objects=None, compress=False, ...)`
  (`request.py:57`, staticmethod) is the inverse: `tracer, (source, glbls, vars) =
  loads(...)`, recompiles the block into `tracer.info.code`, registers it in
  `linecache`, and restores `frame.f_globals`/`f_locals`. It returns a
  ready-to-run tracer; the server then calls `tracer.execute(tracer.info.code)`.

The tracer is pickled whole (it carries the model invocation and a
`SerializedFrame`); the block source travels separately as `interventions` and is
re-attached as `tracer.info.code`.

## `SerializedFrame` — `tracing/util.py`

A real frame can't be pickled, so `tracer.info.frame` is a `SerializedFrame`
(`util.py:82`): a picklable stand-in carrying **only** code metadata
(`co_filename`, `co_firstlineno`, `co_name`) needed for remote tracebacks and
source lookup. `f_locals`/`f_globals` are empty dicts — the real locals/globals
travel in the `interventions` tuple and are written back onto the frame in
`RequestModel.deserialize`. Build one from a live frame with `SerializedFrame.of(frame)`.

Related helpers in the same module: `Scope(dict)` (`:32`), the 3-tier namespace a
captured block runs in (whose iteration yields only the block's own names — what
the reducers rely on); `push(frame, variables)` (`:150`), which writes results
back into a live frame (`PyFrame_LocalsToFast` before 3.13, a plain `update` on
3.13+); and `clean_traceback` (`:139`), which strips internal nnsight frames.

## Edits ride to the server — `Mediator.__getstate__`

An `edit` (from `model.edit(...)`) is a `Mediator` stored in `envoy._edits`.
`Mediator.__getstate__` (`interleaver.py:212`) serializes it cross-version-safely:
it returns `{"reduced": reduce_block(self.node, self.glbls, self.lcls), "copy":
self.copy}` — the edit block reduced to source plus referenced vars, and the
`copy` flag. It **deliberately drops** the compiled `code` and all run state
(`worker`, `pending`, `interleaver`, `batch_group`): those can't and shouldn't
travel. `__setstate__` (`:221`) recompiles the source (under filename `<edit>`)
and re-inits.

## Persistent objects — model/tokenizer by reference

Heavy server-side objects (the model, its tokenizer, the module tree, the
interleaver) are never serialized — they're referenced by ID and re-bound to the
server actor's live objects.

- **Tagging (client):** `Envoy.__getstate__` (`envoy.py:248`) sets
  `interleaver._persistent_id = "Interleaver"` and `_module._persistent_id =
  f"Module:{self.path}"`. `TransformersModel.__getstate__` (`transformers.py:533`)
  additionally tags each non-`None` preprocessor and the pipeline, using the
  `_PERSISTENT` map (`transformers.py:80`): `tokenizer → "Tokenizer"`, `processor
  → "Processor"`, `image_processor → "ImageProcessor"`, `feature_extractor →
  "FeatureExtractor"`, `pipeline → "Pipeline"`.
- **Resolving (server):** `Remotable._remoteable_persistent_objects()`
  (`remotable.py:97`) builds the `{id: object}` map — `{"Interleaver":
  self.interleaver}` plus `Module:<path>` for every envoy in `self.modules()`.
  `TransformersModel` extends it (`transformers.py:460`) with the tokenizer and
  friends. This map is passed to `loads(..., persistent_objects=...)`; a missing ID
  raises `UnknownPersistentIdError`.

### Model identity — `to_model_key`

`Remotable.to_model_key()` (`remotable.py:125`) is
`f"{to_import_path(self._remoteable_class())}:{self._remoteable_model_key()}"`.
`_remoteable_class()` (`:115`) returns `type(self)` by default; deprecated aliases
override it to the canonical class (`LanguageModel`/`VisionLanguageModel` return
`TransformersModel`) so all three wrappings resolve to the one key the server
knows. `TestModelKey` in `tests/test_serialization.py` asserts this.

## The response — `schema/response.py`

`Status(str, Enum)` (`response.py:13`): `RECEIVED, QUEUED, PROVISIONING,
DEPLOYING, DISPATCHED, RUNNING, COMPLETED, ERROR, LOG`. `ResponseModel(BaseModel)`
(`:25`) carries `id`, `status`, `description`, and `data`. On `COMPLETED`, `data`
holds a presigned URL; the saves blob is downloaded separately and
`torch.load`-ed into a `RESULT = Dict[str, Any]`. `pickle`/`unpickle` (`:36`,
`:41`) use `torch.save`/`torch.load` so tensors survive the wire.

## The `remote="local"` dry run — `backends/local.py`

`LocalSimulationBackend(Backend)` (`local.py:37`) exercises the entire remote path
without a server. `model.trace(..., remote="local")` serializes the trace exactly
as `RemoteBackend` would, then deserializes it **with the user's local
(non-installed) modules hidden** — mimicking a server where the user's own source
files don't exist — and runs the deserialized block against the real local model.
If the block references a local function/class that wasn't shipped by value,
deserialize raises `ModuleNotFoundError`, exactly as the server would.

`_SERVER_MODULES` (`local.py:25`) is the allowlist of roots assumed present on
NDIF and never hidden: `{"torch", "numpy", "transformers", "accelerate",
"diffusers", "einops", "peft", "nnsight"}`. To simulate a stricter environment,
edit this set. `TestLocalSimulation` / `TestServerExecution` in
`tests/test_serialization.py` are the reference.

Note: `RemoteBackend._serialize` and `LocalSimulationBackend.__call__` both call
`pull_env()` (from `nnsight.ndif`) first, which registers local, non-installed
modules for by-value pickling so the server (or the simulated deserialize) doesn't
hit `ModuleNotFoundError` on the user's own helpers.

## `obj.save()` and the pymount C extension

There are two ways to mark a value as saved inside a trace:

- `nnsight.save(obj)` — a plain function.
- `obj.save()` — a method mounted onto Python's base `object` type by the
  `nnsight._c.py_mount` C extension (`src/nnsight/_c/py_mount.c`), which edits the
  C-level method table because `object` rejects attribute assignment from Python.

The extension is optional and built by `setup.py` (an `Extension` over
`src/nnsight/_c/py_mount.c`). If no C compiler is available, setuptools silently
skips it and `obj.save()` is unavailable — which is exactly why the Docker image
installs `gcc`/`libc6-dev` so `.save()` works server-side. `nnsight.save(obj)`
works either way. `tests/test_saving.py::TestSaveMethod` is skipped when the mount
isn't built (`skipif(not save_mounted, ...)`).

Both forms now **raise if called outside a running trace** — marking a value where
no trace will read it is a mistake, not a silent no-op. The unguarded mechanism
behind them is `mark()`, used internally by the remote and vLLM-serve backends to
push a result home after a trace has already exited.

## Lifecycle (NDIF round-trip)

1. **Client.** `model.trace(..., remote=True)` builds a `RemoteBackend`
   (`remotable.py:52`). On `__exit__`, `RequestModel.serialize(tracer)` reduces the
   block (`reduce_block`) and `dumps((tracer, interventions))`.
2. **Wire.** `RemoteBackend._post` POSTs the JSON envelope (`request.metadata()`)
   plus the payload blob (optionally zstd-compressed) to `{host}/request`.
3. **Server.** `RequestModel.deserialize(blob, persistent_objects)` runs `loads`;
   `_load_function` recompiles each captured function against the server's Python;
   persistent IDs resolve to the server's live model/tokenizer/module tree.
4. **Execute.** The reconstructed tracer runs against the server-side model; saved
   values are collected.
5. **Return.** Saves are `torch.save`-pickled, zstd-compressed, uploaded, and the
   presigned URL is returned in a `COMPLETED` `ResponseModel`. The client downloads
   and `torch.load`s them; `RemoteBackend._push` writes them into the caller's
   `x = ...save()` variables.

`LocalSimulationBackend` runs steps 1, 3, and 4 in-process.

## Key files

- `src/nnsight/intervention/serialization.py` — `code_reduce` (`:147`),
  `reduce_block` (`:176`), `_lambda_source` (`:56`), `_load_function` (`:191`),
  `_fill` (`:243`), `CustomCloudPickler` (`:253`), `CustomCloudUnpickler` (`:312`),
  `dumps`/`loads` (`:328`/`:349`)
- `src/nnsight/schema/request.py:15` — `RequestModel` (`serialize`/`deserialize`)
- `src/nnsight/schema/response.py` — `Status` (`:13`), `ResponseModel` (`:25`)
- `src/nnsight/tracing/util.py:82` — `SerializedFrame`
- `src/nnsight/intervention/interleaver.py:212` — `Mediator.__getstate__`
- `src/nnsight/intervention/backends/local.py:37` — `LocalSimulationBackend`
- `src/nnsight/_c/py_mount.c` + `setup.py` — the pymount extension

## Related

- [backends.md](./backends.md) — how `RemoteBackend`/`AsyncRemoteBackend` use these payloads
- [vllm-integration.md](./vllm-integration.md) — mediators serialized across process boundaries
- `tests/test_serialization.py`, `tests/test_ndif.py`, `tests/test_editing.py::TestEditSerialization` — the reference suites
