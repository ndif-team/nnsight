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
recompiles it on the far side, so client and server Pythons need not match. This
covers the source-based pickler, block reduction, the persistent-object reference
protocol (model/tokenizer passed by ID), the `RequestModel` envelope,
`SerializedFrame`, and `Mediator.__getstate__` (how an edit rides to the server).

## Example

`dumps`/`loads` (`src/nnsight/intervention/serialization.py`) are the pickle-style
API:

```python
from nnsight.intervention.serialization import dumps, loads

m = 3
loads(dumps(lambda x: x * m))(5)          # -> 15
```

The round-trip suite is `tests/test_serialization.py`.

## Architecture

Two layered mechanisms: `code_reduce` (capture code as text plus only the names it
references) and persistent ids (`_persistent_id`, reference heavy objects instead
of pickling them).

### `code_reduce` and `reduce_block`

- `code_reduce(source, globals, locals)` (`serialization.py`) is the core
  reducer. It returns `(source, used_globals, used_locals)` — the code as text
  plus **only** the globals/locals that source actually references, found via
  `_referenced_names(source)` (which compiles the source and walks
  `co_names` recursively). Names are looked up by subscript, so a `Scope` globals
  resolves through its fallback chain and module globals (`torch`, etc.) stay out
  of the payload.
- `reduce_block(node, globals, locals)` (`serialization.py`) reduces a
  captured `with`-block AST node. It unparses `node.body` to source, padding with
  leading blank lines so each statement keeps its original line number (correct
  remote tracebacks), then calls `code_reduce`. This is shared by the request
  payload (the traced block) and edit-mediator serialization.

### The pickler / unpickler

- `CustomCloudPickler(cloudpickle.CloudPickler)` (`serialization.py`) —
  overrides `_dynamic_function_reduce(func)` to serialize a dynamic
  `def`/`lambda` by source: `_lambda_source(func)` for lambdas, otherwise
  `textwrap.dedent(inspect.getsource(func))`. Closure cell contents are captured
  via cloudpickle's `_get_cell_contents` and folded into the globals. It returns
  the 6-tuple reduce form `(_load_function, args, globals, None, None, _fill)`. On
  `(OSError, TypeError)` — C functions, dataclass methods, REPL functions with no
  recoverable source — it falls back to cloudpickle's bytecode path, which only
  loads on a matching Python.
- `persistent_id(obj)` (`serialization.py`) returns
  `obj.__dict__["_persistent_id"]` when present; otherwise, when the object's
  class defines `_persistent_reference(memo)`, whatever that returns (`Envoy`
  answers `Envoy:<path>` or `None`, see below); otherwise the id of `obj` in the
  `persistent_objects` map handed to `dumps`, if it is one of its values; else
  `None`.
- `CustomCloudUnpickler(pickle.Unpickler)` (`serialization.py`) overrides
  `persistent_load(pid)` to look up `self.persistent_objects[pid]`, raising
  `UnknownPersistentIdError` if the ID is absent, and to hand the payload what
  the object's class `_persistent_resolve(scope)` returns when it defines one
  (`Envoy` returns a copy made for this payload, see below).

### Reconstruction

- `_load_function(...)` (`serialization.py`) rebuilds a function from a
  `code_reduce` tuple. It compiles the (blank-padded) source and **lifts out the
  function's code object** rather than executing the `def` — so defaults and
  annotations aren't re-evaluated against incomplete globals. It compiles under a
  synthetic filename `f"[{name}:{line}] {origin_filename}"` and registers the
  source in `linecache`, so remote tracebacks and `inspect.getsource` work.
- `_fill(func, globals)` (`serialization.py`) populates the function's globals
  **after** pickle has memoized it, so a self-reference resolves via the memo
  instead of recursing during unpickling. This is how recursive and mutually
  recursive functions reconstruct.

### `_lambda_source` (lambda extraction)

`inspect.getsource` on a lambda returns the whole enclosing line, ambiguous when
several lambdas share it (`f, g = lambda x: x*2, lambda x: x+1`).
`_lambda_source` (`serialization.py`) AST-parses the line and disambiguates by
matching parameter names (`co_varnames[:co_argcount]`), then by source-span
enclosure using `func.__code__.co_positions()`. It handles nested lambdas, lambda
defaults, and multi-line lambdas (wrapping unparseable fragments in a `def`).
`tests/test_serialization.py::TestLambda` is the reference for the edge cases.

### Top-level API

`dump`/`dumps` (`serialization.py`) and `load`/`loads`.
`load`/`loads` take `persistent_objects=None` and an `unpickler=` override so a
server can substitute its own unpickler. `DEFAULT_PROTOCOL = 4`.

## The request envelope — `schema/request.py`

`RequestModel(BaseModel)` (`request.py`) is the JSON routing envelope. The
model itself is never carried — it is identified by `model_key`. Fields:

| Field | Meaning |
|-------|---------|
| `model_key: str` | identifies the server-side model (see `to_model_key()`) |
| `session_id: str` | session grouping |
| `compress: bool` | zstd-compress the payload and ask the server to compress its result |
| `env: dict` | per-request environment the server applies to the model before running, from the model's `_remoteable_get_env()`: `TransformersModel` sends `{"attn_implementation": <resolved name>}` on every request, plus `"peft": <adapter repo id>` when an adapter is loaded |

- `serialize(tracer, compress=False)` (`request.py`, classmethod) builds the
  binary execution payload: `variables = dict(tracer.info.frame.f_locals)`,
  `interventions = reduce_block(tracer.node, glbls, variables)`, then
  `dumps((tracer, interventions))`, zstd level 6 if `compress`.
- `deserialize(blob, persistent_objects=None, compress=False, ...)`
  (`request.py`, staticmethod) is the inverse: `tracer, (source, glbls, vars) =
  loads(...)`, recompiles the block into `tracer.info.code`, registers it in
  `linecache`, and restores `frame.f_globals`/`f_locals`. It returns a
  ready-to-run tracer; the server then calls `tracer.execute(tracer.info.code)`.

The tracer is pickled whole (it carries the model invocation and a
`SerializedFrame`); the block source travels separately as `interventions` and is
re-attached as `tracer.info.code`.

## `SerializedFrame` — `tracing/util.py`

A real frame can't be pickled, so `tracer.info.frame` is a `SerializedFrame`
(`util.py`): a picklable stand-in carrying **only** code metadata
(`co_filename`, `co_firstlineno`, `co_name`) needed for remote tracebacks and
source lookup. `f_locals`/`f_globals` are empty dicts — the real locals/globals
travel in the `interventions` tuple and are written back onto the frame in
`RequestModel.deserialize`. Build one from a live frame with `SerializedFrame.of(frame)`.
The reducers rely on `Scope` iteration yielding only the block's own names
([tracing-pipeline.md](./tracing-pipeline.md#scope-how-a-captured-block-reaches-names)).

## Edits ride to the server — `Mediator.__getstate__`

An `edit` (from `model.edit(...)`) is a `Mediator` stored in `envoy._edits`.
`Mediator.__getstate__` (`interleaver.py`) serializes it cross-version-safely:
it returns `{"reduced": reduce_block(self.node, self.glbls, self.lcls), "copy":
self.copy}` — the edit block reduced to source plus referenced vars, and the
`copy` flag. It **deliberately drops** the compiled `code` and all run state
(`worker`, `pending`, `interleaver`, `batch_group`): those can't and shouldn't
travel. `__setstate__` recompiles the source (under filename `<edit>`)
and re-inits.

## Persistent objects: model, tokenizer and envoys by reference

Heavy server-side objects (the model, its tokenizer, the module tree, the
interleaver) are never serialized. They are referenced by ID and re-bound to the
server actor's live objects. The envoy tree over the model is referenced the same
way: the server wraps the same model in the same tree, so a request names an
envoy by its path instead of shipping the tree with every request.

- **Envoys (client):** `Envoy._persistent_reference(memo)` (`envoy.py`) answers
  `Envoy:<path>` for an envoy that holds nothing the server cannot have, and the
  pickler writes that id in place of the object. An envoy that does hold such
  state, on itself or on any envoy below it, ships by value through
  `__getstate__`, and each of its children is asked in turn. The state that
  counts (`_own_state`): an edit (`_edits`); a custom envoy class (`envoys=`);
  an alias from the user's `rename=` (an alias the tree binds on its own, such
  as `E_output` or `base_model`, the server binds too); and, on a plain `Envoy`,
  any attribute beyond its shape, children and aliases (`envoy.steer = ...`). A
  model class's root owns the attributes it sets. So `model` in a trace travels
  as `Envoy:model`, while the `edited` copy that `model.edit()` returns travels
  as its own `__dict__`, edits included, with its children by reference. The
  check stops at the first state it finds and is memoized per payload.
- **Envoys (server):** `Envoy._persistent_resolve(scope)` resolves `Envoy:<path>`
  to a copy of the server's envoy made for this payload: `_twins()` copies the
  whole tree once per payload (shallow copies sharing the modules and the
  interleaver, with their own `_edits` lists and children, aliases and parents
  pointing at each other). Whatever the request does to an envoy (an in-place
  edit, an attribute, an added module) lands on the copy, not on the envoys the
  server keeps for the next request. The copy costs a few microseconds per
  envoy, with no serialization.
- **The map on the client:** `RequestModel.serialize`, `VLLM.interleave` and the
  vLLM edit registration pass the model's `_remoteable_persistent_objects()` to
  `dumps`, so any of its values the payload reaches (the tokenizer, a module the
  block captured) is written as its id without a tag.
- **Tagging (client):** for an envoy shipped by value, `Envoy.__getstate__`
  (`envoy.py`) sets `interleaver._persistent_id = "Interleaver"` and
  `_module._persistent_id = f"Module:{self.path}"`. `TransformersModel.__getstate__` (`transformers.py`)
  additionally tags each non-`None` preprocessor and the pipeline, using the
  `_PERSISTENT` map (`transformers.py`): `tokenizer → "Tokenizer"`, `processor
  → "Processor"`, `image_processor → "ImageProcessor"`, `feature_extractor →
  "FeatureExtractor"`, `pipeline → "Pipeline"`.
- **Resolving (server):** `Remotable._remoteable_persistent_objects()`
  (`remotable.py`) builds the `{id: object}` map — `{"Interleaver":
  self.interleaver}` plus `Envoy:<path>` and `Module:<path>` for every envoy in
  `self.modules()`. `TransformersModel` extends it (`transformers.py`) with the
  tokenizer and friends. This map is passed to `loads(...,
  persistent_objects=...)`; a missing ID raises `UnknownPersistentIdError: <id>`.
- **Parent links (server):** `Envoy.__setstate__` re-links its children, both
  those that came by value in the same payload and the per-payload copies that
  stand in for references, so `.parent` and `.root` from a referenced child lead
  to the envoy the payload carries.

### Model identity — `to_model_key`

The request names the model rather than serializing it; how that key is built is in
[backends.md](./backends.md#model-identity-to_model_key).

## The response — `schema/response.py`

`ResponseModel` carries a job's `id`, `status` (the `Status` enum in `response.py`),
`description`, and `data`. On `COMPLETED`, the saves blob arrives either inline (a
binary websocket frame) or, depending on its size, as a presigned URL in `data` that
the client downloads; either way it is `torch.load`-ed, since `pickle`/`unpickle` use
`torch.save`/`torch.load` so tensors survive the wire.

## The `remote="local"` dry run

`LocalSimulationBackend` serializes and deserializes in-process with local modules
hidden; see [backends.md](./backends.md#localsimulationbackend--a-serverless-dry-run).
Both it and `RemoteBackend._serialize` call `pull_env()` first, which registers
local, non-installed modules for by-value pickling so the user's own helpers ship.

## `save` and `mark`

`nnsight.save(obj)` and `obj.save()` (the optional C mount — see
[testing.md](./testing.md#the-c-extension-save)) both raise outside a running trace.
The unguarded mechanism behind them is `nnsight.tracing.tracer.mark()`, which the remote and vLLM-serve
backends use to push a result home after a trace has already exited.

## Lifecycle (NDIF round-trip)

1. **Client.** `model.trace(..., remote=True)` builds a `RemoteBackend`
   (`remotable.py`). On `__exit__`, `RequestModel.serialize(tracer)` reduces the
   block (`reduce_block`) and `dumps((tracer, interventions))`.
2. **Wire.** `RemoteBackend._post` POSTs the JSON envelope (`request.metadata()`)
   plus the payload blob (optionally zstd-compressed) to `{host}/request`.
3. **Server.** `RequestModel.deserialize(blob, persistent_objects)` runs `loads`;
   `_load_function` recompiles each captured function against the server's Python;
   persistent IDs resolve to the server's live model/tokenizer/module tree.
4. **Execute.** The reconstructed tracer runs against the server-side model; saved
   values are collected.
5. **Return.** Saves are `torch.save`-pickled, zstd-compressed, and returned with
   `COMPLETED`: inline on the websocket or, depending on size, uploaded and sent as a
   presigned URL the client downloads. The client `torch.load`s them; `RemoteBackend._push` writes them into the caller's
   `x = ...save()` variables.

`LocalSimulationBackend` runs steps 1, 3, and 4 in-process.

## Related

- [vllm-integration.md](./vllm-integration.md) — mediators serialized across process boundaries.
- Reference suites: `tests/test_serialization.py`, `tests/test_ndif.py`,
  `tests/test_editing.py::TestEditSerialization`.
