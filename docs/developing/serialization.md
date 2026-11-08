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
references) and persistent ids (the model's objects written as names the receiver
resolves on its own tree, instead of pickling them).

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
- `persistent_id(obj)` (`serialization.py`) writes the model's objects as
  names when the pickler was built with `tree=<any envoy of the model>`: the
  root's `_persistent_attachments()` by identity (`"Interleaver"`, a tokenizer,
  a pipeline), `f"Module:{path}"` for a module the root's interleaver registers,
  `f"Envoy:{path}"` for an `Envoy` of that tree (one whose `interleaver` is the
  root's) that `_state_below()` is false for, and `None` (by value) for one that
  it is true for. An `Envoy` of any other tree, or any `Envoy` with no `tree=`,
  raises `pickle.PicklingError`.
- `reducer_override(obj)` (`serialization.py`) writes the envoy passed as `tree=`,
  when it holds no state, as its name plus one field of state (see
  [the envoy running the payload](#envoys-by-reference)).
- `CustomCloudUnpickler(pickle.Unpickler)` (`serialization.py`) overrides
  `persistent_load(pid)` to resolve the ID from `self.persistent_objects`, with a
  per-payload copy for an `Envoy:<path>`, raising `UnknownPersistentIdError` if the
  ID is absent.

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
| `env: dict` | per-request environment the server applies to the model before running, from the model's `_remoteable_get_env()`: `TransformersModel` sends `{"attn_implementation": <resolved name>}` on every request, plus `"peft": <adapter repo id>` when an adapter is loaded, and with it `"adapters_enabled": False` when it is switched off (the server switches the adapter on for a request without the key) |

- `serialize(tracer, compress=False)` (`request.py`, classmethod) builds the
  binary execution payload: `variables = dict(tracer.info.frame.f_locals)`,
  `interventions = reduce_block(tracer.node, glbls, variables)`, then
  `dumps((tracer, interventions), tree=tracer.envoy)`, zstd level 6 if `compress`.
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

## Persistent objects — model/tokenizer by reference

Heavy server-side objects (the model's modules, its tokenizer and other attachments,
the interleaver) are never serialized: they're written as names and re-bound to the
server actor's live objects.

- **Naming (client):** the pickler is built with `tree=<any envoy of the model>`
  (`RequestModel.serialize` passes `tracer.envoy`, which a trace and a session
  both carry; `VLLM._attach_mediators` and the edit registration pass the
  model). `Envoy._persistent_attachments()` (`envoy.py`) names the interleaver,
  and `TransformersModel` adds its attachments from the
  `_PERSISTENT` map (`transformers.py`): `tokenizer → "Tokenizer"`, `processor
  → "Processor"`, `image_processor → "ImageProcessor"`, `feature_extractor →
  "FeatureExtractor"`, `pipeline → "Pipeline"`; `DiffusionModel` and `VLLM`
  add theirs. A module of the tree is `Module:<path>`, from the interleaver's
  registry, and an envoy is `Envoy:<path>` (below). Nothing is stamped on the
  objects: a tokenizer a block captured on its own (`tok = model.tokenizer`
  above the trace) is found by identity.
- **Resolving (server):** `Remotable._remoteable_persistent_objects()`
  (`remotable.py`) builds the `{name: object}` map from the same definitions:
  the `_persistent_attachments()`, and `Module:<path>` and `Envoy:<path>` for
  every envoy in `self.modules()`. This
  map is passed to `loads(..., persistent_objects=...)`; a missing name raises
  `UnknownPersistentIdError`.

### Envoys by reference

The receiver wraps the same model in the same tree, so a path identifies an envoy,
and an envoy of the tree is written as `Envoy:<path>`. The interleaver identifies
the tree. An envoy of another tree (a second model in scope) raises
`pickle.PicklingError` when the payload is written, and so does any envoy written
with no `tree=`, since the receiver cannot resolve it.

**What goes by value.** An envoy holding state the receiver's tree lacks is pickled
by value, and so is every envoy above it that pickle reaches. A block reaches an
envoy from the one it names by attribute access, and an envoy sent by path resolves
its descendants on the receiver's tree. `Envoy._carries_state()` is true for an
envoy with any of:

- non-empty `_edits`;
- a `rename=` alias bound on it, since the block's source names the alias;
- a child class a caller's `envoys=` map chose. A map a runtime sets on both sides,
  vLLM's `parallel_envoys()` or tensor parallelism's `tp_envoys()`, is a
  `RuntimeEnvoys` and does not count;
- a root class from outside the `nnsight` package, such as a user's subclass of
  `TransformersModel`. A third-party class a receiver also builds goes by value
  too, at the cost of a larger payload.

`Envoy._state_below()` checks an envoy and its descendants, so a stateless sibling
subtree still goes by path. `rename={"attn": "self_attn"}`, bound on each block,
takes the root, `transformer`, `h` and each block by value and leaves every block's
children by path. The pickler checks the envoy it is given, so an `edit()` copy,
which shares its module with the envoy registered at its path, goes by value with its
edits. When no envoy
registered on the tree carries state, one flat check per pickler sends every
registered envoy by path.

A by-value envoy's `__getstate__` is its `__dict__` minus the parent link, and
`__setstate__` links every child to it, so `.parent` and `.root` from a child reach
the by-value envoy above it. An `edit()` copy made before dispatch writes the current
state of the envoy registered at its path, with its own edits, because dispatch
replaced its module and pipeline. A class defined in a script, or in a module
registered for by-value pickling (`nnsight.register`, or `pull_env()` for local
modules), travels by source, `eproperty` descriptors included (`eproperty.__reduce__`).
A class from any other module is written by import path, and a receiver without that
module raises `ModuleNotFoundError`.

**What a payload receives.** `Envoy:<path>` resolves to a copy made for that payload
(`CustomCloudUnpickler.persistent_load`, `_PayloadCopies`). The receiver's tree serves
every payload, and an edit or a plain attribute write through an envoy stays with that
payload. A mounted `torch.nn.Module` goes on the receiver's module and reaches later
payloads, unless the tree refuses it (below). A copy has the same class and path, so it reads the
same values (an eproperty locates a value by path). Its `_module` is the receiver's
`Module:<path>` entry, which a server may have wrapped to guard its model. Its
`_edits` start empty, so an edit
stored on the receiver's envoy stays out of the payload, and its dicts (`_aliases`,
`kwargs`) are its own.

Children are copied one level at a time (`Envoy._materialize`), on the first access to
an envoy above or below the copy, so a block touching one layer copies only the
envoys on its path and their siblings. The table of copies holds them weakly, which
leaves no reference cycle, so a payload is freed without a GC pass. A copy reached
without its parent, such as `layer = model.transformer.h[3]` captured above the block,
holds its parent's copy strongly. Reading `.root` from such a child makes a small
cycle of envoys that a GC pass frees; the payload's tensors are outside it.

**The envoy running the payload.** The envoy passed as `tree=` is written by
`reducer_override` when it holds no state: the callable `_named_envoy`, the argument
`Envoy:<path>`, and the state `{"interleaver": ...}`, whose value is written as the
name `Interleaver`. Pickle resolves the envoy and then calls `__setstate__` with that
state. The receiver's class then does its setup (`Meta.__setstate__` sets
`dispatched`), and the envoy uses the interleaver the receiver resolves for this
request. A wrapper's `interleave` depends on both.

A receiver's unpickler decides what a name resolves to. NDIF's sandbox runner
(`IPCCloudUnpickler`) returns the map's entry with no copy, which is the runner's meta
model, and a new `IPCInterleaver` for `Interleaver`, so the payload's interleaver
replaces the one on that meta model. This is safe while each request has its own
runner process. `__setstate__` links the envoy's direct children, so with
`CustomCloudUnpickler` their copies are made at load, and a module of that level
missing from the map raises `UnknownPersistentIdError` during load.

**Attributes.** A child envoy takes no new public attribute (`Envoy.__setattr__`),
because the receiver's envoy at that path would not have it. A value a block needs
goes in a variable of the block. A `torch.nn.Module` may still be attached, and one
attached above the block does not travel: by path the receiver's envoy lacks it
(`AttributeError` when the block reads it), and by value its name is missing from the
map (`UnknownPersistentIdError` at load). A root
accepts new attributes, since a model wrapper sets its own in `_load`. A new attribute
on a stock root is not sent, so reading it on a receiver raises `AttributeError`.

On gpt2 (166 envoys) the root is 118 B by path and a whole trace payload is about
780 B. An alias on each block takes that payload to 9.4 KB.

A tree that serves many requests at once (vLLM's model runner) marks its interleaver
`shared = True`, and `Envoy._add_envoy` refuses a mount on it, because a copy holds the
receiver's module and every later request would see the new one. A plain value written
on a root, or on an attribute the envoy already has, lands on the payload's copy, so
`shared` does not refuse it. `Envoy.__setattr__` refuses a new public attribute on a
child on every tree.

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
   block (`reduce_block`) and `dumps((tracer, interventions), tree=tracer.envoy)`.
2. **Wire.** `RemoteBackend._post` POSTs the JSON envelope (`request.metadata()`)
   plus the payload blob (optionally zstd-compressed) to `{host}/request`.
3. **Server.** `RequestModel.deserialize(blob, persistent_objects)` runs `loads`;
   `_load_function` recompiles each captured function against the server's Python;
   persistent IDs resolve to the server's live modules and tokenizer, and each
   `Envoy:<path>` to a copy of the server's envoy made for this payload.
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
