---
title: Serialization Internals
one_liner: Source-based function serialization for cross-Python-version remote execution.
tags: [internals, dev]
related: [docs/developing/backends.md, docs/developing/vllm-integration.md, docs/remote/index.md]
sources: [src/nnsight/intervention/serialization.py:1, src/nnsight/_c/py_mount.c]
---

# Serialization Internals

## What this covers

NDIF can run code submitted from any Python 3.9+ client regardless of the server's Python version. To make that work, NNsight serializes functions by **source code** instead of by bytecode, since bytecode is version-specific. This document covers the custom cloudpickle-based pickler, lambda extraction, the persistent-object reference protocol used to pass model handles by ID, and the `pymount` C extension that backs `obj.save()`.

## Architecture / How it works

### Why source instead of bytecode

Standard `pickle` and `cloudpickle` serialize functions via their `__code__` bytecode. Python's bytecode format changes between minor versions (3.10 vs 3.11 vs 3.12), so a function pickled on a 3.10 client can fail to unpickle on a 3.11 server. NNsight's `CustomCloudPickler` (`src/nnsight/intervention/serialization.py:682`) overrides the dynamic-function reducer to capture **source code** plus metadata. On the receiving side, `make_function` (`src/nnsight/intervention/serialization.py:456`) recompiles that source against the current interpreter.

### `CustomCloudPickler` vs `cloudpickle`

`CustomCloudPickler` extends `cloudpickle.Pickler` and overrides:

- `_dynamic_function_reduce(func)` (`src/nnsight/intervention/serialization.py:733`) — replaces cloudpickle's bytecode-based serialization with source capture.
- `reducer_override(obj)` (`src/nnsight/intervention/serialization.py:703`) — adds a special path for dataclass classes (rebuilds them via `@dataclass` on the receiving end so the generated `__init__`/`__repr__`/`__eq__` regenerate against the target Python version) and for frame objects (`_frame_reduce`).
- `persistent_id(obj)` (`src/nnsight/intervention/serialization.py:888`) — checks for `obj.__dict__["_persistent_id"]` and returns it. Objects with that attribute are referenced by ID instead of serialized.

`CustomCloudUnpickler` (`src/nnsight/intervention/serialization.py:920`) extends the standard `pickle.Unpickler` and overrides `persistent_load(pid)` to look up `pid` in a dict of pre-registered objects (e.g. tokenizer, model proxy on the server side).

### What gets captured per function

Inside `_dynamic_function_reduce`, NNsight pulls:

- **Source.** From `func.__source__` if attached, otherwise `inspect.getsource(func)`. For lambdas, the result is then refined by `_extract_lambda_source` (see below).
- **Metadata.** `__name__`, `__qualname__`, `__module__`, `__doc__`, `__defaults__`, `__kwdefaults__`, `__annotations__`, `co_firstlineno`, and `co_filename`.
- **Globals.** Captured globals are deferred to a state setter (`_source_function_setstate` at `src/nnsight/intervention/serialization.py:630`) so circular references and recursive self-references resolve correctly after pickle has memoized the function object.
- **Closures.** Closure values are extracted from cells. Function-typed closure values are deferred via `__deferred_closure__` to break cycles for mutually recursive local functions.

The 6-tuple returned to pickle is `(make_function, args, state, None, None, _source_function_setstate)`.

### Lambda extraction (`_extract_lambda_source`)

`inspect.getsource()` on a lambda returns the entire enclosing line, which is ambiguous when multiple lambdas live on the same source line:

```python
f, g = lambda x: x*2, lambda x: x+1
```

`_extract_lambda_source` (`src/nnsight/intervention/serialization.py:113`) uses `code.co_positions()` (Python 3.11+) to find the lambda's body column offset, then `tokenize.generate_tokens` to walk the source. It locates the `lambda` token whose body colon is the closest match to `co_positions`'s reported column, then scans forward through tokens at depth 0 to find where the lambda body ends (a comma, closing paren, second `:`, NEWLINE, or ENDMARKER). The extracted span is what gets serialized.

This handles:
- Multiple lambdas on the same line
- Nested lambdas (`lambda x: lambda y: x + y`)
- Lambdas with lambda defaults (`lambda x=lambda: 1: x()`)
- Multi-line lambdas (the result is wrapped in parens to be syntactically valid)

On Python <3.11 (no `co_positions`) or on tokenization failure, the function falls back to the full source string.

### `make_function` reconstructor

`make_function` (`src/nnsight/intervention/serialization.py:456`) recompiles source on the receiving side:

- For **plain functions** — compiles `source` as a module, walks `co_consts` looking for a code object whose `co_name == name`, and constructs a `types.FunctionType` from that code object plus minimal globals.
- For **closures** — wraps the function in a factory:
  ```python
  from __future__ import annotations
  def _seri_factory_(closure_var_1, closure_var_2):
      def my_func(...): ...
      return my_func
  ```
  The factory is `exec`'d with closure values, which gives Python a chance to bind cells the same way it does for normal nested function definitions. The `from __future__ import annotations` line keeps annotation strings unevaluated, since the limited reconstruction-time globals may not contain referenced types.
- For **lambdas** — names are mangled (`<lambda>` is not a valid Python name) by assigning the expression to a temporary `_lambda_result_` and returning it.

After construction, `make_function` restores `__module__`, `__doc__`, `__qualname__`, `__kwdefaults__`, `__annotations__`, attaches `__source__` for re-serialization, calls `code.replace(co_firstlineno=firstlineno)` so tracebacks show the right line, and registers the source with `linecache` so `inspect.getsource(func)` returns the original lines on the deserializing side.

### `_source_function_setstate` (deferred globals + closure)

After pickle creates the function via `make_function` and memoizes it, `_source_function_setstate(func, state)` (`src/nnsight/intervention/serialization.py:630`) runs:

1. Updates `func.__dict__` with custom attributes.
2. Updates `func.__globals__` in place with the full captured globals. **In place** is critical: any reference in those globals to the function itself (recursive self-reference) now resolves to the already-memoized `func` object.
3. Fills in deferred closure cells from `__deferred_closure__`. This is how mutually recursive local functions reconstruct correctly.

### Persistent objects (`_persistent_id`)

Some objects shouldn't be serialized at all — e.g. the tokenizer or model proxy on the NDIF server side. NNsight uses pickle's `persistent_id` mechanism:

- On the sender, `CustomCloudPickler.persistent_id(obj)` (`src/nnsight/intervention/serialization.py:888`) returns `obj.__dict__["_persistent_id"]` if present. Pickle then writes that ID instead of the object.
- On the receiver, `CustomCloudUnpickler.persistent_load(pid)` (`src/nnsight/intervention/serialization.py:957`) looks the ID up in a dict provided to the unpickler.

`LanguageModel` and `VLLM` set `tokenizer._persistent_id = "Tokenizer"` and provide `{"Tokenizer": self.tokenizer}` via `_remoteable_persistent_objects()` (`src/nnsight/modeling/language.py:378`, `src/nnsight/modeling/vllm/vllm.py:469`). When the request lands on the server, the server-side tokenizer resolves the ID instead of attempting to pickle a HuggingFace tokenizer across the wire.

### `dumps` / `loads` API

The high-level wrappers (`src/nnsight/intervention/serialization.py:979` and `src/nnsight/intervention/serialization.py:1028`) match the standard pickle naming. Both accept `bytes` and `Path` / `str` paths interchangeably. `loads(data, persistent_objects=...)` is how NNsight passes the lookup dict for resolving persistent IDs.

`save` and `load` (`src/nnsight/intervention/serialization.py:1083`) are aliases for backwards compatibility with code that imported the older names.

### Pymount (`CONFIG.APP.PYMOUNT`) and `obj.save()`

NNsight supports two ways to mark a value as saved:

- `nnsight.save(obj)` — a plain function that adds `id(obj)` to `Globals.saves`.
- `obj.save()` — a method form, mounted onto Python's base `object` class via the `pymount` C extension.

`pymount` (`src/nnsight/_c/py_mount.c`) edits the C-level method table for the `object` type to add `save` (and `stop`) as methods. This is necessary because Python's `object` type does not allow attribute assignment from Python. Once mounted, every Python value has `.save()` available.

`Globals.enter()` (`src/nnsight/intervention/tracing/globals.py:101`) mounts `Object.save` once at first use if `CONFIG.APP.PYMOUNT` is `True` (the default). `Globals._mounted = True` ensures it stays mounted for the lifetime of the process — unmounting was removed in v0.6 because it triggered expensive `PyType_Modified()` calls that invalidated all type caches every trace. See `0.6.0.md` for the performance reasoning.

**Why prefer `nnsight.save(obj)` in new code:**

1. It works on objects that define their own `.save()` method (e.g. `huggingface_hub` repos), which would shadow the mounted version.
2. It's explicit about which object is being saved.
3. It works even with `CONFIG.APP.PYMOUNT = False`, which some embeddings of NNsight may want for safety reasons.

`obj.save()` continues to work and is widely used. Calling it outside a trace raises `RuntimeError` with a clear message (`src/nnsight/intervention/tracing/globals.py:30`).

### Dataclass handling

Dataclass-generated methods (`__init__`, `__repr__`, etc.) have no inspectable source — `@dataclass` produces them via `exec`. `CustomCloudPickler.reducer_override` (`src/nnsight/intervention/serialization.py:707`) detects dataclass classes and reduces them via `_dataclass_reduce` (`src/nnsight/intervention/serialization.py:362`), which captures field info and `@dataclass` parameters. On the receiving side `_make_dataclass_skeleton` (`src/nnsight/intervention/serialization.py:313`) creates the class shell and re-applies `@dataclass`, regenerating methods against the target Python version. Detection of dataclass-generated methods on individual functions (e.g. when the method is referenced standalone) goes through `_is_dataclass_generated_method` (`src/nnsight/intervention/serialization.py:252`).

### Linecache registration

`_register_source_with_linecache` (`src/nnsight/intervention/serialization.py:73`) merges multiple deserialized sources from the same conceptual file into one linecache entry, padding with blank lines so each function sits at its original line number. Without this, `inspect.getsource(func)` on a deserialized function would fail or return wrong content, and tracebacks would point at empty offsets.

## Key files / classes

- `src/nnsight/intervention/serialization.py:682` — `CustomCloudPickler`
- `src/nnsight/intervention/serialization.py:920` — `CustomCloudUnpickler`
- `src/nnsight/intervention/serialization.py:733` — `_dynamic_function_reduce`
- `src/nnsight/intervention/serialization.py:456` — `make_function`
- `src/nnsight/intervention/serialization.py:113` — `_extract_lambda_source`
- `src/nnsight/intervention/serialization.py:630` — `_source_function_setstate`
- `src/nnsight/intervention/serialization.py:362` — `_dataclass_reduce`
- `src/nnsight/intervention/serialization.py:888` — `persistent_id`
- `src/nnsight/intervention/serialization.py:957` — `persistent_load`
- `src/nnsight/intervention/serialization.py:979` — `dumps`
- `src/nnsight/intervention/serialization.py:1028` — `loads`
- `src/nnsight/_c/py_mount.c` — pymount C extension
- `src/nnsight/intervention/tracing/globals.py:101` — `Globals.enter` (where pymount is invoked)

## Lifecycle (NDIF round-trip)

1. **Client side.** User runs `model.trace(..., remote=True)`. `RemoteBackend.request()` (`src/nnsight/intervention/backends/remote.py:514`) builds a `RequestModel` containing the compiled intervention function plus the tracer state, and `serialize()`s it via `dumps`.
2. **Wire.** `RemoteBackend.submit_request` POSTs the serialized bytes (optionally zstd-compressed) over HTTPS.
3. **Server side.** The NDIF worker calls `RequestModel.deserialize(payload, persistent_objects)` which goes through `loads(...)`. `make_function` recompiles each captured function against the server's Python version. Persistent IDs are resolved against the server-provided dict (tokenizer, model proxy, etc.).
4. **Execution.** The reconstructed function runs against the server-side model. Saved values are collected via `Globals.saves` IDs.
5. **Reverse.** Saved values are pickled (with `torch.save` for tensors), zstd-compressed, and streamed back. The client decompresses and `torch.load`s.

`LocalSimulationBackend` exercises steps 1, 3, and 4 in-process — see [backends.md](./backends.md).

## Extension points

- **Custom persistent objects.** Override `Model._remoteable_persistent_objects()` to return additional `{name: object}` mappings. Set `obj._persistent_id = name` on the object so the pickler tags it. `LanguageModel` does this for the tokenizer (`src/nnsight/modeling/language.py:378`).
- **Allowlist tweaks.** `LocalSimulationBackend.SERVER_MODULES` (`src/nnsight/intervention/backends/local_simulation.py:19`) lists modules assumed to exist on NDIF. To simulate a stricter environment locally, edit this set and run with `remote='local'`.
- **Custom serialization for a new type.** Override `reducer_override(obj)` in a subclass of `CustomCloudPickler`, mirroring how dataclasses are handled. Register the reconstructor via `dispatch_table` if the type is built-in.

## Related

- [backends.md](./backends.md) — how `RemoteBackend` and `LocalSimulationBackend` use `dumps`/`loads`
- [vllm-integration.md](./vllm-integration.md) — how mediators are serialized via `extra_args` on `SamplingParams`
- `tests/test_serialization_edge_cases.py`, `tests/test_lambda_serialization.py`, `tests/test_dataclass_serialization.py`, `tests/test_local_simulation.py` — the reference test suite for this subsystem
