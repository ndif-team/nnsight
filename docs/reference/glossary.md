---
title: Glossary
one_liner: Alphabetical index of nnsight-specific terms.
tags: [reference, glossary]
---

# Glossary

Short definitions of terms used throughout the nnsight codebase and docs. For deeper architectural context, see [`NNsight.md`](https://github.com/ndif-team/nnsight/blob/main/NNsight.md).

## Backend

A pluggable execution target for a `Tracer`. The backend receives the compiled intervention function plus a `Tracer` and is responsible for actually running it — locally on the wrapped model, on NDIF (`RemoteBackend`), or against a vLLM async loop (`AsyncVLLMBackend`). See [../developing/backends.md](../developing/backends.md).

## Batcher / Batch group

The `Batcher` class is responsible for stacking the inputs of multiple invokes into a single batched forward pass and slicing values back out for each invoke. A **batch group** is a `(start, size)` pair stored on each `Mediator`; `narrow(group)` extracts that mediator's slice from the full batched tensor and `swap(group, value)` writes a new value into the slice. `LanguageModel`, `VLLM`, and `DiffusionModel` each provide their own subclass (`VLLMBatcher`, `DiffusionBatcher`, …). See [../concepts/batching-and-invokers.md](../concepts/batching-and-invokers.md).

## eproperty

A custom descriptor (`nnsight.intervention.interleaver.eproperty`) used to define hookable properties on `IEnvoy` objects. Each `eproperty` issues a blocking `request()` to the interleaver on `__get__` and a `swap()` on `__set__`. The decorated *stub method* is never executed for its return value — its purpose is to carry pre-setup decorators (e.g. `@requires_output`) and donate its `__name__`/`__doc__`. The descriptor also supports `preprocess`, `postprocess`, and `transform` hooks for custom subclasses. See [../concepts/envoy-and-eproperty.md](../concepts/envoy-and-eproperty.md).

## Envoy

The user-facing proxy class (`nnsight.intervention.envoy.Envoy`) that wraps a single `torch.nn.Module`. Provides `.output`, `.input`, `.inputs`, `.source`, `.skip()`, ad-hoc `__call__()`, and transparent attribute delegation to the underlying module. The Envoy tree mirrors the model's module hierarchy and is built eagerly at construction. See [../concepts/envoy-and-eproperty.md](../concepts/envoy-and-eproperty.md).

## IEnvoy

A `runtime_checkable` `Protocol` that any object using `eproperty` descriptors must satisfy: it must expose an `interleaver` attribute and a `path` string. Both `Envoy` and `OperationEnvoy` satisfy `IEnvoy`. See [../concepts/envoy-and-eproperty.md](../concepts/envoy-and-eproperty.md).

## Interleaver

The orchestrator class (`nnsight.intervention.interleaver.Interleaver`) that runs on the **main thread** alongside the model forward pass. It wraps modules with a skippable forward and a sentinel hook, manages the set of active `Mediator`s, tracks per-provider iteration counts, and brokers `wrap_operation` value broadcasts. See [../concepts/interleaver-and-hooks.md](../concepts/interleaver-and-hooks.md).

## Invoker / Invoke

`Invoker` is the context manager returned by `tracer.invoke(...)`. Each "invoke" defines a separate intervention function whose code is captured and compiled into a `Mediator`. Invokes execute serially in definition order, each on its own worker thread. An empty `tracer.invoke()` (no positional args) is special — it operates on the full batch produced by previous input invokes and does not trigger `_batch()`. See [../usage/invoke-and-batching.md](../usage/invoke-and-batching.md).

## Iteration tracker

A counter that disambiguates multiple calls to the same module within a single trace (common in multi-token generation). The `Interleaver` increments a counter per provider string and appends `.i0`, `.i1`, … to provider strings (`model.transformer.h.0.output.i2` = third call to layer 0). The `Mediator` maintains a parallel `iteration` cursor that selects which call to wait for; `tracer.iter[...]`, `tracer.next()`, and `tracer.all()` move it. See [../usage/generate.md](../usage/generate.md).

## Lazy hook

The hook-execution model nnsight uses since the refactor/transform branch. Modules are not given permanent input/output hooks; instead, a sentinel forward hook is installed once (so PyTorch always takes the hook-dispatch path), and **one-shot hooks** are registered on demand by each mediator when intervention code first reads `.output`/`.input`. Modules no mediator touches incur effectively zero hook overhead. See [../concepts/interleaver-and-hooks.md](../concepts/interleaver-and-hooks.md).

## Mediator

The per-invoke object (`nnsight.intervention.interleaver.Mediator`) that runs intervention code on a **worker thread** and communicates with the `Interleaver` via single-item `event_queue` and `response_queue` queues. Owns its `batch_group`, iteration cursor, history of seen providers (for out-of-order detection), and any pending `transform` callback. See [../concepts/interleaver-and-hooks.md](../concepts/interleaver-and-hooks.md).

## Mediator events (VALUE / SWAP / SKIP / BARRIER / END / EXCEPTION)

The event vocabulary the worker thread sends to the main thread:

- `VALUE` — request the value at a provider string (e.g. `model.layer.output.i0`).
- `SWAP` — replace the value at a provider with a new one (used by `eproperty.__set__`).
- `SKIP` — bypass a module's forward, returning a replacement.
- `BARRIER` — wait at a `tracer.barrier(n)` synchronization point.
- `END` — intervention function finished normally.
- `EXCEPTION` — intervention function raised; the exception is forwarded to the main thread.

See [../concepts/interleaver-and-hooks.md](../concepts/interleaver-and-hooks.md).

## One-shot hook vs persistent hook

A **one-shot** hook self-removes after firing once; nnsight uses these for `.output`/`.input` access and for source-tracing operation hooks. A **persistent** hook stays registered for the full trace; nnsight uses these for `tracer.cache(...)` (must persist across all generation steps) and for the sentinel output hook installed by `wrap_module`. See `nnsight/intervention/hooks.py`.

## Persistent object (serialization)

An object marked as **persistent** is **not pickled by value**. Instead, when nnsight's pickler hits it, only an opaque ID (a "stub" / key / string) is written into the byte stream. On the receiving side, the unpickler looks the ID up in a dict it was given at construction time and substitutes the **actual** object that lives in that process.

The mechanism is built on Python's standard `pickle` `persistent_id` / `persistent_load` protocol:

- **On the sender:** `CustomCloudPickler.persistent_id(obj)` (`src/nnsight/intervention/serialization.py:888`) returns `obj.__dict__["_persistent_id"]` if present. That string is written into the serialized stream in place of the object's contents.
- **On the receiver:** `CustomCloudUnpickler` is constructed with a `persistent_objects: dict[str, Any]` mapping IDs to concrete objects (`serialization.py:946`). When `persistent_load(pid)` is called by pickle for a persistent reference, it looks up `pid` in that dict and returns the matching object. Unknown IDs raise `pickle.UnpicklingError`.

So the meaning of "persistent" is: "I know this object already lives on the remote side and I want it swapped in for my ID — don't ship its bytes, just ship the ID."

The main real-world use case is **NDIF**: the actual `nn.Module`s of the deployed model already exist on the NDIF server. When a user submits a trace, nnsight does not re-pickle the entire model into the request — it tags those modules as persistent, ships only their IDs, and NDIF's unpickler swaps in the live model modules on its end. The same pattern applies to anything else that "already exists" on the server: shared caches, buffers, registered helper objects.

To mark an object as persistent, set `obj.__dict__["_persistent_id"] = "<some-id>"`. To resolve them on the other side, pass `loads(data, persistent_objects={"<some-id>": real_obj, ...})`.

See [../developing/serialization.md](../developing/serialization.md) for the full pickling pipeline (also covers source-based function serialization for cross-Python-version compatibility, which is a separate but related topic).

## Pymount

A C extension (`src/nnsight/_c/py_mount.c`) that injects methods directly into CPython's `PyBaseObject_Type.tp_dict` so that **every** Python object has a `.save()` (and `.stop()`) method while a trace is active. This enables the legacy `tensor.save()` / `[1, 2, 3].save()` syntax inherited from nnsight 0.4. As of v0.6, pymount is mounted once at import and never unmounted (no per-trace `PyType_Modified` overhead). Disable via `CONFIG.APP.PYMOUNT = False` and use `nnsight.save(obj)` instead. See [../usage/save.md](../usage/save.md).

## Requester / Provider strings

A pair of strings used to match worker requests to model values. The **provider string** is built by the model's hook when a module fires (e.g. `model.transformer.h.0.output.i0`). The **requester string** is built by an `eproperty` when intervention code accesses a value. The `Mediator.handle()` method matches them and either delivers the value, defers the request, or raises `OutOfOrderError`. See [../concepts/interleaver-and-hooks.md](../concepts/interleaver-and-hooks.md).

## Sentinel hook

The empty `register_forward_hook` that `Interleaver.wrap_module` installs on every wrapped module. It returns `output` unchanged. Its purpose is purely structural: PyTorch's `Module.__call__` fast-paths around hook dispatch when no hooks are registered, so the sentinel ensures the hook-dispatch path is always taken — letting one-shot hooks register dynamically and still fire. See [../concepts/interleaver-and-hooks.md](../concepts/interleaver-and-hooks.md).

## Source tracing / SourceAccessor / OperationAccessor

Source tracing (`module.source`) lets you intervene on **intermediate operations** inside a module's forward, not just its boundaries. nnsight parses the forward's AST, wraps every call with `wrap_operation`, and replaces the forward with the instrumented version. A `SourceAccessor` is the global, per-fn cache of the rewritten code and the per-call-site `OperationAccessor`s. Each `OperationAccessor` owns the live hook lists (`pre_hooks`, `post_hooks`, `fn_hooks`, `fn_replacement`) for one operation across all Envoys / Interleavers that ever touched it. See [../usage/source.md](../usage/source.md).

## SourceEnvoy / OperationEnvoy

User-facing proxies for source tracing. A `SourceEnvoy` is what `module.source` returns — printing it shows the forward with operation names highlighted. Each operation in the forward is exposed as an `OperationEnvoy` with the same `.input`, `.inputs`, `.output`, and `.source` interface as a regular `Envoy`. See [../usage/source.md](../usage/source.md).

## Trace / Tracer

A **trace** is a single `with model.trace(...):` block — one execution of the model with a set of interventions captured and run alongside it. A **`Tracer`** is the class that orchestrates capture → parse → compile → execute for that block. Subclasses include `InterleavingTracer` (the default), `BackwardsTracer` (for `tensor.backward()`), `ScanningTracer` (for `model.scan`), and `EditingTracer` (for `model.edit`). See [../concepts/deferred-execution.md](../concepts/deferred-execution.md).

## Tracing context

A `with` block whose context manager is one of `model.trace`, `model.generate`, `model.scan`, `model.session`, `model.edit`, or `tensor.backward()`. Inside any tracing context, intervention code is captured (not executed inline), and `.save()` / `nnsight.save()` is required to persist values past the context boundary.

## Worker thread

Each invoke runs its compiled intervention function on its own `threading.Thread`. The main thread (running the model) and the worker thread communicate via two single-item queues, creating a strict ping-pong execution pattern: only one thread runs at a time, eliminating the need for explicit data locks. See [../concepts/interleaver-and-hooks.md](../concepts/interleaver-and-hooks.md).

## WrapperModule

A trivial `torch.nn.Module` subclass (`nnsight.util.WrapperModule`) whose forward is `lambda x: x`. nnsight uses `WrapperModule` instances as hook anchors — extra modules added to a model so that values produced outside the natural module boundaries (a HuggingFace `.generate()` return, a vLLM logit tensor, a sampled token id) can be exposed as `.output` of a regular `Envoy`. Notable instances:

- `model.generator` (on `LanguageModel`) — wraps the final output of HuggingFace `.generate()`.
- `model.generator.streamer` (on `LanguageModel`) — fires per-token during generation.
- `model.logits` (on `VLLM`) — exposes per-step logit tensors.
- `model.samples` (on `VLLM`) — exposes per-step sampled token ids.

See [../models/](../models/).
