---
title: Glossary
one_liner: Alphabetical index of nnsight-specific terms.
tags: [reference, glossary]
---

# Glossary

Short definitions of terms used throughout the nnsight codebase and docs.

## Backend

What actually runs a captured `with` block on `Tracer.__exit__`. The default `Backend` (`src/nnsight/tracing/backend.py`) runs it in place; `RemoteBackend` / `AsyncRemoteBackend` ship it to NDIF; `LocalSimulationBackend` serializes/deserializes it and runs it in-process (a `remote="local"` dry run). Passed via the `backend=` kwarg or selected by `remote=`.

## Batcher / batch group

The `Batcher` (`src/nnsight/intervention/batching.py`) combines several invokes' inputs into one batched forward and slices values back out per invoke. A **batch group** is a `[start, size]` row range stored on each `Mediator`; `narrow(value, group)` extracts that worker's rows from a batched tensor and `widen(...)` / `gather_skip(...)` splice an edit or skip back in. Models that batch (`TransformersModel`, `VLLM`, `DiffusionModel`) supply the input-assembly logic via `_batch_size` / `_batch`.

## Dispatch

Loading a model's real weights. A model built without `dispatch=True` sits on the **meta device** (structure only, no weights); the weights load lazily the first time a trace actually runs (`Meta.dispatch`). `scan` deliberately does not dispatch — it runs under fake tensors.

## Envoy

The user-facing proxy (`src/nnsight/intervention/envoy.py`) wrapping a single `torch.nn.Module`. The envoy tree mirrors the model's module hierarchy, reachable by the same attribute paths (`model.transformer.h[0].mlp`). Exposes `.input`, `.inputs`, `.output`, `.source`, `.skip()`, ad-hoc `__call__()`, and delegates unknown attributes to the underlying module. `NNsight` and the model wrappers are `Envoy` subclasses.

## eproperty

The descriptor (`src/nnsight/intervention/eproperty.py`) behind a served value: reading it parks the worker until the model reaches `"{host.path}.{key}"`, writing it fires a `SWAP`. `Envoy.input` / `.inputs` / `.output`, `tracer.result`, and `VLLM.logits` / `.samples` are all eproperties, and you can define your own on a model subclass. See [docs/developing/extending-envoy.md](../developing/extending-envoy.md) for the decorator, its `preprocess` / `postprocess` / `transform` / `provide` callbacks, and `key` sharing.

## Event (VALUE / SWAP / SKIP / BARRIER)

The `Event` enum (`src/nnsight/intervention/interleaver.py`) — what a parked worker is asking its parent for:

- `VALUE` — read the value at a location: `(VALUE, location)`.
- `SWAP` — replace the value at a location: `(SWAP, location, value)`.
- `SKIP` — bypass a gated computation, using a replacement: `(SKIP, location, value)`.
- `BARRIER` — wait on the other blocks, not on the model: `(BARRIER, None)`.


## greenlet

A cooperative, single-threaded coroutine (`greenlet` package). nnsight interleaves intervention code and the forward pass with greenlets, **not** OS threads: each block runs in its own worker greenlet that switches control back to the model side whenever it parks on a location. Because only one greenlet runs at a time, there are no locks or queues.

## Interleaver

The model-side driver (`Interleaver`, `src/nnsight/intervention/interleaver.py`). Installs the forward pre/post hooks on every module (via `instrument`), holds the list of `Mediator` workers, and — as the forward reaches each location — calls `handle(location, value)`, offering the value to every parked worker and returning the possibly-edited value back into the run. One interleaver is shared across an envoy tree.

## Invoker / invoke

`Invoker` (`src/nnsight/intervention/tracer.py`) is the context manager from `tracer.invoke(...)`. Each `with tracer.invoke(x):` contributes its input as one batch group and its body runs as a worker scoped to those rows. An empty `tracer.invoke()` sees the whole batch. Invokes are collected first, then their inputs are combined into a single batched forward.

## Iteration / occurrence

A location can be reached many times in one run (each step of a generation loop). Each visit is an **occurrence**, tagged `.i0`, `.i1`, … The `Mediator` tracks a per-location count and an `iteration` cursor selecting which occurrence a request binds to; `tracer.iter[...]` / `tracer.all()` move that cursor. Requesting a location out of order raises `OutOfOrderError`, as does a loop that names an end the run never reaches; an open `iter[:]` / `all()` warns there instead, since outrunning the model is how it ends.

## Location / provider string

The string that names a value in a run — a module path plus a handle (`"model.transformer.h.0.output"`, `"model.transformer.h.0.input"`), the run's `"result"`, or a model-specific one (`"logits"`, `"samples"`). Occurrence-tagged `.i{n}` when matched. The interleaver's hook fires a location; a worker parks on the one it wants; `Mediator.handle` matches them.

## Mediator

The per-block worker object (`Mediator`, `src/nnsight/intervention/interleaver.py`). Wraps one captured block (a trace/invoke body, or a stored edit) and runs it in a greenlet, parking on a location whenever the block reads or writes one. Owns its `batch_group`, iteration cursor, per-location counts, and any `tracer.cache()` caches. Its classmethods `value` / `swap` / `skip` / `barrier` are the API the `Envoy` properties call to park.

## Meta device

Where an undispatched model lives — module structure with no real weight data (`MetaDevice`, `src/nnsight/modeling/mixins/meta.py`). Lets nnsight build a model's envoy tree and run `scan` without loading gigabytes of weights.

## NNsightDeprecationWarning

The category every nnsight deprecation is raised under (`nnsight.NNsightDeprecationWarning`). A `FutureWarning` rather than a `DeprecationWarning`, so it is shown wherever the deprecated call sits — a script, an imported module, a library — instead of only in `__main__`. Silence nnsight's alone with `warnings.filterwarnings("ignore", category=nnsight.NNsightDeprecationWarning)`.

## OutOfOrderError

Raised when intervention code asks for a location the model already ran past, or one it never reached. Workers must request locations in the order the model produces them. See [docs/errors/out-of-order-error.md](../errors/out-of-order-error.md).

## Persistent object (serialization)

An object marked persistent is **not** pickled by value — only an opaque id goes into the stream, and the receiver swaps in the real object it already holds. It is how a remote request ships a model's modules as ids (`"Module:<path>"`, `"Interleaver"`, ...) instead of re-pickling weights the server already has. See [docs/developing/serialization.md](../developing/serialization.md).

## Scope

The namespace a captured block runs in (`Scope`, `src/nnsight/tracing/util.py`). A `dict` subclass layering three sources: a snapshot of the frame's locals at capture time, the frame's live locals shared with sibling blocks, and the frame's globals. Passed as `exec`'s globals so nested `def`/`lambda` in a block can reach the block's own names, and so values assigned in one invoke are visible to later ones.

## skip

Bypass a module's (or operation's) forward, using a replacement as its output — `envoy.skip(replacement)` or `source_envoy.skip(replacement)`. Implemented as the `SKIP` event against a gate installed on every module's forward.

## source (source tracing)

Operation-level access inside a module's forward. `envoy.source` returns a `Source` (`src/nnsight/intervention/source.py`) that decomposes the forward into named operations `{callable}_{occurrence}` (`fc1_0`, `relu_0`, `relu_1`, ...) and `{target}_{occurrence}` for each assignment (`h_0`). Indexing one (`envoy.source.relu_0`) gives a `SourceEnvoy` with the same `.input` / `.inputs` / `.output` / `.skip` / `.source` interface as an `Envoy` — one level finer. `print(envoy.source)` renders the forward with each op labelled; `.source` on a `SourceEnvoy` drills recursively into a called function. Requesting an op on a forward with no Python source (a builtin) raises `SourceNotAvailable`; decorated forwards are peeled or instrumented as they are.

## Tracer

The class that turns a `with` block into deferred, controlled execution: capture the block's source → parse → compile its body → run it via a backend (`src/nnsight/tracing/tracer.py`). Subclasses: `InterleavingTracer` (`model.trace` / `.generate` / `.pipe`), `ScanningTracer` (`model.scan`, under fake tensors), `EditingTracer` (`model.edit`), `Iterations` (`tracer.iter`), and a plain `Tracer` for a session. `Invoker` is a tracer for one `invoke` block.

## Tracing context

Any `with` block whose context manager is `model.trace` / `generate` / `pipe` / `scan` / `session` / `edit` or `tensor.backward()`. Inside it, the body is captured (not run inline) and executed by a backend; values you want to keep past the block must be marked with `.save()` / `nnsight.save(...)`.

## WrapperModule

A trivial `nn.Module` whose forward returns its input unchanged. Used as a hook anchor to expose a value not produced by a real submodule. Instances: `model.generator` / `model.generator.streamer` on `TransformersModel` (generation output / per-step tokens).
