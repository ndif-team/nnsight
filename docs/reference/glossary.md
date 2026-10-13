---
title: Glossary
one_liner: One-to-two-line definitions of nnsight-specific terms (Envoy, Mediator, Interleaver, controller, occurrence, invoke, …), each linked to its home page.
tags: [reference, glossary]
---

# Glossary

**Backend** — what runs a captured block when the `with` exits: in-process by default, on NDIF with `remote=True`, serialized-then-local with `remote="local"`. [developing/backends.md](../developing/backends.md)

**Batch group / rows** — the slice `[start, size]` of the combined batch an invoke owns; values whose leading dim is the batch size are narrowed to it. [concepts/batching-and-invokers.md](../concepts/batching-and-invokers.md)

**Controller** — the forward nnsight installs on every wrapped module. Idle, it calls the real forward; in a trace, it hands the module's input, skip gate and output to the interleaver. [concepts/interleaver-and-controller.md](../concepts/interleaver-and-controller.md)

**Dispatch** — loading real weights. A model built without `dispatch=True` sits on the meta device until the first real trace. [models/transformers-model.md](../models/transformers-model.md)

**Envoy** — the wrapper around one `nn.Module`, mirroring the module tree; exposes `.input`, `.inputs`, `.output`, `.skip()`, `.source`. `NNsight` and the model classes are Envoys. [concepts/envoy.md](../concepts/envoy.md)

**eproperty** — the descriptor behind a served value (`.output`, `tracer.result`, `VLLM.logits`): reading parks until the location is reached, writing swaps it. [developing/extending-envoy.md](../developing/extending-envoy.md)

**Event** — what a parked worker asks for: `VALUE`, `SWAP`, `SKIP` or `BARRIER`. [concepts/threading-and-mediators.md](../concepts/threading-and-mediators.md#the-events)

**greenlet** — a cooperative coroutine on one OS thread; each worker is one, taking turns with the model. [concepts/threading-and-mediators.md](../concepts/threading-and-mediators.md)

**Interleaver** — the one model-side driver per envoy tree. Holds the workers; each controller handoff calls `Interleaver.handle(location, value)`, which offers the value to every worker and returns the possibly-edited value to the forward. [concepts/interleaver-and-controller.md](../concepts/interleaver-and-controller.md)

**Invoke** — `with tracer.invoke(x):` — one input's rows and one worker in a shared batched forward; an empty `tracer.invoke()` sees the whole batch. [usage/invoke-and-batching.md](../usage/invoke-and-batching.md)

**Location** — the string naming a value in a run: `model.transformer.h.0.output`, `result`, `logits`, … A controller hands a location to the interleaver; a worker parks on the one it wants. [concepts/interleaver-and-controller.md](../concepts/interleaver-and-controller.md)

**Mediator** — the runtime object for one block (trace body, invoke, or edit): its greenlet, rows, occurrence cursor and caches. [concepts/threading-and-mediators.md](../concepts/threading-and-mediators.md)

**Meta device** — module structure with no weight data; lets nnsight build the envoy tree and `scan` without loading weights. [usage/scan.md](../usage/scan.md)

**NNsightDeprecationWarning** — the `FutureWarning` subclass every nnsight deprecation is raised under. [migrating-to-0.8.md](migrating-to-0.8.md#deprecated--warn-under-nnsightdeprecationwarning)

**Occurrence** — one visit to a location; a request is tagged `.i0`, `.i1`, … for the visit it wants. `tracer.iter` selects which. [concepts/interleaver-and-controller.md](../concepts/interleaver-and-controller.md#occurrences)

**OutOfOrderError** — a location was requested after the model ran past it, or was never reached. [errors/out-of-order-error.md](../errors/out-of-order-error.md)

**Persistent object** — an object pickled as an id the receiver resolves to its own copy (how remote requests reference modules without shipping weights). [developing/serialization.md](../developing/serialization.md)

**Scope** — the namespace a captured block runs in: a snapshot of the frame's locals, locals shared with sibling blocks, and globals. [concepts/deferred-execution.md](../concepts/deferred-execution.md)

**skip** — `envoy.skip(value)` / `op.skip(value)`: don't run the module or operation; use `value` as its output. [usage/skip.md](../usage/skip.md)

**source** — `envoy.source`: every call and assignment inside a module's forward as an operation with `.input` / `.output` / `.skip`. [concepts/source-tracing.md](../concepts/source-tracing.md)

**Tracer** — turns a `with` block into deferred execution (`model.trace` / `generate` / `pipe` / `scan` / `edit` / `session`, `tracer.iter`, `tracer.invoke`). [concepts/deferred-execution.md](../concepts/deferred-execution.md)

**WrapperModule** — an identity module that lets a value no real submodule produces be served at its `.output` (`model.generator`, `model.generator.streamer`). [usage/generate.md](../usage/generate.md)
