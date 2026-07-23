---
title: Developing NNsight
one_liner: Internals reference for contributors and agents extending nnsight.
tags: [internals, dev]
related: [docs/developing/architecture-overview.md, docs/developing/tracing-pipeline.md, docs/developing/interleaver-internals.md]
sources: []
---

# Developing NNsight

This folder is the internals reference for nnsight. It sits one level below the
user-facing `docs/usage/`, `docs/concepts/`, and `docs/models/` folders: docs here
cite source `file:line`, describe data flow between subsystems, and explain
extension points for custom tracers, backends, and model runtimes.

Audience: nnsight contributors and AI agents whose users want to debug or extend
the library. If you are looking for "how do I run a trace," start in
`docs/usage/`; for the mental models behind the machinery, start in
`docs/concepts/`.

## The one-paragraph model

`with model.trace(x): ...` does **not** run its body inline. A `Tracer` captures
the block's source, compiles the body to a standalone code object, and — on
`__exit__` — runs it *interleaved* with the model's forward pass. The body runs in
a **greenlet** (a `Mediator`); it parks whenever it reads or writes an activation
(`model.layer.output`), the model runs until a forward hook reaches that location,
the value is handed to the worker (edited on the way back if the worker wrote to
it), and the worker resumes. One `Interleaver` owns the hooks and the workers; a
`Backend` decides what "run the block" means (execute locally, ship to NDIF, store
as an edit).

## What this covers

- **Tracing** (`src/nnsight/tracing/`) — capture a `with` block's source, parse
  and compile the body, run it through a backend, and push results back into the
  caller's frame. Model-agnostic; no torch.
- **Interleaving** (`src/nnsight/intervention/interleaver.py`) — greenlet workers
  (`Mediator`) coordinated with the model's forward pass via PyTorch hooks
  (`Interleaver`), using the `Event` protocol (`VALUE`/`SWAP`/`SKIP`/`BARRIER`).
- **Envoy** (`src/nnsight/intervention/envoy.py`) — the module proxy that exposes
  `.input`/`.output`/`.source`/`.skip` and drives `interleave()`.
- **Batching** (`src/nnsight/intervention/batching.py`) — combine several
  `tracer.invoke(...)` inputs into one forward and scope each block to its rows.
- **Backends** (`src/nnsight/tracing/backend.py`,
  `src/nnsight/intervention/backends/`) — local, remote (NDIF), local-simulation.

## Table of contents

### Big picture

- `docs/developing/architecture-overview.md` — top-down map: Tracer → Backend →
  Interleaver → Mediator (greenlet) → PyTorch hooks → Envoy.
- `docs/developing/tracing-pipeline.md` — capture → parse → build → compile →
  execute, `Scope`, the skip hook, the per-site cache.

### Interleaver

- `docs/developing/interleaver-internals.md` — `Interleaver`, `Mediator`, the
  greenlet park/switch dance, `Event`, `handle()` fan-out, iteration tagging.
- `docs/developing/hook-system.md` — how forward hooks are installed at instrument
  time and pass through when idle; the source/skip controller.
- `docs/developing/source-internals.md` — `.source` operation-level
  access via AST-instrumented forwards.

### Batching

- `docs/developing/batching-internals.md` — `Batcher` add/narrow/widen, batch
  groups, `gather_skip`/`assemble_skip`.

### Backends

- `docs/developing/backends.md` — the `Backend` base and the existing backends
  (`RemoteBackend`, `AsyncRemoteBackend`, `LocalSimulationBackend`), and how a
  model selects one from `remote=`.
- `docs/developing/adding-a-new-backend.md` — recipe for subclassing `Backend`.

### Extending

- `docs/developing/extending-envoy.md` — the extension surface for new hookable
  values: an `eproperty` descriptor on a model/runtime subclass, served from the
  driver with its `.provide`.
- `docs/developing/adding-a-new-runtime.md` — plug a new model type or inference
  engine in via the modeling mixins.
- `docs/developing/serialization.md` — source-based serialization for remote runs.

### Reference

- `docs/developing/performance.md` — where overhead lives and how to measure it.
- `docs/developing/testing.md` — running the suite offline; what each test covers.
- `docs/developing/contributing.md` — house style, branch/commit conventions.

### Related concept docs

The `docs/concepts/` folder holds the shorter, mental-model versions of several of
these topics — `deferred-execution.md`, `threading-and-mediators.md`,
`interleaver-and-hooks.md`, `batching-and-invokers.md`, `source-tracing.md`,
`envoy.md`. Read those first if you want the "why" before the "how".

## Related

- `docs/concepts/index.md` — mental models for the same machinery.
- `docs/models/index.md` — the model classes (`NNsight`, `TransformersModel`,
  `DiffusionModel`, `VLLM`) that wrap the intervention layer.
