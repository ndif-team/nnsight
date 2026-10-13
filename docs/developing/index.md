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

Up: [docs/index.md](../index.md) — all sections.

## What this covers

- **Tracing** (`src/nnsight/tracing/`) — capture a `with` block's source, parse
  and compile the body, run it through a backend, and push results back into the
  caller's frame. Model-agnostic; no torch.
- **Interleaving** (`src/nnsight/intervention/interleaver.py`) — greenlet workers
  (`Mediator`) coordinated with the model's forward pass by a controller installed
  as each module's `forward` (`Interleaver`), using the `Event` protocol
  (`VALUE`/`SWAP`/`SKIP`/`BARRIER`).
- **Envoy** (`src/nnsight/intervention/envoy.py`) — the module proxy that exposes
  `.input`/`.output`/`.source`/`.skip` and drives `interleave()`.
- **Batching** (`src/nnsight/intervention/batching.py`) — combine several
  `tracer.invoke(...)` inputs into one forward and scope each block to its rows.
- **Backends** (`src/nnsight/tracing/backend.py`,
  `src/nnsight/intervention/backends/`) — local, remote (NDIF), local-simulation.

## Table of contents

### Big picture

- `docs/developing/architecture-overview.md` — top-down map: Tracer → Backend →
  Interleaver → Mediator (greenlet) → controller → Envoy, and the one trace
  lifecycle walkthrough.
- `docs/developing/tracing-pipeline.md` — capture → parse → build → compile →
  execute, `Scope`, the skip hook, the per-site cache.

### Interleaver

- `docs/developing/interleaver-internals.md` — `Interleaver`, `Mediator`, the
  greenlet park/switch dance, `Event`, `handle()` fan-out, iteration tagging.
- `docs/developing/controller.md` — the per-module controller installed at
  instrument time: the handoff, the skip gate, and the source-instrumented body.
- `docs/developing/source-internals.md` — `.source` operation-level
  access via AST-instrumented forwards.

### Batching

- `docs/developing/batching-internals.md` — `Batcher` add/narrow/widen, batch
  groups, `gather_skip`/`assemble_skip`.

### Distributed values

- `src/nnsight/intervention/fragments.py` (module docstring) — `Fragments`, the one
  seam vLLM and tensor parallelism share for making a sharded value whole before a
  worker sees it; the runtimes' rules live in `modeling/tp/fragments.py` and
  `modeling/vllm/fragments.py`.

### Backends

- `docs/developing/backends.md` — the `Backend` base and the existing backends
  (`RemoteBackend`, `AsyncRemoteBackend`, `LocalSimulationBackend`), how a
  model selects one from `remote=`, and how to add your own.

### Extending

- `docs/developing/extending.md` — subclass `NNsight`/`Envoy`: batching
  (`_batch_size`/`_batch`), attached modules, a custom `Batcher`, `Loadable`, a
  custom tracer.
- `docs/developing/extending-envoy.md` — the extension surface for new hookable
  values: an `eproperty` descriptor on a model/runtime subclass, served from the
  driver with its `.provide`.
- `docs/developing/adding-a-new-runtime.md` — plug a new model type or inference
  engine in via the modeling mixins.
- `docs/developing/serialization.md` — source-based serialization for remote runs.

### Runtimes

- `docs/developing/vllm-integration.md` — the vLLM runtime: model runner, batcher,
  how interventions ride on engine requests.

### Reference

- `docs/developing/performance.md` — where overhead lives and how to measure it.
- `docs/developing/testing.md` — running the suite offline; what each test covers.
- `docs/developing/contributing.md` — house style, branch/commit conventions.

## Related

- `docs/concepts/index.md` — the shorter mental-model versions of these topics; read
  those first for the "why" before the "how".
- `docs/models/index.md` — the model classes that wrap the intervention layer.
