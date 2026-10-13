---
title: Developing NNsight
one_liner: Internals reference for contributors and agents extending nnsight.
tags: [internals, dev]
related: [docs/developing/architecture-overview.md, docs/developing/tracing-pipeline.md, docs/developing/interleaver-internals.md]
sources: []
---

# Developing NNsight

The internals reference for nnsight: source citations, data flow
between subsystems, and how to add tracers, backends, runtimes and served values.
For running a trace, start in [`docs/usage/`](../usage/index.md); for the shorter
mental models behind this machinery, read [`docs/concepts/`](../concepts/index.md)
first; for the model classes, [`docs/models/`](../models/index.md).

Up: [docs/index.md](../index.md) — all sections.

## Table of contents

### Big picture

- [architecture-overview.md](./architecture-overview.md) — top-down map: Tracer →
  Backend → Interleaver → Mediator (greenlet) → controller → Envoy, and the one
  trace lifecycle walkthrough.
- [tracing-pipeline.md](./tracing-pipeline.md) — capture → parse → build → compile
  → execute, `Scope`, the skip hook, the per-site cache.

### Interleaver

- [interleaver-internals.md](./interleaver-internals.md) — `Interleaver`,
  `Mediator`, the greenlet park/switch dance, `Event`, `handle()` fan-out,
  iteration tagging, the caches.
- [controller.md](./controller.md) — the per-module controller installed at
  instrument time: the handoff, the skip gate, the source-instrumented body.
- [source-internals.md](./source-internals.md) — `.source` operation-level access
  via AST-instrumented forwards.

### Batching

- [batching-internals.md](./batching-internals.md) — `Batcher` add/narrow/widen,
  batch groups, `gather_skip`/`assemble_skip`.

### Distributed values

- `src/nnsight/intervention/fragments.py` (module docstring) — `Fragments`, the one
  seam vLLM and tensor parallelism share for making a sharded value whole before a
  worker sees it; the runtimes' rules live in `modeling/tp/fragments.py` and
  `modeling/vllm/fragments.py`.

### Backends

- [backends.md](./backends.md) — the `Backend` base and the existing backends
  (`RemoteBackend`, `AsyncRemoteBackend`, `LocalSimulationBackend`), how a model
  selects one from `remote=`, and how to add your own.

### Extending

- [extending.md](./extending.md) — subclass `NNsight`/`Envoy`: batching
  (`_batch_size`/`_batch`), attached modules, a custom `Batcher`, `Loadable`, a
  custom tracer.
- [extending-envoy.md](./extending-envoy.md) — new served values: an `eproperty`
  on a model/runtime subclass, fed from the driver with its `.provide`; `envoys=`.
- [adding-a-new-runtime.md](./adding-a-new-runtime.md) — plug a new model type or
  inference engine in via the modeling mixins.
- [serialization.md](./serialization.md) — source-based serialization for remote
  runs.

### Runtimes

- [vllm-integration.md](./vllm-integration.md) — the vLLM runtime: model runner,
  batcher, how interventions ride on engine requests.

### Reference

- [performance.md](./performance.md) — where overhead lives and how to measure it.
- [testing.md](./testing.md) — running the suite offline; what each test covers.
- [contributing.md](./contributing.md) — house style, branch/commit conventions.
