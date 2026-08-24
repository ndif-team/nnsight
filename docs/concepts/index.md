---
title: Concepts Index
one_liner: Mental-model docs for nnsight's capture, greenlet-interleaving, hook, and source machinery.
tags: [concept, index]
related: [docs/usage/index.md, docs/reference/index.md]
sources: [src/nnsight/intervention/interleaver.py, src/nnsight/intervention/envoy.py, src/nnsight/intervention/source.py, src/nnsight/tracing/tracer.py]
---

# Concepts Index

Foundational mental models for working with nnsight. Read these in order for a top-down picture of how `with model.trace(...)` actually runs.

## Docs

- [Deferred Execution](deferred-execution.md) — How `with model.trace(...)` captures a code block (`capture → parse → build → compile → execute`), then runs it interleaved with the model instead of inline. `save()` and the nesting/save boundary.
- [Threading and Mediators](threading-and-mediators.md) — Each block is one `Mediator` running in a **greenlet** (not a thread). Workers *park* and *switch*, exchanging typed events with the model side: `VALUE` / `SWAP` / `SKIP` / `BARRIER`.
- [Interleaver and Controller](interleaver-and-controller.md) — One shared `Interleaver` installs a controller forward on every module at wrap time; it passes through when not interleaving and hands off through `Interleaver.handle` when it is. Occurrence tagging (`.i{n}`) and out-of-order detection.
- [Envoy](envoy.md) — `Envoy` wraps a `torch.nn.Module` and exposes `.input` / `.inputs` / `.output` as `eproperty` descriptors over `Mediator.value` / `Mediator.swap`, plus `.skip` (method) and `.source` (property). Extension by subclassing, by attaching modules (`__call__(hook=True)`, edits), and by adding custom `eproperty` values served with `.provide`.
- [Batching and Invokers](batching-and-invokers.md) — `tracer.invoke(...)` as one worker on its own batch slice; the `Batcher` `narrow`/`widen`; empty invokes on the full batch; `barrier()` for cross-invoke value sharing.
- [Source Tracing](source-tracing.md) — `.source` rewrites a module's forward AST so every call site becomes a hookable location, bracketed through the same `Interleaver.handle` primitive. `Source`, `SourceEnvoy`, recursive `.source`, and the per-module controller.

## Related

- [Usage Index](../usage/index.md) — recipe pages for `trace`, `generate`, `pipe`, `scan`, `edit`, `session`.
- [Models Index](../models/index.md) — `NNsight`, `TransformersModel`, and the deprecated aliases.
