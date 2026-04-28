---
title: Concepts Index
one_liner: Mental-model docs for nnsight's tracing, interleaving, hook, and source machinery.
tags: [concept, index]
related: [docs/usage/index.md, docs/gotchas/index.md]
sources: [src/nnsight/intervention/interleaver.py, src/nnsight/intervention/envoy.py, src/nnsight/intervention/source.py]
---

# Concepts Index

Foundational mental models for working with nnsight. Read these in order if you want a top-down picture.

## Docs

- [Deferred Execution](deferred-execution.md) — How `with model.trace(...)` captures a code block, compiles it into a function, and runs it in a worker thread.
- [Threading and Mediators](threading-and-mediators.md) — Each invoke is one `Mediator` (one worker thread). Mediators communicate with the main thread via an event protocol (VALUE / SWAP / SKIP / BARRIER / END / EXCEPTION).
- [Interleaver and Hooks](interleaver-and-hooks.md) — The lazy one-shot hook architecture. Sentinel forward hook, `add_ordered_hook`, `mediator_idx` ordering, and `Mediator.hooks` lifecycle.
- [Envoy and eproperty](envoy-and-eproperty.md) — `Envoy` wraps a `torch.nn.Module` and exposes `.output` / `.input` / `.inputs` / `.source` / `.skip` / `.next` via the `eproperty` descriptor. Custom eproperties (`preprocess` / `postprocess` / `transform`) for extending the interception API.
- [Batching and Invokers](batching-and-invokers.md) — `tracer.invoke(...)` as a worker thread, empty invokes on the combined batch, cross-invoke variable sharing, and when you need a `barrier()`.
- [Source Tracing](source-tracing.md) — `.source` rewrites a module's forward AST so every call site is hookable. `SourceAccessor`, `OperationAccessor`, recursive source, and the per-Envoy `SourceEnvoy` / `OperationEnvoy` wrappers.

## Related

- [NNsight.md](../../NNsight.md) — Deep technical doc covering the full architecture.
- [0.6.0 release notes](../../0.6.0.md) — What changed in this branch (lazy hooks, persistent pymount, code caching).
