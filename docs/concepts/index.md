---
title: Concepts Index
one_liner: Mental models for how a trace runs — deferred execution, workers, controllers, Envoys, invokes and .source.
tags: [concept, index]
related: [docs/usage/index.md, docs/developing/index.md]
sources: [src/nnsight/intervention/interleaver.py, src/nnsight/intervention/envoy.py, src/nnsight/tracing/tracer.py]
---

# Concepts Index

Up: [docs/index.md](../index.md) — all sections.

Why nnsight behaves the way it does. For recipes see [usage/](../usage/index.md); for the code see [developing/](../developing/index.md).

- [Deferred Execution](deferred-execution.md) — a trace body is captured and run alongside the forward pass, not inline.
- [Threading and Mediators](threading-and-mediators.md) — each block is a worker that takes turns with the model on one thread; why order matters.
- [Interleaver and Controller](interleaver-and-controller.md) — how each module serves its input/skip/output, and what an occurrence (`.i0`) is.
- [Envoy](envoy.md) — the module wrapper: `.input`, `.output`, `.skip`, `.source`, calling a module, changing the tree.
- [Batching and Invokers](batching-and-invokers.md) — invokes as workers over their own rows of one batch; empty invokes; which values get scoped.
- [Source Tracing](source-tracing.md) — `.source` as finer-grained locations inside a forward, under the same rules.
