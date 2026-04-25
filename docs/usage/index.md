---
title: Usage Index
one_liner: Recipe-style docs for every user-facing nnsight feature, one sharp page per topic.
tags: [usage, index]
related: [docs/concepts/index.md, docs/gotchas/index.md, docs/reference/index.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/tracing/tracer.py]
---

# Usage Index

One-liner per feature. Click through for the canonical pattern, variations, and gotchas.

## Tracing entry points

- [trace](trace.md) — `model.trace(input)`: single forward pass with interventions.
- [generate](generate.md) — `model.generate(input, max_new_tokens=N)`: multi-token autoregressive generation.
- [scan](scan.md) — `model.scan(input)`: validate shapes / interventions with `FakeTensorMode` (no real compute).
- [edit](edit.md) — `model.edit(inplace=...)`: persist interventions onto the model so they fire on every future trace.
- [session](session.md) — `model.session()`: bundle multiple traces into one logical request (and one remote round trip).

## Inputs and batching

- [invoke-and-batching](invoke-and-batching.md) — `tracer.invoke(input)` to add inputs (one thread per invoke), empty invokes for whole-batch operations.
- [barrier](barrier.md) — `tracer.barrier(n)` to synchronize cross-invoke variable sharing on the same module.
- [rename-modules](rename-modules.md) — `rename={...}` aliases for ergonomic module paths.

## Reading and writing values

- [save](save.md) — `nnsight.save(...)` / `obj.save()`: persist a value across the trace boundary.
- [access-and-modify](access-and-modify.md) — `.output` / `.input` / `.inputs`: reading values and writing them back (in-place vs replacement, tuple outputs).
- [source](source.md) — `module.source.<op>.output`: hook intermediate operations inside a module's forward, recursively.
- [cache](cache.md) — `tracer.cache(modules=..., include_inputs=...)`: persistent activation cache that survives generation steps.

## Generation control

- [iter](iter.md) — `tracer.iter[slice|int|list]`: target specific generation steps.
- [all-and-next](all-and-next.md) — `tracer.all()` / `tracer.next()` / `module.next()`: blanket and manual stepping.
- [stop-and-early-exit](stop-and-early-exit.md) — `tracer.stop()` to abort the forward pass early.
- [skip](skip.md) — `module.skip(replacement)`: bypass a module's compute entirely.

## Gradients

- [backward-and-grad](backward-and-grad.md) — `with tensor.backward():` (a separate interleaving session) and `.grad` access.

## Control flow

- [conditionals-and-loops](conditionals-and-loops.md) — Plain Python `if` / `for` inside a trace (v0.5+).

## Extending nnsight

- [extending](extending.md) — Custom `Envoy` subclasses via `envoys=`, custom `eproperty` (preprocess / postprocess / transform), and integration patterns for new runtimes.
