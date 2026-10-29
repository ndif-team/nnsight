---
title: Usage Index
one_liner: Recipe-style docs for every user-facing nnsight feature, one sharp page per topic.
tags: [usage, index]
related: [docs/index.md, docs/concepts/index.md, docs/errors/symptom-index.md, docs/reference/index.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/tracer.py]
---

# Usage Index

One page per feature: the canonical pattern, variations, and the feature's pitfalls.

Up: [docs/index.md](../index.md) — all sections.

## Tracing entry points

- [trace](trace.md) — `model.trace(input)`: single forward pass with interventions.
- [generate](generate.md) — `model.generate(input, max_new_tokens=N)`: multi-token generation through the model; returns **token ids** (`tracer.result`), with the checkpoint's own `generation_config` (which may sample).
- [pipe](pipe.md) — `model.pipe(input, ...)`: run the whole task pipeline; returns its **records** (decoded text, labels, ...).
- [scan](scan.md) — `model.scan(input)`: shapes and intervention validity without loading weights or running real compute.
- [edit](edit.md) — `model.edit(inplace=...)`: persistent / always-on interventions that fire on every future trace (steering, LoRA/SAE attach).
- [session](session.md) — `model.session()`: multiple traces or loops over prompts in one scope that shares values (and one remote round trip).

## Inputs and batching

- [invoke-and-batching](invoke-and-batching.md) — `tracer.invoke(input)`: several prompts in one forward, padding and positions, activation patching across prompts, empty invokes for whole-batch operations.
- [barrier](barrier.md) — `tracer.barrier(n)`: activation patching across prompts at the same module (cross-invoke sharing).
- [rename-modules](rename-modules.md) — `rename={...}` aliases for ergonomic module paths.

## Reading and writing values

- [save](save.md) — `nnsight.save(...)` / `obj.save()`: persist a value across the trace boundary.
- [access-and-modify](access-and-modify.md) — `.output` / `.input` / `.inputs`: reading values and writing them back (in-place vs replacement, tuple outputs).
- [source](source.md) — `module.source.<op>.output`: operations inside a module's forward, e.g. attention scores before softmax.
- [cache](cache.md) — `tracer.cache(modules=..., include_inputs=...)`: bulk activation cache that accumulates across generation steps.

## Generation control

- [iteration](iteration.md) — `tracer.iter[slice|int|list]`, `tracer.all()`: per-step targeting over a repeated run, and the ways a loop goes wrong.
- [stop-and-early-exit](stop-and-early-exit.md) — `tracer.stop()` to abort the forward pass early.
- [skip](skip.md) — `module.skip(replacement)`: bypass a module's compute entirely.

## Gradients

- [backward-and-grad](backward-and-grad.md) — `with tensor.backward():` and `.grad`: gradients of activations, gradient attribution, editing gradients.

## Control flow

- [conditionals-and-loops](conditionals-and-loops.md) — plain Python `if` / `for` inside a trace, branching on tensor values.
