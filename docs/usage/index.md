---
title: Usage Index
one_liner: Recipe-style docs for every user-facing nnsight feature, one sharp page per topic.
tags: [usage, index]
related: [docs/concepts/index.md, docs/gotchas/index.md, docs/reference/index.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/tracer.py]
---

# Usage Index

One-liner per feature. Click through for the canonical pattern, variations, and gotchas.

All examples use `TransformersModel`:

```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)
```

(`LanguageModel` / `VisionLanguageModel` still work but are deprecated thin aliases that warn on construction — use `TransformersModel(task=...)`.)

## Tracing entry points

- [trace](trace.md) — `model.trace(input)`: single forward pass with interventions.
- [generate](generate.md) — `model.generate(input, max_new_tokens=N)`: multi-token generation through the model; returns **token ids** (`tracer.result`). Greedy by default.
- [pipe](pipe.md) — `model.pipe(input, ...)`: run the whole task pipeline; returns its **records** (decoded text, labels, ...).
- [scan](scan.md) — `model.scan(input)`: validate shapes / interventions under `FakeTensorMode` (no real compute, no dispatch).
- [edit](edit.md) — `model.edit(inplace=...)`: persist interventions onto the model so they fire on every future trace.
- [session](session.md) — `model.session()`: bundle multiple traces into one scope that shares values (and one remote round trip).

## Inputs and batching

- [invoke-and-batching](invoke-and-batching.md) — `tracer.invoke(input)` to add inputs (one greenlet worker per invoke), empty invokes for whole-batch operations.
- [barrier](barrier.md) — `tracer.barrier(n)` to synchronize cross-invoke variable sharing on the same module.
- [rename-modules](rename-modules.md) — `rename={...}` aliases for ergonomic module paths.

## Reading and writing values

- [save](save.md) — `nnsight.save(...)` / `obj.save()`: persist a value across the trace boundary. **Raises if called outside a trace.**
- [access-and-modify](access-and-modify.md) — `.output` / `.input` / `.inputs`: reading values and writing them back (in-place vs replacement, tuple outputs).
- [source](source.md) — `module.source.<op>.output`: hook intermediate operations inside a module's forward.
- [cache](cache.md) — `tracer.cache(modules=..., include_inputs=...)`: bulk activation cache that accumulates across generation steps.

## Generation control

- [iter-all-next](iter-all-next.md) — `tracer.iter[slice|int|list]`, `tracer.all()`: per-step targeting and blanket recursion over a repeated run.
- [stop-and-early-exit](stop-and-early-exit.md) — `tracer.stop()` to abort the forward pass early.
- [skip](skip.md) — `module.skip(replacement)`: bypass a module's compute entirely.

## Gradients

- [backward-and-grad](backward-and-grad.md) — `with tensor.backward():` (a nested interleaving session) and `.grad` access.

## Control flow

- [conditionals-and-loops](conditionals-and-loops.md) — Plain Python `if` / `for` inside a trace.

## Extending nnsight

- [extending](extending.md) — Custom `Envoy` subclasses, attaching modules to the tree, and integration patterns for new runtimes.
