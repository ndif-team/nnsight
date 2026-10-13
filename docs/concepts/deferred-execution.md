---
title: Deferred Execution
one_liner: A trace block doesn't run inline — nnsight captures its body from source and runs it alongside the model's forward pass, pausing at each Envoy access until the model reaches that module.
tags: [concept, mental-model, tracing]
related: [docs/concepts/threading-and-mediators.md, docs/concepts/interleaver-and-controller.md, docs/concepts/envoy.md, docs/developing/tracing-pipeline.md]
sources: [src/nnsight/tracing/tracer.py, src/nnsight/intervention/tracer.py]
---

# Deferred Execution

## The idea

The body of `with model.trace(...)` does not run where you wrote it. On entering the
`with`, nnsight reads the block's source, compiles the body on its own, stops the
interpreter from running it inline, and then runs the compiled body *alongside* the
model's forward pass. When the body reads an Envoy property like `.output`, it pauses
until the model reaches that module, receives the real value, and continues.

So a trace body is ordinary Python operating on real tensors — it just runs in
lockstep with the model rather than before or after it. When the block exits, only
the values you `.save()` are bound back into your scope
([usage/save.md](../usage/save.md)).

```python
import torch
from nnsight import NNsight

model = NNsight(torch.nn.Sequential(torch.nn.Linear(5, 10), torch.nn.Linear(10, 2)))

with model.trace(torch.rand(1, 5)):
    hidden = model[0].output.save()   # pauses until layer 0 has run, then gets its output

print(hidden.shape)   # torch.Size([1, 10])
```

`trace(x)` makes the whole body one worker over `x`; `trace()` plus
`tracer.invoke(...)` blocks makes one worker per invoke
([batching-and-invokers.md](batching-and-invokers.md)). `model.scan(x)` runs the same
machinery on fake tensors for shapes only ([usage/scan.md](../usage/scan.md)). The
step-by-step pipeline is in
[developing/tracing-pipeline.md](../developing/tracing-pipeline.md).

## What follows from it

- **The body runs against a snapshot of your scope**, not your frame directly. It
  sees your locals and globals, but its assignments come back only if saved.
- **The body must start on its own line.** `with model.trace(x): y = ...` is refused;
  nnsight can only intercept a body at the start of a line.
- **The source must be readable.** Files, notebook cells and `python -c` work; a
  script piped on stdin or a string passed to `exec` does not. Each file's source is
  read once, so a file edited mid-process is traced as first seen. Errors from this
  step: [errors/with-block-not-found.md](../errors/with-block-not-found.md).
- **Order matters.** Because the body is paused and resumed by the forward pass, it
  must ask for locations in the order the model produces them
  ([threading-and-mediators.md](threading-and-mediators.md)).

## Related

- [Threading and Mediators](threading-and-mediators.md) — how the paused body and the model take turns.
- [Interleaver and Controller](interleaver-and-controller.md) — the model side that delivers values.
