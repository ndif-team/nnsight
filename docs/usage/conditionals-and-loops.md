---
title: Conditionals and Loops
one_liner: Standard Python if/for work inside trace contexts because the worker receives real tensors.
tags: [usage, control-flow, python]
related: [docs/usage/trace.md, docs/usage/session.md, docs/usage/access-and-modify.md]
sources: [src/nnsight/intervention/interleaver.py, src/nnsight/intervention/envoy.py]
---

# Conditionals and Loops

## What this is for

Inside any nnsight tracing context (`model.trace`, `model.generate`, `model.pipe`,
`model.scan`, `model.session`), the body is captured and run in a worker
**greenlet**. When the worker reads `module.output`, it parks until the model
produces the value — and then receives the **actual `torch.Tensor`**. So normal
Python control flow over those values just works: `if`, `for`, `while`,
comprehensions, helper functions, etc.

There is no proxy class and no special control-flow API — the worker just gets the
real tensor.

## When to use / when not to use

- Use `if`/`for` freely on activation values.
- Don't use `for` to loop over generation steps — use `tracer.iter[...]`
  ([iter-all-next.md](iter-all-next.md)).
- Don't use `for` to access modules out of order within one invoke — modules must be
  accessed in forward-pass order. Use multiple invokes
  ([invoke-and-batching.md](invoke-and-batching.md)).

## Canonical pattern

```python
import torch
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("Hello world"):
    out = model.transformer.h[0].output

    # real tensor -> torch.all returns a real bool
    if torch.all(out < 1e5):
        model.transformer.h[-1].output[:] = 0

    final = model.transformer.h[-1].output.save()
```

`final` is all zeros — the branch was taken because `out` is a real tensor.

## Branching on tensor values

```python
with model.trace("Hello world"):
    hs   = model.transformer.h[5].output
    norm = hs.norm(dim=-1).mean()

    if norm > 10.0:                      # real 0-d tensor; Python calls __bool__
        model.transformer.h[6].output[:] = 0

    final = model.output.logits.save()
```

## Comprehensions and helper functions

```python
def normalize(x):
    return (x - x.mean()) / x.std()

with model.trace("Hello world"):
    norm  = normalize(model.transformer.h[0].output).save()   # helper call
    means = [model.transformer.h[i].output.mean() for i in range(12)]
    means = nnsight.save(means)          # save a list of tensors in one call
```

All entries are real, fully-resolved torch tensors. The comprehension reads layers
0..11 in forward-pass order, which is required (see below).

## `try` needs a statement above it

A trace body cannot begin with `try:` — nnsight intercepts the body at its first
line, and a `try` there is the one statement Python gives it no way back out of:

```python
with model.trace("Hello world"):
    try:                                  # ValueError at `with` entry
        h = model.transformer.h[0].output.save()
    except RuntimeError:
        h = None
```

```
ValueError: A traced `with` block cannot start with `try:`; nnsight intercepts
the body at its first line, and a `try` there is the one statement Python gives
it no way back out of. Put any statement above the `try`, or move the `try`
outside the block.
```

Anywhere else in the body it is ordinary Python:

```python
with model.trace("Hello world"):
    layers = model.transformer.h            # any statement will do
    try:
        h = layers[0].output.save()
    except RuntimeError:
        h = None
```

Wrapping the whole `with` in a `try` — to log failures around the trace — is
unaffected; the restriction is only on the block's own first statement.

## Loops inside an invoke

Loops are fine as long as they respect forward-pass order:

```python
with model.trace("Hello world"):
    activations = nnsight.save([])
    for i in range(len(model.transformer.h)):   # 0..N-1, execution order
        activations.append(model.transformer.h[i].output)
```

Reading `h[5].output` then `h[2].output` in the same invoke is out of order and
raises `OutOfOrderError`.

## Looping over prompts (use `model.session`)

```python
with model.session():
    results = nnsight.save([])
    for p in ["Hello", "World", "Test"]:
        with model.trace(p):
            results.append(model.lm_head.output[0, -1].argmax(dim=-1))
```

The outer `session()` captures the body once and runs the loop as plain Python, so
several traces share one scope. For remote workloads use
`model.session(remote=True)` so it's one round-trip. See [session.md](session.md).

## Gotchas

- **`if`/`for` are real Python** — they decide over current activation values; they
  do not create per-step branches in the model's forward.
- **`for step in tracer.iter[:]:` is different** — that loops over generation steps,
  not Python iterations ([iter-all-next.md](iter-all-next.md)).
- **Python loops cannot reorder module access** — iteration order must match the
  forward pass, or `OutOfOrderError`.
- **Inside a `model.session()` body but outside an inner trace you're in plain
  Python** — `module.output` is not accessible there. Open a `model.trace(...)`
  first.
- **`if some_tensor:` on a multi-element tensor raises `RuntimeError`** — exactly as
  in vanilla PyTorch. Reduce to a scalar (`.all()`, `.item()`, ...) first.
- **A trace body cannot start with `try:`** — put any statement above it, or the
  `try` outside the `with`.

## Related

- [trace.md](trace.md)
- [session.md](session.md)
- [access-and-modify.md](access-and-modify.md)
- [iter-all-next.md](iter-all-next.md)
