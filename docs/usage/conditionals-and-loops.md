---
title: Conditionals and Loops
one_liner: Standard Python if/for work inside trace contexts because the worker receives real tensors.
tags: [usage, control-flow, python]
related: [docs/usage/trace.md, docs/usage/session.md, docs/usage/access-and-modify.md, docs/usage/iteration.md]
sources: [src/nnsight/intervention/interleaver.py, src/nnsight/intervention/envoy.py]
---

# Conditionals and Loops

Inside any tracing context, reading `module.output` returns the **actual
`torch.Tensor`**. There is no proxy and no special control-flow API, so ordinary
Python — `if`, `for`, `while`, comprehensions, helper functions — runs over real
values.

```python
import nnsight
import torch
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

def normalize(x):
    return (x - x.mean()) / x.std()

with model.trace("Hello world"):
    hs = model.transformer.h[5].output
    if hs.norm(dim=-1).mean() > 10.0:     # a real 0-d tensor; Python calls __bool__
        model.transformer.h[6].output[:] = 0

    z = normalize(model.transformer.h[7].output).save()
    means = nnsight.save([model.transformer.h[i].output.mean() for i in range(8, 12)])
```

## Rules

- **Loops cannot reorder module access.** Within one invoke, modules must be read in
  forward-pass order (the comprehension above goes 8..11), or `OutOfOrderError`.
  Use separate invokes for anything else ([invoke-and-batching](invoke-and-batching.md)).
- **A Python `for` does not loop over generation steps** — use `tracer.iter[...]`
  ([iteration](iteration.md)). Branches decide over the current values; they do not
  create per-step branches in the model.
- **Looping over prompts**: wrap the traces in [`model.session()`](session.md), or
  batch them as invokes of one trace ([invoke-and-batching](invoke-and-batching.md)).
- **`if some_tensor:` on a multi-element tensor raises `RuntimeError`**, as in plain
  PyTorch. Reduce to a scalar (`.all()`, `.item()`) first.
- **A trace body cannot begin with `try:`** — see
  [with-block-not-found](../errors/with-block-not-found.md#a-trace-body-that-starts-with-try).
