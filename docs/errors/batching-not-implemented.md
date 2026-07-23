---
title: Model Does Not Support Batching Multiple Invokes
one_liner: "NotImplementedError: <ModelClass> does not support batching multiple invokes — two or more input invokes on a model whose _batch() isn't implemented."
tags: [error, batching, setup]
related: [docs/usage/invoke-and-batching.md, docs/concepts/batching-and-invokers.md, docs/usage/extending.md]
sources: [src/nnsight/intervention/envoy.py:597, src/nnsight/intervention/envoy.py:608, src/nnsight/intervention/batching.py:190, src/nnsight/modeling/transformers.py:682]
---

# Model Does Not Support Batching Multiple Invokes

## Symptom

```
NotImplementedError: NNsight does not support batching multiple invokes
```

The class name is whatever model you used — e.g. a custom `NNsight` subclass prints
its own name.

`TransformersModel` implements batching, so it won't hit this for ordinary text
inputs; it raises its own, more specific messages only for un-batchable multimodal
inputs (`src/nnsight/modeling/transformers.py:682`, `:710`):

```
NotImplementedError: Batching multimodal generate inputs isn't supported; pass a single text/images payload.
NotImplementedError: Can't batch these inputs; pass text or token ids.
```

## Cause

`Envoy._batch` (`src/nnsight/intervention/envoy.py:597`) is the hook that combines
several invokes' inputs into one forward. The base default passes a single invoke
straight through but raises for two or more
(`src/nnsight/intervention/envoy.py:608`):

```python
def _batch(self, invokes, fn):
    if not invokes:
        return tuple(), {}
    if len(invokes) == 1:
        return invokes[0]
    raise NotImplementedError(
        f"{type(self).__name__} does not support batching multiple invokes"
    )
```

`Batcher.assemble` calls it once all invokes are collected
(`src/nnsight/intervention/batching.py:190`). So the error fires only when **two or
more invokes contribute input rows**. A single input invoke, or one input invoke
plus empty invokes, never needs `_batch` to merge anything.

`TransformersModel` overrides `_batch_size` (tokenizes / counts rows) and `_batch`
(pads + concatenates), so it batches out of the box. A bare `NNsight(my_module)`
inherits the base default and can't merge arbitrary tensor inputs without help.

## Fix

### Option 1 — one input invoke + empty invokes

Empty invokes (`tracer.invoke()`) contribute no rows and operate on the whole batch,
so they never touch `_batch`. This works even on bare `NNsight`, and is the usual
way to split interventions to avoid execution-order conflicts:

```python
with model.trace() as tracer:
    with tracer.invoke(input_tensor):        # one input invoke
        a = model.layer5.output.save()
    with tracer.invoke():                    # empty invoke — whole batch, new worker
        b = model.layer2.output.save()
```

### Option 2 — use TransformersModel for HF models

```python
# WRONG — bare NNsight doesn't know how to batch two token inputs
from nnsight import NNsight
from transformers import AutoModelForCausalLM
model = NNsight(AutoModelForCausalLM.from_pretrained("openai-community/gpt2"))
with model.trace() as tracer:
    with tracer.invoke("Hello"): ...
    with tracer.invoke("World"): ...         # NotImplementedError
```

```python
# FIXED — TransformersModel implements _batch_size + _batch
from nnsight import TransformersModel
model = TransformersModel("openai-community/gpt2")
with model.trace() as tracer:
    with tracer.invoke("Hello"): ...
    with tracer.invoke("World"): ...
```

### Option 3 — implement `_batch_size` and `_batch` yourself

For a non-HF model, override both on your `NNsight` subclass — `_batch_size`
returns the row count of an invoke's input, `_batch` merges the collected invokes
into one `(args, kwargs)`:

```python
import torch
from nnsight import NNsight

class MyModel(NNsight):
    def _batch_size(self, *inputs, **kwargs):
        return inputs[0].shape[0] if inputs else 0     # number of rows

    def _batch(self, invokes, fn):
        args = [inp[0][0] for inp in invokes]          # each invoke's first arg
        return (torch.cat(args, dim=0),), {}
```

See [docs/usage/extending.md](../usage/extending.md) and
`src/nnsight/modeling/transformers.py` for a real reference implementation.

## Related

- [docs/usage/invoke-and-batching.md](../usage/invoke-and-batching.md)
- [docs/concepts/batching-and-invokers.md](../concepts/batching-and-invokers.md)
- [docs/usage/extending.md](../usage/extending.md)
