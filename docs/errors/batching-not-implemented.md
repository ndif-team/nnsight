---
title: Model Does Not Support Batching Multiple Invokes
one_liner: "NotImplementedError: <ModelClass> does not support batching multiple invokes — two or more input invokes on a model whose _batch() isn't implemented."
tags: [error, batching, setup]
related: [docs/usage/invoke-and-batching.md, docs/concepts/batching-and-invokers.md, docs/usage/extending.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/batching.py, src/nnsight/modeling/transformers.py]
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
inputs (`TransformersModel._batch_generate` / `._batch`):

```
NotImplementedError: Batching multimodal generate inputs isn't supported; pass a single text/images payload.
NotImplementedError: Can't batch these inputs; pass text or token ids.
```

## Cause

`Envoy._batch` (`src/nnsight/intervention/envoy.py`) is what combines several
invokes' inputs into one forward. The base default passes a single invoke straight
through and raises for two or more:

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

`Batcher.assemble` calls it once all invokes are collected, so the error fires only when **two or
more invokes contribute input rows**. A single input invoke, or one input invoke
plus empty invokes, never needs `_batch` to merge anything.

`TransformersModel` overrides `_batch_size` (tokenizes / counts rows) and `_batch`
(pads + concatenates), so it batches out of the box. A bare `NNsight(my_module)`
inherits the base default and can't merge arbitrary tensor inputs without help.

## Fix

### Add rows in one invoke, or use empty invokes

Empty invokes (`tracer.invoke()`) contribute no rows and see the whole batch, so
they never touch `_batch`. This works on bare `NNsight`, and is the usual way to
split interventions that would otherwise conflict on execution order:

```python
with model.trace() as tracer:
    with tracer.invoke(input_tensor):        # one input invoke
        a = model.layer5.output.save()
    with tracer.invoke():                    # empty invoke — whole batch, new worker
        b = model.layer2.output.save()
```

### Or teach the model to batch

`TransformersModel` overrides `_batch_size` (counts rows) and `_batch` (pads and
concatenates), so HuggingFace models batch out of the box — reach for it rather
than `NNsight(hf_model)`. For a non-HF model, implement the same two methods on
your subclass: see [docs/usage/extending.md](../usage/extending.md), which carries
the recipe and a worked example.

## Related

- [docs/usage/invoke-and-batching.md](../usage/invoke-and-batching.md)
- [docs/concepts/batching-and-invokers.md](../concepts/batching-and-invokers.md)
- [docs/usage/extending.md](../usage/extending.md)
