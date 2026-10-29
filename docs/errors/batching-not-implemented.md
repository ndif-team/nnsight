---
title: Model Does Not Support Batching Multiple Invokes
one_liner: "NotImplementedError: <ModelClass> does not support batching multiple invokes — two or more input invokes on a model whose _batch() isn't implemented."
tags: [error, batching, setup]
related: [docs/usage/invoke-and-batching.md, docs/concepts/batching-and-invokers.md, docs/developing/extending.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/batching.py, src/nnsight/modeling/processing.py]
---

# Model Does Not Support Batching Multiple Invokes

## Symptom

```
NotImplementedError: NNsight does not support batching multiple invokes
```

The class name is whatever model you used. `TransformersModel` batches ordinary
text inputs; it raises its own messages only for un-batchable multimodal inputs:

```
NotImplementedError: Batching multimodal generate inputs isn't supported; pass a single text/images payload.
NotImplementedError: Can't batch these inputs; pass text or token ids.
```

## Cause

Combining several invokes' inputs into one forward is the model class's
`_batch()` method. The base `Envoy._batch` passes a single invoke through and
raises for two or more, so the error fires only when **two or more invokes
contribute input rows**. One input invoke plus any number of empty invokes never
needs merging. A bare `NNsight(my_module)` inherits that default.

## Fix

Put all rows in one invoke, and use empty invokes (`tracer.invoke()`) to split
interventions — they contribute no rows and see the whole batch:

```python
with model.trace() as tracer:
    with tracer.invoke(input_tensor):        # the one input invoke
        a = model.layer5.output.save()
    with tracer.invoke():                    # empty invoke — whole batch, new worker
        b = model.layer2.output.save()
```

Or let the model batch: `TransformersModel` overrides `_batch_size` and `_batch`
(pads and concatenates), so use it rather than `NNsight(hf_model)`. For a non-HF
model, implement the same two methods on a subclass — recipe and worked example in
[docs/developing/extending.md](../developing/extending.md).

## Related

- [docs/usage/invoke-and-batching.md](../usage/invoke-and-batching.md)
- [docs/concepts/batching-and-invokers.md](../concepts/batching-and-invokers.md)
