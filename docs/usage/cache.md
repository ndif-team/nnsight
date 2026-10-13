---
title: Activation Cache
one_liner: tracer.cache() records module outputs (and optionally inputs) into a path- and attribute-addressable view.
tags: [usage, cache, intervention]
related: [docs/usage/access-and-modify.md, docs/usage/iteration.md, docs/usage/save.md, docs/usage/invoke-and-batching.md, docs/usage/rename-modules.md]
sources: [src/nnsight/intervention/cache.py, src/nnsight/intervention/tracer.py]
---

# Activation Cache

`tracer.cache(...)` records the activations of many modules at once: every
selected module, across the whole run, one entry per visit (so one per step in a
generation loop). It observes values **after** interventions apply. Reach for it
when you want the same value from many modules or across every generation step;
for one value, `module.output.save()` is simpler.

## Canonical pattern

```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("The Eiffel Tower is in") as tracer:
    cache = tracer.cache(modules=[model.transformer.h[0], "model.lm_head"])

cache["model.transformer.h.0"].output              # by path: a tensor
cache.transformer.h[0].output                      # or by navigating the tree
```

`modules=` takes Envoy objects or path strings; `None` caches every module.
**Always pass it on a real model**: caching all of GPT-2's modules costs roughly
20x the memory and time of caching its 12 blocks, and a cache costs the same as a
hand-written `save()` loop over the same modules. `cache.keys()` lists cached
paths in the order the run reaches them, not the order you passed, so key off the
path rather than zipping against `modules=`.

Renamed modules work in both path and tree access; see
[rename-modules](rename-modules.md).

## API

```python
tracer.cache(
    modules=None,                 # None | list[Envoy | str]; None = every module
    device=torch.device("cpu"),   # where captures go; None leaves them in place
    dtype=None,                   # optional cast
    detach=True,                  # cut captures off from autograd
    include_output=True,
    include_inputs=False,         # True adds .inputs / .input to each entry
    non_blocking=False,           # True is async and needs your own sync
)
```

Returns a `CacheView`, already saved, so it survives past the trace.

- **`include_inputs=True` can pin the live KV cache.** The `device=` move reaches
  tensors and the containers around them, not tensors held inside other objects.
  A HuggingFace decoder block is called with `past_key_values=<DynamicCache>`, so
  every layer's entry keeps a reference to the model's live KV cache on the
  compute device, and GPU memory grows with every cache you hold. Release the
  caches after the sweep, or leave `include_inputs=False` and `.save()` the
  specific `.input` you want. Without it, `.inputs` is `None`.
- **`non_blocking=True` is unsafe unless you synchronize.** The copy to `device`
  is enqueued asynchronously and nothing synchronizes before you read it (reading
  a CPU tensor does not synchronize CUDA), so a capture can be read mid-copy and
  be silently wrong — more often the larger the run. Call
  `torch.cuda.synchronize()` after the trace before reading.

## Cache across generation steps

`cache[path].output` is the value itself for a single visit and a `list` for
several; `len(cache[path])` is the visit count:

```python
with model.generate("Hello", max_new_tokens=3, do_sample=False) as tracer:
    cache = tracer.cache(modules=[model.transformer.h[-1]])

len(cache["model.transformer.h.11"])            # 3 (one per step)
cache["model.transformer.h.11"].output[0]       # step 0's tensor
```

Index the value (`.output[i]`); the view itself is not subscriptable by visit. A
shared-weight module hit twice in one forward also gives a list.

## Cache + interventions

```python
with model.trace("Hello") as tracer:
    cache = tracer.cache()
    model.transformer.h[0].output[:] = 0

(cache["model.transformer.h.0"].output == 0).all()   # True
```

## Caching inside an invoke

A cache opened inside `tracer.invoke(...)` records that invoke's rows only, at the
**whole batch's padded length**, with nothing marking which positions are pad; pad
positions carry real, often large, values. Mask before reducing over the sequence
axis, or index from the right (`[:, -1]`). To capture the whole batch, open the
cache in an empty `tracer.invoke()`. Padding and masks: see
[invoke-and-batching](invoke-and-batching.md).

## Gradients through a cache

Pass `detach=False` and `device=None` to get captures you can call `backward()`
on. This keeps the whole graph alive as long as the cache, which is why
`detach=True` is the default.

```python
with model.trace("the cat") as tracer:
    cache = tracer.cache(modules=[model.transformer.h[0]], detach=False, device=None)

cache["model.transformer.h.0"].output.sum().backward()   # fills model .grad
```

## Modules called ad hoc

A module you call yourself inside the trace body (an adapter, an SAE) appears in
the cache, with its submodules, only when called with `hook=True` (see
[access-and-modify](access-and-modify.md#calling-modules-directly-inside-a-trace)):

```python
with model.trace("The Eiffel Tower is in") as tracer:
    cache = tracer.cache()
    acts = model.transformer.h[9].output
    model.transformer.h[9].output = model.transformer.h[9].adapter(acts, hook=True)

"model.transformer.h.9.adapter" in cache.keys()          # True; False without hook=True
```

## Saving a cache to disk

A `CacheView` pickles without the model, so `torch.save` / `torch.load` don't drag
the weights along. Path access works on the loaded view; **tree navigation needs
the model re-attached**, or it raises
`AttributeError: 'NoneType' object has no attribute 'path'`:

```python
back = torch.load(path, weights_only=False)
back["model.transformer.h.0"].output       # works
back._cache.model = model                  # re-attach, then:
back.transformer.h[0].output               # works
```

## Gotchas

- **Declare `tracer.cache(...)` before reading or modifying a model value**, or it
  raises `ValueError: tracer.cache() must be declared before reading or modifying a
  model value`. It must be called inside a trace.
- **A kept cache keeps the model**, so `del model` frees no weights while a cache
  from it is alive. Detach the model first; path access keeps working:

  ```python
  cache._cache.model = None
  del model
  cache["model.transformer.h.0"].output      # still works
  ```
- **Wrap a collection run in `torch.no_grad()`** unless you need gradients, and
  keep the batch shape fixed across a sweep: the same prompt in differently shaped
  batches agrees to floating-point noise, not bit-exactly.
