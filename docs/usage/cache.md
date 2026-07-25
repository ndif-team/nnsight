---
title: Activation Cache
one_liner: tracer.cache() records module outputs (and optionally inputs) into a path- and attribute-addressable view.
tags: [usage, cache, intervention]
related: [docs/usage/access-and-modify.md, docs/usage/iter-all-next.md, docs/usage/save.md]
sources: [src/nnsight/intervention/cache.py, src/nnsight/intervention/tracer.py]
---

# Activation Cache

## What this is for

`tracer.cache(...)` records the activations of many modules at once during a
trace. Because the interleaver already funnels every module input/output through
`Interleaver.handle` (applying interventions first), the cache is just a
**post-intervention observer** — it needs no per-module hooks. It captures *every*
selected module across the whole run: every layer, and (in a generation loop)
every step.

Reach for it when you want the same value from many modules, or activations across
all generation steps without writing per-step `.save()` calls.

## When to use / when not to use

- Use when you want the same value from many modules.
- Use when you want activations across every generation step (cache appends one
  entry per step automatically).
- Use when you want post-intervention values (the cache observes after
  interventions apply).
- Skip when you only need one value — `module.output.save()` is simpler.

## Canonical pattern

```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("The Eiffel Tower is in") as tracer:
    cache = tracer.cache()                          # every module's output

# By path:
cache["model.transformer.h.0"].output              # a tensor
# Or by navigating the tree:
cache.transformer.h[0].output                      # same value
```

`cache.keys()` lists the cached paths (all of them, at the root).

## Variations

### Cache a subset of modules

Pass Envoy objects or path strings:

```python
with model.trace("The Eiffel Tower is in") as tracer:
    cache = tracer.cache(modules=[
        model.transformer.h[0],
        model.transformer.h[5],
        "model.lm_head",
    ])
```

### Include inputs

```python
with model.trace("Hello") as tracer:
    cache = tracer.cache(include_inputs=True)

cache["model.transformer.h.0"].inputs   # (args, kwargs)
cache["model.transformer.h.0"].input    # first positional/keyword arg
```

Without `include_inputs=True`, `.inputs` is `None`.

### Storage transforms

```python
import torch
with model.trace("Hello") as tracer:
    cache = tracer.cache(
        device=torch.device("cpu"),  # move captured tensors here (default CPU); None = leave
        dtype=torch.float32,         # optional cast (default: keep)
        detach=True,                 # detach from autograd (default)
        include_output=True,
        include_inputs=False,
        non_blocking=True,           # async device transfer (default); set False to
                                     # synchronize the copy (see note below)
    )
```

`non_blocking=True` (the default) makes the move to `device` asynchronous, which is
faster and safe under nnsight's single-stream execution — captured values are read
after the run (and any Python read syncs anyway). Set `non_blocking=False` only if
you move captured tensors across CUDA streams yourself and need the copy finished
before another stream reads them.

### Cache across generation steps

A module reached once per step accumulates one entry per step:

```python
with model.generate("Hello", max_new_tokens=3, do_sample=False) as tracer:
    cache = tracer.cache(modules=[model.transformer.h[-1]])

len(cache["model.transformer.h.11"])            # 3 (one per step)
cache["model.transformer.h.11"].output          # a list of 3 tensors
```

### Single vs multiple visits

`cache[path].output` unwraps automatically: a **single visit** returns the value
directly (a tensor), **multiple visits** return a `list`. `len(cache[path])` is the
visit count.

```python
# single forward -> one visit -> tensor
with model.trace("Hello") as tracer:
    cache = tracer.cache(modules=[model.transformer.h[-1]])
type(cache["model.transformer.h.11"].output).__name__   # 'Tensor'

# generation -> N visits -> list of tensors
with model.generate("Hello", max_new_tokens=3, do_sample=False) as tracer:
    cache = tracer.cache(modules=[model.transformer.h[-1]])
isinstance(cache["model.transformer.h.11"].output, list)   # True
```

### Cache + interventions

The cache observes post-intervention values:

```python
with model.trace("Hello") as tracer:
    cache = tracer.cache()
    model.transformer.h[0].output[0][:] = 0

(cache["model.transformer.h.0"].output == 0).all()   # True
```

### Cache honors renames

Alias navigation resolves against the model's envoy tree, so renamed modules and
`ModuleList` indices work in cache keys too:

```python
g = TransformersModel("openai-community/gpt2", dispatch=True, rename={"mlp": "my_mlp"})
with g.trace("Hello") as tracer:
    cache = tracer.cache()
torch.equal(cache.transformer.h[0].my_mlp.output,
            cache["model.transformer.h.0.mlp"].output)   # True
```

See [rename-modules.md](rename-modules.md).

## API

```python
tracer.cache(
    modules=None,                 # None | list[Envoy | str]; None = every module
    device=torch.device("cpu"),   # None leaves tensors where they are
    dtype=None,
    detach=True,
    include_output=True,
    include_inputs=False,
    non_blocking=True,            # async device transfer; False synchronizes it
)
```

Returns a `CacheView` (already saved, so it survives past the trace).

## Gotchas

- **Only modules reached *after* the `tracer.cache(...)` call are captured.** Call
  it early (right after opening the trace).
- **`tracer.cache()` must be called inside a trace.** It registers on the running
  worker's mediator; outside interleaving there is nothing to attach to.
- **Multiple visits accumulate into a list.** Across generation steps (or a
  shared-weight module hit twice), `cache[path].output` is a `list`. Don't assume a
  single tensor. Index into the view for a specific visit.
- **A cache opened inside an invoke records that invoke's rows only** — not the
  whole combined batch. See [invoke-and-batching.md](invoke-and-batching.md).
- **The cache moves tensors to CPU by default.** Pass `device=None` (or a device)
  to keep them elsewhere.

## Related

- [access-and-modify.md](access-and-modify.md) — one-off `.output` / `.input`.
- [iter-all-next.md](iter-all-next.md) — generation-step iteration.
- [rename-modules.md](rename-modules.md) — alias-aware cache keys.
