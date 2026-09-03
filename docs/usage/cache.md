---
title: Activation Cache
one_liner: tracer.cache() records module outputs (and optionally inputs) into a path- and attribute-addressable view.
tags: [usage, cache, intervention]
related: [docs/usage/access-and-modify.md, docs/usage/iter-all-next.md, docs/usage/save.md, docs/usage/invoke-and-batching.md]
sources: [src/nnsight/intervention/cache.py, src/nnsight/intervention/tracer.py]
---

# Activation Cache

## What this is for

`tracer.cache(...)` records the activations of many modules at once during a
trace. Because the interleaver already funnels every module input/output through
`Interleaver.handle` (applying interventions first), the cache is just a
**post-intervention observer** — it needs no per-module controllers of its own. It
captures *every* selected module across the whole run: every layer, and (in a
generation loop) every step.

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

`cache.keys()` lists the cached paths (all of them, at the root), in the order
the run reaches them — not the order you passed `modules=`.

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
        non_blocking=False,          # default: synchronous copy. True is async and
                                     # requires your own sync (see note below)
    )
```

!!! warning "`non_blocking=True` is opt-in, and unsafe unless you synchronise"

    The move to `device` is enqueued asynchronously and **nothing synchronises
    before you read the result**, so a capture off a CUDA device can be read while
    the copy is still in flight. Reading a CPU tensor does *not* synchronise CUDA.

    Measured on GPT-2, batch 128 x 64 tokens, 12 blocks captured, idle GPU:

    | | differs from a plain `register_forward_hook` |
    |---|---|
    | `non_blocking=True` | **10/10 batches**, up to 100% of elements wrong |
    | `non_blocking=False` (the default) | 0/10, bit-identical |

    The values are not permanently wrong -- they arrive late; a
    `torch.cuda.synchronize()` after the trace makes the same capture bit-exact.
    The window scales with copy size, so a small example looks fine while a
    corpus-scale run is silently corrupted. This is why the default is
    `False`: correctness costs roughly 15-20% throughput, and the failure it
    prevents is invisible.

    Pass `non_blocking=True` only if you synchronise yourself before reading.

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
    model.transformer.h[0].output[:] = 0

(cache["model.transformer.h.0"].output == 0).all()   # True
```

### Caching inside an invoke

A cache opened inside `tracer.invoke(...)` records that invoke's rows — at the
**whole batch's padded sequence length**, and nothing in the view marks which
positions are pad. A two-token prompt batched against an eight-token one comes
back as `[1, 8, 768]`, six of those positions padding:

```python
with model.trace() as tracer:
    with tracer.invoke("the cat"):                                     # 2 tokens
        short = tracer.cache(modules=[model.transformer.h[0]])
    with tracer.invoke("a much longer prompt here about many things"):  # 8 tokens
        pass

short["model.transformer.h.0"].output.shape          # torch.Size([1, 8, 768])
# the same prompt alone: torch.Size([1, 2, 768])
```

Pad positions are not blank, and at GPT-2's block 0 they carry a *larger* norm
than either real token — `192.4` at each of the six, against `136.7` and `56.7`
for "the" and " cat". A mean over the sequence axis of that capture is three
quarters padding. Reduce over the sequence axis only after masking, or index from
the right (`[:, -1]`), which is stable because the padding is on the left.
[invoke-and-batching.md](invoke-and-batching.md) has the mechanism and the mask.

To capture the **whole** combined batch instead, open the cache in an empty
`tracer.invoke()`, which sees every row:

```python
with model.trace() as tracer:
    with tracer.invoke("the cat"): pass
    with tracer.invoke("a much longer prompt here about many things"): pass
    with tracer.invoke():
        batch = tracer.cache(modules=[model.transformer.h[0]])

batch["model.transformer.h.0"].output.shape          # torch.Size([2, 8, 768])
```

### Gradients through a cache

`detach=True` (the default) stores values cut off from autograd. Pass
`detach=False`, and `device=None` so the tensors stay where the graph is, to get
captures you can call `backward()` on:

```python
with model.trace("the cat") as tracer:
    cache = tracer.cache(modules=[model.transformer.h[0]], detach=False, device=None)

cache["model.transformer.h.0"].output.requires_grad      # True
cache["model.transformer.h.0"].output.sum().backward()   # fills model .grad
```

This keeps the whole graph alive for as long as the cache is, so it costs the
memory a normal backward pass would. `detach=True` is the default for that reason.

### Modules called ad hoc

A cache records the modules the run reaches. Calling a module yourself inside the
trace body does not reach it unless you pass `hook=True`, which runs the module's
full `__call__`:

```python
model.transformer.h[9].adapter = MyAdapter()

with model.trace("The Eiffel Tower is in") as tracer:
    cache = tracer.cache()
    acts = model.transformer.h[9].output
    model.transformer.h[9].output[:] = model.transformer.h[9].adapter(acts, hook=True)

"model.transformer.h.9.adapter" in cache.keys()          # True; False without hook=True
```

Without `hook=True` neither the adapter nor its submodules appear in the cache —
which is the case where you most want them. See
[edit.md](edit.md) for attaching the module in the first place.

### Saving a cache to disk

A `CacheView` pickles: `Cache.__getstate__` drops the model, so `torch.save` /
`torch.load` round-trip without dragging the weights along. Path access works
straight away on the loaded view; **tree navigation needs the model re-attached**,
or it raises `AttributeError: 'NoneType' object has no attribute 'path'`:

```python
back = torch.load(path, weights_only=False)
back["model.transformer.h.0"].output       # works
back._cache.model = model                  # re-attach, then:
back.transformer.h[0].output               # works
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
    non_blocking=False,           # default; True is async but needs your own sync
)
```

Returns a `CacheView` (already saved, so it survives past the trace).

## Gotchas

- **Declare `tracer.cache(...)` before reading or modifying a model value.** Cache
  routes are fixed before the model starts; opening one after an activation access
  raises `ValueError: tracer.cache() must be declared before reading or modifying a
  model value`. Nothing is captured silently: the ordering is enforced, not assumed.
- **`tracer.cache()` must be called inside a trace.** It registers on the running
  worker's mediator; outside interleaving there is nothing to attach to.
- **Multiple visits accumulate into a list.** Across generation steps (or a
  shared-weight module hit twice), `cache[path].output` is a `list`. Don't assume a
  single tensor. Index the *value*, `cache[path].output[i]` — the view itself is
  not subscriptable by visit.
- **`cache.keys()` is in reached order.** `zip(my_modules, cache.keys())`
  misaligns whenever `modules=` was not already in forward order. Key off the path.
- **A cache opened inside an invoke records that invoke's rows only**, padded to
  the batch's length. An empty `tracer.invoke()` sees the whole batch. See
  [invoke-and-batching.md](invoke-and-batching.md).
- **The cache moves tensors to CPU by default.** Pass `device=None` (or a device)
  to keep them elsewhere.
- **Always pass `modules=` on a real model.** Caching all 151 of GPT-2's modules
  costs 1.1 GiB and 20x the time of caching its 12 blocks (36 MiB). A cache and a
  hand-written `save()` loop over the same modules cost the same (19.0 ms against
  18.8 ms, batch 32 x 64); the cache saves you the loop, not time.
- **Wrap a collection run in `torch.no_grad()`** unless you need gradients, and
  keep the batch shape fixed across a sweep: the same prompt in differently shaped
  batches gives activations that agree to floating-point noise, not bit-exactly.

## Related

- [access-and-modify.md](access-and-modify.md) — one-off `.output` / `.input`.
- [iter-all-next.md](iter-all-next.md) — generation-step iteration.
- [rename-modules.md](rename-modules.md) — alias-aware cache keys.
- [invoke-and-batching.md](invoke-and-batching.md) — rows, padding, and what an
  invoke's slice of the batch contains.
