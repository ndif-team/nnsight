---
title: Activation Cache
one_liner: tracer.cache() records module outputs (and optionally inputs) into a dict-like object that survives generation steps.
tags: [usage, cache, intervention]
related: [docs/usage/access-and-modify.md, docs/usage/iter.md, docs/usage/save.md]
sources: [src/nnsight/intervention/tracing/tracer.py:28, src/nnsight/intervention/tracing/tracer.py:465, src/nnsight/intervention/hooks.py:356, src/nnsight/intervention/hooks.py:397]
---

# Activation Cache

## What this is for

`tracer.cache(...)` registers persistent forward hooks on the target modules and accumulates their outputs (and optionally inputs) into a `Cache.CacheDict`. Unlike the one-shot hooks behind `.output` / `.input`, cache hooks fire on every forward pass and are assigned `mediator_idx = float('inf')` so they always fire **after** any intervention hooks — the cache captures **post-intervention** values (`src/nnsight/intervention/hooks.py:390`).

The cache is the right tool when you want activations from many modules in one shot, or activations across all generation steps without writing per-step `.save()` calls.

## When to use / when not to use

- Use when you want the same value from many modules.
- Use when you want activations across every generation step (cache appends across steps automatically).
- Use when you want post-intervention values (cache hooks run after intervention hooks).
- Skip when you only need one value — `module.output.save()` is simpler and avoids hooking the whole tree.

## Canonical pattern

```python
with model.trace("Hello") as tracer:
    cache = tracer.cache()  # Cache every module by default

# Dict access (envoy paths):
print(cache['model.transformer.h.0'].output)

# Or attribute access:
print(cache.model.transformer.h[0].output)
```

## Variations

### Cache a subset of modules

Pass either Envoy objects or path strings:

```python
with model.trace("Hello") as tracer:
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

inputs = cache['model.transformer.h.0'].inputs   # (args, kwargs)
first  = cache['model.transformer.h.0'].input    # First positional/keyword arg
```

### Storage transforms

```python
with model.trace("Hello") as tracer:
    cache = tracer.cache(
        device=torch.device("cpu"),  # Move to CPU before storing (default)
        dtype=torch.float32,         # Cast (default: keep)
        detach=True,                 # Detach from autograd (default)
        include_output=True,
        include_inputs=False,
    )
```

### Cache across generation steps

Cache hooks are persistent for the lifetime of the interleaver. They append a new `Cache.Entry` per forward pass when the same path is hit twice:

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    cache = tracer.cache(modules=[model.transformer.h[-1]])

# cache['model.transformer.h.11'] is now a list of 5 Entry objects
# (one per generation step) since each step hit the same path.
```

### Cache + interventions

Cache hooks run after intervention hooks (`mediator_idx=inf`), so caches see post-intervention values:

```python
with model.trace("Hello") as tracer:
    cache = tracer.cache()
    model.transformer.h[0].output[0][:] = 0

# cache['model.transformer.h.0'].output[0] is all zeros
```

## API

```python
tracer.cache(
    modules=None,           # None | List[Envoy | str]; None = all modules
    device=torch.device("cpu"),
    dtype=None,
    detach=True,
    include_output=True,
    include_inputs=False,
)
```

Returns a `Cache.CacheDict` (already wrapped in `.save()`). Hook handles live on `mediator.hooks` and are removed automatically when the interleaver exits (`src/nnsight/intervention/tracing/tracer.py:546`).

## Gotchas

- **Call `tracer.cache(...)` BEFORE the interventions you want it to capture.** Cache hooks always fire last on a given module, but they only attach for modules registered at cache creation time.
- **`tracer.cache()` must be called inside an interleaving context** (i.e. inside `.trace()` / `.generate()` / `.session()`); calling it at module construction time raises `ValueError("Cannot create a cache outside an invoker.")` (`src/nnsight/intervention/tracing/tracer.py:501`).
- **Repeated forward hits accumulate.** When the same module fires twice (e.g. across generation steps, or shared-weight modules), the entry becomes a `list[Entry]`. Don't assume a single `Entry` per path.
- **The cache moves tensors to CPU by default.** If you need them on GPU, pass `device=None` or the desired device.
- See [docs/gotchas/save.md](../gotchas/save.md) for the full set.

## Related

- [access-and-modify](access-and-modify.md) — One-off `.output` / `.input` access.
- [iter](iter.md) — Iteration semantics for generation.
- [docs/concepts/interleaver-and-hooks.md](../concepts/interleaver-and-hooks.md) — Why cache hooks fire last.
