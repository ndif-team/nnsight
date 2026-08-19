---
title: Iteration — iter / all
one_liner: Target intervention code at specific generation steps with `for step in tracer.iter[...]` and `tracer.all()`.
tags: [usage, generation, iteration]
related: [docs/usage/generate.md, docs/usage/source.md, docs/usage/invoke-and-batching.md]
sources: [src/nnsight/intervention/iterator.py, src/nnsight/intervention/tracer.py, src/nnsight/intervention/interleaver.py]
---

# Iteration — `tracer.iter` / `tracer.all()`

## What this is for

In `model.generate(...)`, the model runs forward once per generated token, so each
module is reached once per step. **Iteration APIs bind a stretch of trace body to a
chosen range of those steps:**

- `for step in tracer.iter[slice | int | list]:` — loop the body over the selected
  step(s). Inside the body, every `.output` / `.input` read binds to the current
  step, and `step` is the real integer index.
- `tracer.all()` — shorthand for `tracer.iter[:]` (every step).

For a single forward (`model.trace(x)`) there is only step 0, so you don't need
these.

> Note: the old `tracer.next()` / `module.next()` manual-stepping API is **gone**
> in this rewrite. Step targeting is done with the loop forms below.

## Canonical pattern

### `tracer.iter[:N]` — every step, with the step index

```python
import nnsight
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.generate("Hello", max_new_tokens=3, do_sample=False) as tracer:
    toks = nnsight.save([])
    for step in tracer.iter[:3]:                 # steps 0, 1, 2
        toks.append(model.lm_head.output[0, -1].argmax(dim=-1))
    ids = tracer.result.save()                   # final generated ids
# len(toks) == 3, ids.shape == (1, 4)   (prompt token + 3 generated)
```

### `tracer.all()` — every step, no index needed

```python
with model.generate("Hello", max_new_tokens=3, do_sample=False) as tracer:
    hidden = nnsight.save([])
    for step in tracer.all():                     # == tracer.iter[:]
        model.transformer.h[0].output[:] = 0   # zero-ablate layer 0 every step
        hidden.append(model.transformer.h[-1].output)
```

## Variations

### Slice — bounded range

```python
with model.generate("Hello", max_new_tokens=5, do_sample=False) as tracer:
    out = nnsight.save([])
    for step in tracer.iter[1:3]:                  # steps 1 and 2 only
        out.append(model.lm_head.output)
```

### Int — single step

```python
with model.generate("Hello", max_new_tokens=5, do_sample=False) as tracer:
    for step in tracer.iter[0]:                    # only the prefill step
        first = model.lm_head.output.save()
```

### List — explicit steps

```python
with model.generate("Hello", max_new_tokens=5, do_sample=False) as tracer:
    out = nnsight.save([])
    for step in tracer.iter[[0, 2, 4]]:            # those steps only
        out.append(model.lm_head.output)
```

### Per-step conditional

`step` is the actual integer index, so a plain Python `if` works:

```python
with model.generate("Hello", max_new_tokens=5, do_sample=False) as tracer:
    for step in tracer.iter[:5]:
        if step == 2:
            model.transformer.h[0].output[:] = 0
        # other steps pass through
```

## How it works

`tracer.iter` returns an `Iterations` object; subscripting selects the range
(`iterator.py`). Looping over it walks the running mediator's `iteration` pointer
across the selected steps — before each yield it pins `iteration` so the first read
in the body binds to that occurrence. Whatever `iteration` was before the loop is
restored on exit, so loops can nest. `tracer.all()` is `tracer.iter[:]`.

## Deprecated: the `with` form

`with tracer.iter[...]:` still works but emits a `DeprecationWarning`. It does the
same thing the long way — the loop is moved inside, re-running the block per step.
Prefer `for step in tracer.iter[...]:`.

```python
with model.generate("Hello", max_new_tokens=3, do_sample=False) as tracer:
    with tracer.iter[:3]:            # DeprecationWarning
        x = model.transformer.h[0].output.save()
```

## Gotchas

- **Open-ended `iter[:]` / `all()` do not let trailing code run.** They loop until
  the model stops generating; the final over-run request is thrown into the worker
  as `OutOfOrderError` (caught and warned), which unwinds the loop **and every line
  after it**. So `tracer.result.save()` placed *after* an open-ended loop never
  runs. To capture per-step values *and* the final result, use a **bounded**
  `iter[:N]` matching `max_new_tokens` — then trailing code runs (see the canonical
  pattern above). This differs from old nnsight, which special-cased this via a
  `default_all` bound.
- **`max_new_tokens` is a cap, not a guarantee.** If the model stops early (EOS /
  stop string), steps that didn't happen warn `'...' was never reached: the model
  ran fewer iterations than the loop requested. Values from reached iterations are
  kept.`
- **Negative step values raise `ValueError`** (`tracer.iter step cannot be
  negative: -1`) — there is no "last step" shorthand.
- **Regular-module access after the loop is out of order.** Those forward passes are
  already done, so requesting a module's `.output`/`.input` after the loop raises
  `OutOfOrderError`.
- **Source-op iteration counts invocations, not forward passes.** An op that fires
  once per forward is indexed per generation step; an op that loops within one
  forward (e.g. an MoE expert loop) is indexed per fire. See [source.md](source.md).
- **`model.iter` / `model.all()` are deprecated** — use `tracer.iter` /
  `tracer.all()`.

## Related

- [generate.md](generate.md) — generation context.
- [source.md](source.md) — how `.source` interacts with iteration.
- [invoke-and-batching.md](invoke-and-batching.md) — per-invoke iteration.
