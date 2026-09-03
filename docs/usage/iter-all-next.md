---
title: Iteration — iter / all
one_liner: Target intervention code at specific generation steps with `for step in tracer.iter[...]` and `tracer.all()`.
tags: [usage, generation, iteration]
related: [docs/usage/generate.md, docs/gotchas/iteration.md, docs/usage/source.md, docs/usage/invoke-and-batching.md]
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

## The one rule

**A loop must not ask for a step the run does not make.** A bound the run meets
is fine and the code after the loop runs. A bound it does not meet raises
`OutOfOrderError`, naming the iteration asked for and the count the run reached:

```python
with model.generate("Hello", max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[:10]:                # 10 steps of a 3-step run
        model.transformer.h[0].output[:] = 0
# OutOfOrderError: '...i3' was never reached: the loop asked for iteration 3 ...
```

`max_new_tokens` is an upper bound, so a bound matching it is safe only when
nothing ends the generation sooner. `min_new_tokens=N` suppresses EOS until N
tokens have been generated, which makes a bound of N hold:

```python
with model.generate("Hello", max_new_tokens=5, min_new_tokens=5) as tracer:
    picks = nnsight.save([])
    for step in tracer.iter[:5]:
        picks.append(model.lm_head.output[0, -1].argmax(dim=-1))
    ids = tracer.result.save()                   # runs
```

A stop string ends the run wherever it matches regardless, so pair
`stop_strings=` with an open loop.

### When the step count is unknown

`tracer.iter[:]` / `tracer.all()` end *by* asking for a step the run does not
make, so they warn rather than raise — and the same unwind still discards the
statements after the loop. Anything you need afterwards goes in a separate empty
invoke, which is its own worker:

```python
with model.generate(max_new_tokens=3, do_sample=False) as tracer:
    with tracer.invoke("Hello"):
        picks = nnsight.save([])
        for step in tracer.all():
            picks.append(model.lm_head.output[0, -1].argmax(dim=-1))
    with tracer.invoke():
        ids = tracer.result.save()               # runs
```

Values saved inside the loop survive either way. See
[../gotchas/iteration.md](../gotchas/iteration.md) for the full set of cases.

## Deprecated: the `with` form

`with tracer.iter[...]:` still works but emits an `NNsightDeprecationWarning`. It
does the same thing the long way — the loop is moved inside, re-running the block
per step — with one visible difference: because it owns the loop, it catches its
own over-run, truncates to the steps that ran, and lets the code after the block
run silently. Prefer `for step in tracer.iter[...]:`.

```python
with model.generate("Hello", max_new_tokens=3, do_sample=False) as tracer:
    with tracer.iter[:3]:            # NNsightDeprecationWarning
        x = model.transformer.h[0].output.save()
```

## Gotchas

- **An open loop whose body reads no module never returns.** Nothing parks the
  worker, so the index generator spins with no warning and no timeout. Bound the
  loop, or read something inside it.
- **Negative step values raise `ValueError`** (`tracer.iter step cannot be
  negative: -1`) — there is no "last step" shorthand.
- **Order the body the way the forward runs.** Reading layer 6 and then writing
  layer 2 inside the loop parks the write on the *next* step, so every
  intervention lands one step late (see
  [../gotchas/iteration.md](../gotchas/iteration.md)).
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
- [../gotchas/iteration.md](../gotchas/iteration.md) — every way a loop goes wrong.
- [source.md](source.md) — how `.source` interacts with iteration.
- [invoke-and-batching.md](invoke-and-batching.md) — per-invoke iteration.
