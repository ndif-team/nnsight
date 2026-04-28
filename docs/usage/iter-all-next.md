---
title: Iteration — iter / all / next
one_liner: Generation-step control via tracer.iter[...], tracer.all(), and module.next() / tracer.next().
tags: [usage, generation, iteration]
related: [docs/usage/generate.md, docs/concepts/source-tracing.md, docs/gotchas/iteration.md]
sources: [src/nnsight/intervention/tracing/iterator.py, src/nnsight/intervention/envoy.py, src/nnsight/intervention/interleaver.py]
---

# Iteration — `tracer.iter` / `tracer.all()` / `.next()`

## What this is for

In `model.generate(...)`, the model runs forward once per generated token. **Iteration APIs target intervention code at specific generation steps**:

- `tracer.iter[slice|int|list]` — pick which step(s) a block of intervention code targets.
- `tracer.all()` — equivalent to `tracer.iter[:]`. Reads as "do this for every step."
- `tracer.next(step=1)` / `module.next(step=1)` — manually advance the mediator's `iteration` counter for straight-line code without a loop.

Inside an iter loop body, every `.output` / `.input` access is bound to the current step.

The mechanism: `IteratorTracer.__iter__` sets `mediator.iteration = i` before each yield. One-shot intervention hooks compare against `mediator.iteration_tracker[path]`, which is bumped after every forward pass by persistent "tracker-bumping" hooks installed at loop entry (`src/nnsight/intervention/tracing/iterator.py:95`).

## When to use / when not to use

- Use `tracer.iter[...]` when you want different interventions on different generation steps, or when you need the step index in your code.
- Use `tracer.all()` when you want the SAME intervention recursively on every step and you don't need the step index.
- Use `tracer.next()` for a small fixed number of explicit steps where a `for` loop adds noise.
- Skip all of these for a single-step trace — plain `.trace(input)` already targets step 0.

## Canonical pattern

### `tracer.iter[:]` — every step, with the step index

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    logits = list().save()

    for step in tracer.iter[:]:                    # all steps
        logits.append(model.lm_head.output[0][-1].argmax(dim=-1))
```

### `tracer.all()` — every step, no step index needed

```python
with model.generate("Hello", max_new_tokens=3) as tracer:
    hidden_states = list().save()

    for step in tracer.all():                          # = tracer.iter[:]
        model.transformer.h[0].output[:] = 0        # zero-ablate every step
        hidden_states.append(model.transformer.h[-1].output)
```

## Variations

### Slice — bounded range

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    out = list().save()
    for step in tracer.iter[1:3]:                  # steps 1 and 2 only
        out.append(model.lm_head.output)
```

### Int — single step

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    for step in tracer.iter[0]:                    # only the prefill step
        first = model.lm_head.output.save()
```

### List — explicit steps

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    out = list().save()
    for step in tracer.iter[[0, 2, 4]]:            # specific steps
        out.append(model.lm_head.output)
```

### Per-step conditional

`step` is the actual integer iteration number, so a normal Python `if` works:

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    for step in tracer.iter[:]:
        if step == 2:
            model.transformer.h[0].output[:] = 0
        # other steps pass through unchanged
```

### `tracer.next()` — manual stepping at the tracer

```python
with model.generate("Hello", max_new_tokens=3) as tracer:
    hs0 = model.transformer.h[-1].output.save()
    tracer.next()                                                  # advance to step 1
    hs1 = model.transformer.h[-1].output.save()
    tracer.next()                                                  # advance to step 2
    hs2 = model.transformer.h[-1].output.save()
```

`tracer.next(step=1)` does `self.model.interleaver.current.iteration += step` and returns the tracer (`src/nnsight/intervention/tracing/tracer.py:460`).

### `module.next()` — manual stepping at the module (alias, deprecated)

```python
with model.generate("Hello", max_new_tokens=3) as tracer:
    hs0 = model.transformer.h[-1].output.save()                # step 0
    hs1 = model.transformer.h[-1].next().output.save()         # step 1
    hs2 = model.transformer.h[-1].next().output.save()         # step 2
```

`module.next(step=1)` is an alias for `tracer.next(step=1)` — both bump the same `mediator.iteration`. The `module.next()` form **emits a `DeprecationWarning`** (`src/nnsight/intervention/envoy.py:440`); prefer `tracer.next()`.

## When to choose `iter[]` vs `next()`

| Situation                                            | Use                |
|------------------------------------------------------|--------------------|
| Same intervention every step                         | `tracer.all()`     |
| Want the step index in your code                     | `tracer.iter[:]`   |
| Subset of steps (slice / list / single int)          | `tracer.iter[...]` |
| 2–3 explicit steps, straight-line code               | `tracer.next()`    |
| Code AFTER the iter block must run (post-loop logic) | `tracer.next()` or bounded `tracer.iter[:N]` (see Gotchas) |

## How it works

`tracer.iter` returns an `IteratorProxy`; subscripting returns an `IteratorTracer` (`src/nnsight/intervention/tracing/iterator.py:55`). On `__iter__`:

1. **Persistent iter hooks register**, one per wrapped module, with `mediator_idx = float('inf')` so they fire AFTER any intervention or cache hook (`src/nnsight/intervention/tracing/iterator.py:172`).
2. For each step `i`: set `mediator.iteration = i`, yield `i` to the loop body, then advance.
3. After the body runs and the model executes for the step, the iter hooks bump `iteration_tracker[<path>.input]` and `iteration_tracker[<path>.output]` for every module — and recursively for every operation under any `SourceAccessor`.
4. On exit (normal or exception), iter hooks are removed in `finally` and `mediator.iteration` is restored.

`tracer.all()` is implemented as `tracer.iter[:]` (`src/nnsight/intervention/tracing/tracer.py:457`).

## Gotchas

- **`tracer.iter[:]` and `tracer.all()` are unbounded — code AFTER the loop is skipped.** The unbounded iterator never receives a "stop" signal once generation finishes; it sits waiting for the next iteration that never comes. nnsight emits a warning, but the lines after the loop never execute. Either pass the input to a separate empty `tracer.invoke()` after the loop, use bounded iteration (`tracer.iter[:N]`), or use `tracer.next()`. See [docs/gotchas/iteration.md](../gotchas/iteration.md).
- **Iter loops cannot be entered with `with`.** `with tracer.iter[...]:` is deprecated and emits a `DeprecationWarning`. Always use `for step in tracer.iter[...]:` (`src/nnsight/intervention/tracing/iterator.py:320`).
- **Negative iteration values raise `ValueError`** — there is no "last step" shorthand.
- **First-time `.source` access mid-loop misses one step.** If the first ever `.source` access on a module happens at step N>0, that step's operation hooks miss because the operation tracker is still at 0. Touch `.source` before the loop. See [source.md § Gotchas](source.md#gotchas).
- **`WrapperModule`s (e.g. `generator`, `streamer`) are skipped** by the iter hooks. Their values are pushed via `eproperty.provide`, which bumps the tracker itself.
- **`module.next()` is deprecated; prefer `tracer.next()`.** `model.transformer.h[-1].next()` still works but emits `DeprecationWarning` (`src/nnsight/intervention/envoy.py:440`). The behavior is identical — both bump the same `mediator.iteration`.
- **`.next()` only advances the iteration counter.** It does not block until the model has actually run that step. The model still runs forward in lockstep with hook resolution; `.next()` just tells the mediator which step's hooks to register next.
- **Don't mix `.next()` with an `iter[...]` loop.** Inside a `for step in tracer.iter[...]` block, `mediator.iteration` is overwritten on each yield. Calling `.next()` inside the loop will be clobbered on the next iteration.

## Related

- [generate](generate.md) — Generation context.
- [docs/concepts/source-tracing.md](../concepts/source-tracing.md) — How `.source` interacts with iteration.
- [docs/gotchas/iteration.md](../gotchas/iteration.md) — The unbounded-iter trailing-code footgun.
- [docs/concepts/threading-and-mediators.md](../concepts/threading-and-mediators.md) — How `mediator.iteration` and `iteration_tracker` interact.
