---
title: tracer.all() and .next()
one_liner: Apply interventions to every generation step (tracer.all()) or step manually (tracer.next() / module.next()).
tags: [usage, iteration, generation]
related: [docs/usage/iter.md, docs/usage/generate.md, docs/gotchas/iteration.md]
sources: [src/nnsight/intervention/tracing/tracer.py:457, src/nnsight/intervention/tracing/tracer.py:460, src/nnsight/intervention/envoy.py:439]
---

# `tracer.all()` and `.next()`

## What this is for

Two convenience APIs on top of `tracer.iter`:

- `tracer.all()` — equivalent to `tracer.iter[:]`. Reads as "do this for every step." See `src/nnsight/intervention/tracing/tracer.py:457`.
- `tracer.next(step=1)` / `module.next(step=1)` — manually advance the mediator's `iteration` counter. Useful when you want to access the same module at multiple steps within a single straight-line block (no loop).

## When to use / when not to use

- Use `tracer.all()` when you want the SAME intervention recursively on every step and you don't need the step index.
- Use `tracer.next()` for a small fixed number of explicit steps where a `for` loop adds noise.
- For varying interventions per step, or when you want to collect step-indexed values, use `tracer.iter[...]`. See [iter.md](iter.md).
- For a single-step trace, neither is needed — `.trace()` is one step.

## Canonical pattern

### `tracer.all()`

```python
with model.generate("Hello", max_new_tokens=3) as tracer:
    hidden_states = list().save()

    for step in tracer.all():                          # = tracer.iter[:]
        model.transformer.h[0].output[0][:] = 0        # zero-ablate every step
        hidden_states.append(model.transformer.h[-1].output[0])
```

### `module.next()` for manual stepping

```python
with model.generate("Hello", max_new_tokens=3) as tracer:
    hs0 = model.transformer.h[-1].output[0].save()                # step 0
    hs1 = model.transformer.h[-1].next().output[0].save()         # step 1
    hs2 = model.transformer.h[-1].next().output[0].save()         # step 2
```

`module.next(step=1)` increments `mediator.iteration` by `step` and returns the same Envoy, so the next `.output` / `.input` access lands on the new step.

### `tracer.next()` (equivalent at the tracer level)

```python
with model.generate("Hello", max_new_tokens=3) as tracer:
    hs0 = model.transformer.h[-1].output[0].save()
    tracer.next()                                                  # advance to step 1
    hs1 = model.transformer.h[-1].output[0].save()
    tracer.next()                                                  # advance to step 2
    hs2 = model.transformer.h[-1].output[0].save()
```

`tracer.next(step=1)` does `self.model.interleaver.current.iteration += step` and returns the tracer (`src/nnsight/intervention/tracing/tracer.py:460`).

## When to choose `iter[]` vs `next()`

| Situation                                            | Use                |
|------------------------------------------------------|--------------------|
| Same intervention every step                         | `tracer.all()`     |
| Want the step index in your code                     | `tracer.iter[:]`   |
| Subset of steps (slice / list / single int)          | `tracer.iter[...]` |
| 2–3 explicit steps, straight-line code               | `.next()`          |
| Code AFTER the iter block must run (post-loop logic) | `.next()` or bounded `tracer.iter[:N]` (see Gotchas) |

## Gotchas

- **`tracer.all()` is unbounded — code after the loop never runs**, just like `tracer.iter[:]`. Use a separate empty `tracer.invoke()` after the loop, or use `.next()` instead. See [docs/gotchas/iteration.md](../gotchas/iteration.md).
- **`module.next()` is deprecated; prefer `tracer.next()`.** `model.transformer.h[-1].next()` still works but emits `DeprecationWarning` (`src/nnsight/intervention/envoy.py:441`). The behavior is identical — both bump the same `mediator.iteration`.
- **`.next()` only advances the iteration counter.** It does not block until the model has actually run that step. The model still runs forward in lockstep with hook resolution; `.next()` just tells the mediator which step's hooks to register next.
- **Don't mix `.next()` with an `iter[...]` loop.** Inside a `for step in tracer.iter[...]` block, `mediator.iteration` is overwritten on each yield. Calling `.next()` inside the loop will be clobbered on the next iteration.

## Related

- [iter](iter.md) — Per-step targeting with explicit step indices.
- [generate](generate.md) — Generation context.
- [docs/concepts/threading-and-mediators.md](../concepts/threading-and-mediators.md) — How `mediator.iteration` and `iteration_tracker` interact.
