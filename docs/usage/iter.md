---
title: Per-Step Iteration with tracer.iter
one_liner: tracer.iter[slice|int|list] runs intervention code at specific generation steps.
tags: [usage, iteration, generation]
related: [docs/usage/generate.md, docs/usage/all-and-next.md, docs/gotchas/iteration.md]
sources: [src/nnsight/intervention/tracing/iterator.py:55, src/nnsight/intervention/tracing/iterator.py:184, src/nnsight/intervention/tracing/iterator.py:95]
---

# Per-Step Iteration with `tracer.iter`

## What this is for

`tracer.iter[...]` controls which generation step a block of intervention code targets. Each forward pass through the model is one step. Inside the loop body, every `.output` / `.input` access is bound to the current step.

The mechanism: `IteratorTracer.__iter__` sets `mediator.iteration = i` before each yield. One-shot intervention hooks compare against `mediator.iteration_tracker[path]`, which is bumped after every forward pass by persistent "tracker-bumping" hooks installed at loop entry (`src/nnsight/intervention/tracing/iterator.py:95`).

## When to use / when not to use

- Use when you want a different intervention on different generation steps.
- Use when you want to collect a value at every step (`tracer.iter[:]`).
- Skip for a single-step trace — plain `.trace(input)` already targets step 0.
- Skip when you want a blanket recursion of the same intervention across all steps without caring about the step index — `tracer.all()` is the same as `tracer.iter[:]` and reads cleaner. See [all-and-next](all-and-next.md).

## Canonical pattern

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    logits = list().save()

    for step in tracer.iter[:]:                    # all steps
        logits.append(model.lm_head.output[0][-1].argmax(dim=-1))
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
            model.transformer.h[0].output[0][:] = 0
        # other steps pass through unchanged
```

## How it works

`tracer.iter` returns an `IteratorProxy`; subscripting returns an `IteratorTracer` (`src/nnsight/intervention/tracing/iterator.py:55`). On `__iter__`:

1. **Persistent iter hooks register**, one per wrapped module, with `mediator_idx = float('inf')` so they fire AFTER any intervention or cache hook (`src/nnsight/intervention/tracing/iterator.py:172`).
2. For each step `i`: set `mediator.iteration = i`, yield `i` to the loop body, then advance.
3. After the body runs and the model executes for the step, the iter hooks bump `iteration_tracker[<path>.input]` and `iteration_tracker[<path>.output]` for every module — and recursively for every operation under any `SourceAccessor`.
4. On exit (normal or exception), iter hooks are removed in `finally` and `mediator.iteration` is restored.

## Gotchas

- **`tracer.iter[:]` is unbounded — code AFTER the loop is skipped.** The unbounded iterator never receives a "stop" signal once generation finishes; it sits waiting for the next iteration that never comes. nnsight emits a warning, but the lines after the loop never execute. Either pass the input to a separate empty `tracer.invoke()` after the loop, use bounded iteration (`tracer.iter[:N]`), or use `tracer.next()`. See [docs/gotchas/iteration.md](../gotchas/iteration.md).
- **Iter loops cannot be entered with `with`.** `with tracer.iter[...]:` is deprecated and emits a `DeprecationWarning`. Always use `for step in tracer.iter[...]:` (`src/nnsight/intervention/tracing/iterator.py:320`).
- **Negative iteration values raise `ValueError`** — there is no "last step" shorthand.
- **First-time `.source` access mid-loop misses one step.** If the first ever `.source` access on a module happens at step N>0, that step's operation hooks miss because the operation tracker is still at 0. Touch `.source` before the loop. See [source.md § Gotchas](source.md#gotchas).
- **`WrapperModule`s (e.g. `generator`, `streamer`) are skipped** by the iter hooks. Their values are pushed via `eproperty.provide`, which bumps the tracker itself.

## Related

- [all-and-next](all-and-next.md) — `tracer.all()` (= `tracer.iter[:]`) and manual stepping with `.next()`.
- [generate](generate.md) — Generation context.
- [docs/concepts/threading-and-mediators.md](../concepts/threading-and-mediators.md) — How `mediator.iteration` flows.
