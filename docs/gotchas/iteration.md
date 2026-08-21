---
title: Iteration Pitfalls (iter, all)
one_liner: Multi-step generation footguns — the for-loop form, unbounded iter[:] dropping trailing code, all() = iter[:], and per-location occurrence counting.
tags: [gotcha, generate, iter, all]
related: [docs/usage/iter-all-next.md, docs/usage/generate.md]
sources: [src/nnsight/intervention/iterator.py, src/nnsight/intervention/interleaver.py:605]
---

# Iteration Pitfalls

## TL;DR
- The loop form is `for step in tracer.iter[...]:` (or `for step in tracer.all():`). The old `with tracer.iter[...]:` block still works but is **deprecated** and warns.
- **`tracer.next()` / `module.next()` do not exist.** Use `tracer.iter[i]` / `tracer.iter[[i, j]]` to target specific steps.
- Unbounded `tracer.iter[:]` (and `tracer.all()`, which *is* `iter[:]`) runs until the model stops producing steps. The final loop iteration asks for a step the model never runs, so the worker is **unwound there** — **any code after the loop in the same block does NOT run** (a warning is emitted, not an error). This is true even inside `generate(...)`; there is no `default_all` bound.
- To run code after the loop, move the trailing code into a **separate empty `tracer.invoke()`**. **Bounding the loop (`tracer.iter[:N]`) is not a reliable fix**: if the model stops early — EOS, a stop string, any generation shorter than `N` — the bounded loop parks on a step that never runs and drops the trailing code exactly as the unbounded form does (it warns: `'model.output.iN' was never reached`). Since `max_new_tokens` is only ever an *upper* bound, a bound cannot guarantee the loop completes. The empty-invoke form always works.
- Values collected *during* the loop are kept even when the loop is unbounded — only trailing code is dropped.
- `tracer.iter[N]` targets the `(N+1)`-th **occurrence** of a location. For a top-level block (once per step) that equals "step N"; for a module called several times per step, it's the call count.

---

## Unbounded `tracer.iter[:]` drops trailing code

### Symptom
Code written after a `for ... in tracer.iter[:]:` loop never runs — a variable assigned there is `UnboundLocalError`/undefined afterward. nnsight warns:
```
'model.transformer.h.-1.output.i3' was never reached: the model ran fewer iterations than the loop requested. Values from reached iterations are kept.
```

### Cause
`tracer.iter[:]` has no stop, so the loop keeps handing out step indices. When the model has generated its last token, the next loop iteration parks on a location the model never reaches; at run end `check_dangling_mediators` (`src/nnsight/intervention/interleaver.py:605`) throws `OutOfOrderError` into the worker to unwind it (running `finally` blocks) and **warns** instead of raising. Because the worker is unwound at the loop, everything written after the loop is skipped. `tracer.all()` returns `self.iter[:]`, so it behaves identically.

There is no `default_all` mechanism — `max_new_tokens` does not turn `iter[:]` into a bounded, cleanly-terminating loop.

### Wrong code
```python
with model.generate("The Eiffel Tower is in", max_new_tokens=3) as tracer:
    steps = nnsight.save([])
    for _ in tracer.iter[:]:
        steps.append(model.transformer.h[-1].output[:, -1, :])
    ids = tracer.result.save()   # NEVER runs — `ids` is undefined afterward
# steps is populated (3 entries); ids is not defined
```

### Right code (RECOMMENDED — separate empty invoke)
The empty invoke is its own worker on the same batch, so its code runs regardless of the loop:
```python
with model.generate(max_new_tokens=3) as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        steps = nnsight.save([])
        for _ in tracer.iter[:]:
            steps.append(model.transformer.h[-1].output[:, -1, :])
    with tracer.invoke():
        ids = tracer.result.save()      # safe — runs in its own worker
```

### Right code (bounded iter — only when you don't need early-stop robustness)
```python
with model.generate("The Eiffel Tower is in", max_new_tokens=3) as tracer:
    steps = nnsight.save([])
    for _ in tracer.iter[:3]:
        steps.append(model.transformer.h[-1].output[:, -1, :])
    ids = tracer.result.save()          # runs — the loop terminated
```
If the model stops early (EOS/stop strings) before step 3, the un-reached iterations warn but trailing code still runs.

### Mitigation / how to spot it early
- See `'...' was never reached` in the warnings, or a variable defined after an `iter[:]` loop that's missing? This is it.
- `tracer.all()` is the same as `iter[:]` — same fix.

---

## `tracer.iter[N]` counts occurrences, not always generation steps

### Symptom
You expect `tracer.iter[2]` to mean "the 3rd generation step" for every module, but for a module called multiple times per step it targets a different call.

### Cause
Each visit to a location is tagged with its occurrence index; `tracer.iter[N]` binds the request to occurrence `N` (`src/nnsight/intervention/iterator.py`). A top-level transformer block fires once per generation step, so occurrence `N` == step `N`. A module called `k` times within one forward (a recurrent inner module, or one called in a loop) fires `k` occurrences per step, so occurrence `N` lands somewhere inside a step.

### Right code
```python
# a top-level block, once per step — iter[2] is generation step 2
with model.generate("Hello", max_new_tokens=3) as tracer:
    for step in tracer.iter[:3]:
        if step == 2:
            model.transformer.h[0].output[:] = 0
```

### Mitigation
- For top-level blocks the two coincide. For inner modules, count calls per forward (`print(parent.source)`) to translate steps into occurrences.

---

## Selecting specific steps

`tracer.iter` accepts:
- a slice — `tracer.iter[:3]` (steps 0–2), `tracer.iter[2:5]` (2–4);
- an int — `tracer.iter[2]` (just step 2);
- a list — `tracer.iter[[0, 2, 4]]` (those steps only).

```python
with model.generate("Hello", max_new_tokens=6) as tracer:
    for step in tracer.iter[[0, 2, 4]]:
        model.transformer.h[0].output[:] = 0
```

---

## The deprecated `with tracer.iter[...]:` form

### Symptom
```
DeprecationWarning: `with tracer.iter[...]:` is deprecated; use `for step in tracer.iter[...]:` instead.
```

### Cause / fix
The `with`-block form re-runs the captured block once per step; the `for` form is a plain loop over the body. They behave the same (including the open-ended unwind), but prefer the `for` form.

```python
# deprecated
with model.generate("Hello", max_new_tokens=2) as tracer:
    with tracer.iter[:2]:
        ...
# preferred
with model.generate("Hello", max_new_tokens=2) as tracer:
    for _ in tracer.iter[:2]:
        ...
```

---

## Related
- [docs/usage/iter-all-next.md](../usage/iter-all-next.md) — full `tracer.iter[...]` / `.all()` reference.
- [docs/usage/generate.md](../usage/generate.md) — multi-token generation.
- [docs/gotchas/order-and-deadlocks.md](order-and-deadlocks.md) — module access order rules within a step.
