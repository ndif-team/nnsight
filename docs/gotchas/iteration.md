---
title: Iteration Pitfalls (iter, all, next)
one_liner: Multi-step generation footguns — unbounded iter[:] swallowing trailing code, .all() = iter[:], .next() vs iter, and per-call counters.
tags: [gotcha, generate, iter, all, next]
related: [docs/usage/iter-all-next.md, docs/usage/generate.md]
sources: [src/nnsight/intervention/tracing/iterator.py:209, src/nnsight/intervention/tracing/iterator.py:96, src/nnsight/intervention/tracing/tracer.py:457]
---

# Iteration Pitfalls

## TL;DR

**Default mental model: assume any code after a `tracer.iter[...]` loop in the same trace will NOT run.** Always default to the separate-empty-invoke pattern when you want code to run "after" the iter loop (for example, to access `tracer.result` or any module's `.output` post-iter).

- `for step in tracer.iter[:]` and `tracer.all()` are *unbounded* by default. The for-loop body keeps yielding forever; the worker thread never returns to whatever code follows the loop. **Module access in trailing code** raises `OutOfOrderError` (the model's forward passes are already done). Pure-Python trailing code happens to keep running on the worker thread but you should never rely on this.
- Even **bounded** slices like `tracer.iter[:10]` are NOT a guarantee — if the model only generates 7 tokens (e.g., it hits EOS before `max_new_tokens`), iterations 7–9 never happen and any code waiting for them produces `MissedProviderError` warnings.
- nnsight has a `default_all` mechanism (set on the interleaver by the model) that fills in a default stop for `iter[:]`. `LanguageModel.generate(..., max_new_tokens=N)` sets it to `N`. So `tracer.iter[:]` inside `model.generate(...)` is effectively `tracer.iter[:max_new_tokens]` — but `max_new_tokens` is a CAP, not the actual count, so the same bounded-but-not-guaranteed caveat applies.
- **Always recommend the separate-invoke pattern** to users who want post-iter code: put the iter loop in one `tracer.invoke(prompt)`, put the post-iter access in a separate empty `tracer.invoke()` that runs as its own thread. See "Right code (option 2)" below.
- `.next()` is the per-step manual alternative — use it when you want each step to be its own block of code.
- The iteration tracker is **only maintained inside an iter loop**: persistent hooks are registered on entry, removed on exit. Outside an iter loop, the tracker is whatever it was when the last iter loop finished — it is not auto-reset and there are no hooks bumping it.
- Once an iter loop has advanced the tracker past `N`, asking for `tracer.iter[N]` later in the same mediator hangs forever (you'll get a `MissedProviderError`). Use a fresh invoke for each iter range you want to access.

---

## Unbounded `tracer.iter[:]` and trailing module access

### Symptom
You write a for-loop over `tracer.iter[:]` and add code after it inside the same trace. **Pure-Python trailing code (e.g. mutating a list, printing) actually does run**, but any line that requests another module's `.output` / `.input` raises an `OutOfOrderError` ("Value was missed for ...") because the model's forward passes are already done. nnsight also prints a warning like `Execution complete but '...' was not provided. If this was in an Iterator at iteration N this iteration did not happen.`

### Cause
`tracer.iter[:]` is an open-ended slice with no `stop`. The `IteratorTracer.__iter__` generator (`src/nnsight/intervention/tracing/iterator.py:209`) keeps incrementing `i` and yielding indefinitely until `stop` is non-None.

There are three ways `stop` can become set:
1. The user provides one explicitly: `tracer.iter[:N]`.
2. `mediator.all_stop` was set during the loop body.
3. `interleaver.default_all` is set by the model. **`LanguageModel.generate(..., max_new_tokens=N)` sets `default_all = N`** (`src/nnsight/modeling/language.py:151`), so a plain `tracer.iter[:]` inside `generate(...)` effectively becomes `tracer.iter[:max_new_tokens]`.

For plain `tracer.iter[:]` outside `generate(...)`, none of those fire — the loop yields forever and the worker thread is stuck in the for-loop's `yield` even after the model returns. `check_dangling_mediators` (`src/nnsight/intervention/interleaver.py:652`) emits the "was not provided" warning.

**Even with `default_all` set:** `max_new_tokens` is a *cap*, not the *actual* number of tokens generated. The model can stop earlier (EOS token, stop strings, etc.). So `tracer.iter[:]` inside `generate(max_new_tokens=10)` may iterate fewer than 10 times. Any iteration that didn't actually happen produces a `was not provided` warning at trace exit if the user requested values for it.

`tracer.all()` is exactly `tracer.iter[:]` (`src/nnsight/intervention/tracing/tracer.py:457`) — same pitfall, same `default_all` interaction.

### Wrong code
```python
with model.generate("Hello", max_new_tokens=3) as tracer:
    hidden_steps = list().save()
    for step in tracer.iter[:]:
        hidden_steps.append(model.transformer.h[-1].output)

    # The line below requests a module value AFTER all forward passes are done →
    # OutOfOrderError, plus a "was not provided" warning.
    final_logits = model.lm_head.output.save()
```

### Right code (RECOMMENDED — separate empty invoke)
This is the pattern to recommend by default. Even when bounded iter would "work", the separate-invoke form is more robust against early model termination (EOS / stop strings) and makes the intent obvious. The empty invoke runs as its own thread on the same batch, **after** the iter-loop invoke finishes:

```python
with model.generate(max_new_tokens=3) as tracer:
    with tracer.invoke("Hello"):                # iter loop lives here
        hidden_steps = list().save()
        for step in tracer.iter[:]:
            hidden_steps.append(model.transformer.h[-1].output)

    with tracer.invoke():                       # empty invoke — runs after
        final_logits = model.lm_head.output.save()    # safe: own thread, own forward pass
        result = tracer.result.save()                 # also safe here
```

### Right code (option 2: bounded iter — only safe when no post-loop module access)
```python
with model.generate("Hello", max_new_tokens=3) as tracer:
    hidden_steps = list().save()
    for step in tracer.iter[:3]:    # bounded — but iterations may still be skipped if model stops early
        hidden_steps.append(model.transformer.h[-1].output)
    # No module access after the loop — only post-loop pure-Python is safe
```

Don't put module access (e.g., `model.lm_head.output.save()`) after a bounded iter slice unless you can guarantee every iteration fires. The model can stop early (EOS, stop strings) and you'll get `MissedProviderError` warnings for the iterations that didn't happen.

### Mitigation / how to spot it early
- See `Execution complete but '...' was not provided. If this was in an Iterator at iteration N this iteration did not happen` in the warnings? You hit this.
- If a variable defined *after* a `for step in tracer.iter[:]` loop is missing outside the trace, it's this gotcha.
- `tracer.all()` is the same thing under a different name — same fix applies.

---

## `tracer.all()` is `iter[:]` in disguise

### Symptom
Same as the above — module-access code after `tracer.all()` raises `OutOfOrderError`.

### Cause
`InterleavingTracer.all` literally returns `self.iter[:]` (`src/nnsight/intervention/tracing/tracer.py:457`). It's a thin alias for the unbounded slice.

### Wrong code
```python
with model.generate("Hello", max_new_tokens=3) as tracer:
    for step in tracer.all():
        model.transformer.h[0].output[:] = 0
    final = model.lm_head.output.save()    # OutOfOrderError — model already done
```

### Right code
```python
with model.generate("Hello", max_new_tokens=3) as tracer:
    for step in tracer.iter[:3]:           # bounded
        model.transformer.h[0].output[:] = 0
    final = model.lm_head.output.save()
```

Or use the empty-invoke pattern from the previous gotcha.

### Mitigation / how to spot it early
- Treat `tracer.all()` as bounded only when there is no code after it. Otherwise prefer explicit bounds.

---

## `.next()` vs `iter[]` — when to use which

### Symptom
You want to inspect a *specific* generation step but writing `for step in tracer.iter[2]:` feels heavy, or you don't want to set up a loop at all.

### Cause
There are two compatible APIs and they target different ergonomics:

- `tracer.iter[i]` / `tracer.iter[1:3]` / `tracer.iter[[0, 2]]` — declarative. Each yielded step sets `mediator.iteration` so subsequent `.output`/`.input` accesses target the right step.
- `module.next()` / `tracer.next()` — imperative. Bumps `mediator.iteration` by 1 (or N). The next access lands on the new step.

Use `.next()` when each step is its own logical block of code; use `iter[...]` when steps share a body and you want to vary by step index.

### Right code (iter)
```python
with model.generate("Hello", max_new_tokens=3) as tracer:
    for step in tracer.iter[:3]:
        if step == 2:
            model.transformer.h[0].output[:] = 0
```

### Right code (next)
```python
with model.generate("Hello", max_new_tokens=3) as tracer:
    hs1 = model.transformer.h[-1].output.save()
    hs2 = model.transformer.h[-1].next().output.save()
    hs3 = model.transformer.h[-1].next().output.save()
```

### Mitigation / how to spot it early
- If your steps share a body, use `iter[...]`.
- If your steps are distinct snippets, use `.next()` and avoid the unbounded-iter footgun entirely.

---

## The iteration tracker is only live **inside** an iter loop

### Symptom
You assume `model.layer.output` "remembers" how many times a module has fired across the whole trace and that `tracer.iter[N]` indexes into a global counter. Instead, requesting `.iter[N]` for an `N` you've already passed hangs forever, eventually surfacing as a `MissedProviderError` ("Execution complete but ... was not provided").

### Cause
The persistent iter-tracking hooks are **scoped to the iter loop**, not to the trace. `register_iter_hooks` (`src/nnsight/intervention/tracing/iterator.py:95`) registers them on `IteratorTracer.__iter__` entry, and the `finally` block (`iterator.py:286`-`291`) removes them when the loop exits. Outside an iter loop, **no hook is bumping `mediator.iteration_tracker`**.

That has two important consequences:

1. **The tracker is `0` for every provider path before any iter loop has run.** A bare `.next()` or a subsequent access without a wrapping `tracer.iter[...]` does not auto-advance the tracker.
2. **The tracker is not reset between iter loops.** When an iter loop ends, the hooks are removed, but the values in `mediator.iteration_tracker` are left where they were. If you've already iterated past step 10 (so `tracker[<path>] == 11`), and then you ask for `tracer.iter[5]`, the mediator is being asked to wait for "the 5th call" of a module whose tracker is already at 11. That call is in the past — the model never fires it again, the worker waits forever, and `check_dangling_mediators` raises `MissedProviderError` (`src/nnsight/intervention/interleaver.py:652`).

### Symptom — concrete

```python
# WRONG — second iter loop reaches into the past
with model.generate("Hello", max_new_tokens=20) as tracer:
    with tracer.invoke("Hello"):
        for step in tracer.iter[:15]:           # tracker for h[-1].output ends at ~15
            hs = model.transformer.h[-1].output.save()
        for step in tracer.iter[5]:             # asks for the "5th" call — already past
            still_hs = model.transformer.h[-1].output.save()
            # MissedProviderError: Execution complete but `model.transformer.h.-1.output.i5` was not provided.
```

### Right code
Either run the second iter range without going past it first, or restructure into separate empty invokes (each invoke gets a fresh mediator and a fresh tracker):

```python
# FIXED — two invokes, two mediators, two fresh trackers
with model.generate(max_new_tokens=20) as tracer:
    with tracer.invoke("Hello"):
        for step in tracer.iter[:15]:
            hs = model.transformer.h[-1].output.save()
    with tracer.invoke():
        for step in tracer.iter[5]:
            still_hs = model.transformer.h[-1].output.save()
```

### What about recurrent inner modules?

If a module fires **multiple times within a single generation step** (e.g. Mamba's per-token state update, a module called in a loop inside another module's forward), the iter hooks bump it multiple times per step. Within an iter loop, `tracer.iter[N]` for that module targets the `(N+1)`-th *call*, not the `(N+1)`-th *generation step*.

For top-level transformer blocks that run exactly once per generation step, the two coincide. For inner recurrent modules, they don't.

```python
# WRONG MENTAL MODEL: "iter[2] means 'second generation step' for every module"
with model.generate("Hello", max_new_tokens=3) as tracer:
    for step in tracer.iter[2]:
        # If recurrent_module is called 4x per generation step, this targets
        # the 3rd CALL of recurrent_module — somewhere inside generation step 0,
        # not the 3rd generation step.
        v = recurrent_module.output.save()
```

### Mental fix

- The iter tracker is **bound to the lifetime of the iter loop** that installed it.
- Within an iter loop, `iter[N]` for a module means "the `(N+1)`-th time this module's forward has fired since the iter loop started." For top-level blocks, that's "step `N`"; for recurrent inner modules, it isn't.
- If you've already passed step `N` once, you can't ask for it again in the same mediator — it's a missed provider.

### Mitigation / how to spot it early
- If you get `MissedProviderError: Execution complete but ... was not provided` after an iter loop and a `.iter[N]` for a low `N`, you've gone past `N` already. Restructure into separate invokes.
- Print `mediator.iteration_tracker` in a debugger inside the iter loop to confirm which counters are at what value.
- For recurrent inner modules, count how many times the parent's `forward` calls them (via `print(parent.source)` or `inspect.getsource(type(parent).forward)`) to translate generation-step targets into call counts.

---

## Related
- [docs/usage/iter-all-next.md](../usage/iter-all-next.md) — full reference for `tracer.iter[...]`, `.all()`, and `.next()` semantics.
- [docs/usage/generate.md](../usage/generate.md) — multi-token generation.
- [docs/gotchas/order-and-deadlocks.md](order-and-deadlocks.md) — module access order rules (still apply within an iter step).
