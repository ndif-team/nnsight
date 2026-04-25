---
title: Iteration Pitfalls (iter, all, next)
one_liner: Multi-step generation footguns — unbounded iter[:] swallowing trailing code, .all() = iter[:], .next() vs iter, and per-call counters.
tags: [gotcha, generate, iter, all, next]
related: [docs/usage/iter.md, docs/usage/all-and-next.md, docs/usage/generate.md]
sources: [src/nnsight/intervention/tracing/iterator.py:209, src/nnsight/intervention/tracing/iterator.py:96, src/nnsight/intervention/tracing/tracer.py:457]
---

# Iteration Pitfalls

## TL;DR
- `for step in tracer.iter[:]` and `tracer.all()` are *unbounded*. Code *after* the for-loop body inside the same trace never executes — the iterator is still waiting for the next step when the model finishes.
- Fix with bounded `iter[:N]`, or move the post-iter logic into a separate empty `tracer.invoke()`.
- `.next()` is the per-step manual alternative — use it when you want each step to be its own block of code.
- The iteration tracker counts module *call counts* per provider path, not generation steps. For models with recurrent inner modules (e.g. Mamba) that fire multiple times per generation step, the tracker advances faster than the step counter.

---

## Unbounded `tracer.iter[:]` swallows trailing code

### Symptom
You write a for-loop over `tracer.iter[:]` and add code after it inside the same trace. The post-loop code never runs — variables defined after the loop come back as `NameError`. nnsight prints a warning like `Execution complete but '...' was not provided. If this was in an Iterator at iteration N this iteration did not happen.`

### Cause
`tracer.iter[:]` is an open-ended slice with no `stop`. The `IteratorTracer.__iter__` generator (`src/nnsight/intervention/tracing/iterator.py:209`) keeps incrementing `i` and yielding indefinitely until `stop` is non-None, but for an unbounded `iter[:]` the only way `stop` becomes set is if `mediator.all_stop` or `interleaver.default_all` was set elsewhere — which doesn't happen for plain `iter[:]`.

When the model's `max_new_tokens` is reached, the model simply returns. The worker thread is mid-`yield` waiting for the next step. The trace exits. `check_dangling_mediators` (`src/nnsight/intervention/interleaver.py:652`) emits the warning. Anything in your code after the for-loop never executes because control never came back from the generator.

`tracer.all()` is exactly `tracer.iter[:]` (`src/nnsight/intervention/tracing/tracer.py:457`) — same pitfall.

### Wrong code
```python
with model.generate("Hello", max_new_tokens=3) as tracer:
    hidden_steps = list().save()
    for step in tracer.iter[:]:
        hidden_steps.append(model.transformer.h[-1].output[0])

    # NEVER RUNS
    final_logits = model.lm_head.output.save()
print(final_logits)   # NameError
```

### Right code (option 1: bounded iter)
```python
with model.generate("Hello", max_new_tokens=3) as tracer:
    hidden_steps = list().save()
    for step in tracer.iter[:3]:    # bounded — terminates cleanly
        hidden_steps.append(model.transformer.h[-1].output[0])

    final_logits = model.lm_head.output.save()
```

### Right code (option 2: separate empty invoke)
When using multiple invokes, do not pass input to `generate()` — pass it to the first invoke. The empty invoke runs as its own thread on the same batch:

```python
with model.generate(max_new_tokens=3) as tracer:
    with tracer.invoke("Hello"):
        hidden_steps = list().save()
        for step in tracer.iter[:]:
            hidden_steps.append(model.transformer.h[-1].output[0])

    with tracer.invoke():    # empty invoke — separate thread, runs after
        final_logits = model.lm_head.output.save()
```

### Mitigation / how to spot it early
- See `Execution complete but '...' was not provided. If this was in an Iterator at iteration N this iteration did not happen` in the warnings? You hit this.
- If a variable defined *after* a `for step in tracer.iter[:]` loop is missing outside the trace, it's this gotcha.
- `tracer.all()` is the same thing under a different name — same fix applies.

---

## `tracer.all()` is `iter[:]` in disguise

### Symptom
Same as the above — code after `tracer.all()` doesn't run.

### Cause
`InterleavingTracer.all` literally returns `self.iter[:]` (`src/nnsight/intervention/tracing/tracer.py:457`). It's a thin alias for the unbounded slice.

### Wrong code
```python
with model.generate("Hello", max_new_tokens=3) as tracer:
    for step in tracer.all():
        model.transformer.h[0].output[0][:] = 0
    final = model.lm_head.output.save()    # never runs
```

### Right code
```python
with model.generate("Hello", max_new_tokens=3) as tracer:
    for step in tracer.iter[:3]:           # bounded
        model.transformer.h[0].output[0][:] = 0
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
            model.transformer.h[0].output[0][:] = 0
```

### Right code (next)
```python
with model.generate("Hello", max_new_tokens=3) as tracer:
    hs1 = model.transformer.h[-1].output[0].save()
    hs2 = model.transformer.h[-1].next().output[0].save()
    hs3 = model.transformer.h[-1].next().output[0].save()
```

### Mitigation / how to spot it early
- If your steps share a body, use `iter[...]`.
- If your steps are distinct snippets, use `.next()` and avoid the unbounded-iter footgun entirely.

---

## Iteration tracker counts module *calls*, not generation steps

### Symptom
For recurrent inner modules (e.g. Mamba's per-token state update inside a single forward pass), `model.layer.next().output` advances by 1 per call rather than 1 per generation step. Or: an interaction between `.iter[N]` and `.source` operations fires on the wrong "step".

### Cause
`mediator.iteration_tracker` is a `defaultdict(int)` keyed by *provider path*. The persistent iter hooks installed by `register_iter_hooks` (`src/nnsight/intervention/tracing/iterator.py:96`) bump the tracker for `<path>.input` and `<path>.output` once per forward dispatch through that module. If a module fires multiple times within a single generation step (because it's recurrent or called in a loop inside another module), its tracker advances multiple times.

This is by design — one-shot hooks need to know "how many times has this provider fired". But it means `iteration` is not a generation-step index — it's a per-provider-path call counter.

### Wrong assumption
```python
# WRONG MENTAL MODEL: "iter[2] means 'second generation step' for every module"
with model.generate("Hello", max_new_tokens=3) as tracer:
    for step in tracer.iter[2]:
        # If recurrent_module is called 4x per generation step, this targets
        # the 2nd CALL of recurrent_module, which is part of generation step 0,
        # not the 2nd generation step.
        v = recurrent_module.output.save()
```

### Mental fix
For each module, the iteration index `i` means "the (i+1)-th time this module's forward fires within the trace", not "generation step i". For top-level transformer blocks called once per generation step, the two coincide. For recurrent inner modules, they don't.

### Mitigation / how to spot it early
- Print `mediator.iteration_tracker` (in a debugger) at a known point — confirms which counter is at what value.
- If a module fires more than once per generation step (count its appearances in the forward pass via `print(model)` and the parent's source), translate generation-step targets into call counts manually.

---

## Related
- [docs/usage/iter.md](../usage/iter.md) — full reference for `tracer.iter[...]`.
- [docs/usage/all-and-next.md](../usage/all-and-next.md) — `.all()` and `.next()` semantics.
- [docs/usage/generate.md](../usage/generate.md) — multi-token generation.
- [docs/gotchas/order-and-deadlocks.md](order-and-deadlocks.md) — module access order rules (still apply within an iter step).
