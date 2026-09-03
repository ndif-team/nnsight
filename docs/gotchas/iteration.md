---
title: Iteration Pitfalls (iter, all)
one_liner: Multi-step generation footguns — a loop that outruns the run, an open loop that never returns, and writes that land one step late.
tags: [gotcha, generate, iter, all]
related: [docs/usage/iter-all-next.md, docs/usage/generate.md]
sources: [src/nnsight/intervention/iterator.py, src/nnsight/intervention/interleaver.py]
---

# Iteration Pitfalls

## TL;DR

- **A loop must not ask for a step the run does not make.** That is the whole
  rule; bounded versus open is not the axis.
- A bound the run meets is fine, and the code after the loop runs.
  `for step in tracer.iter[:3]` against a three-step generation is correct.
- A bound the run does not meet raises `OutOfOrderError`, naming the iteration
  asked for and the count reached.
- `max_new_tokens` is an upper bound — EOS or a stop string ends generation
  sooner. `min_new_tokens=N` holds a generation to N steps against EOS.
- An open `tracer.iter[:]` / `tracer.all()` ends *by* outrunning the run, so it
  warns instead. Values saved inside it are kept; the statements after it do not
  run.
- The loop form is `for step in tracer.iter[...]:`. A `with tracer.iter[...]:`
  block is deprecated and warns.
- `tracer.iter[N]` targets the `(N+1)`-th **occurrence** of a location. For a
  module reached once per step that is step `N`; for one called several times
  per forward it is the call count.

---

## A loop that outruns the run

### Symptom

```
OutOfOrderError: 'model.transformer.h.6.output.i3' was never reached: the loop
asked for iteration 3 of 'model.transformer.h.6.output' and the run reached it 3
times, so the loop was cut short and nothing after it ran. ...
```

Or, from an open loop, the same situation as a warning and a name that is never
bound afterwards:

```
UserWarning: 'model.transformer.h.6.output.i3' was never reached: an open
`tracer.iter[:]` / `tracer.all()` loop ends by asking for a step the run does
not make. Values saved inside the loop are kept; the statements after it did
not run.
```

### Cause

The loop hands out step indices; a read or write in the body parks the worker
until the model reaches that occurrence. When the loop hands out a step the
model never runs, the worker stays parked there forever, and
`check_dangling_mediators` unwinds it at the loop — which discards every
statement the block has after the loop.

Whether that is an error or a note depends on who chose the end. A bound you
wrote (`iter[:10]`, `iter[2]`, `iter[[0, 2, 4]]`) is a claim about the run, so a
run that cannot supply it raises. An open loop has no end of its own —
outrunning the model is how it finishes — so it warns.

### Wrong code

```python
with model.generate("The Eiffel Tower is in", max_new_tokens=3) as tracer:
    steps = nnsight.save([])
    for step in tracer.iter[:10]:        # 10 steps asked of a 3-step run
        steps.append(model.transformer.h[-1].output[:, -1, :])
    ids = tracer.result.save()
# OutOfOrderError
```

### Right code (bound the loop to what the run makes)

```python
with model.generate("The Eiffel Tower is in", max_new_tokens=3) as tracer:
    steps = nnsight.save([])
    for step in tracer.iter[:3]:
        steps.append(model.transformer.h[-1].output[:, -1, :])
    ids = tracer.result.save()           # runs
# len(steps) == 3, ids.shape == (1, 10)
```

A bound *below* the step count is equally fine — the loop ends on its own before
the run does.

### Right code (an open loop plus a separate empty invoke)

When the step count is genuinely unknown, loop openly and put what has to happen
afterwards in its own invoke. That invoke is a second worker on the same batch,
so the loop's unwind does not reach it:

```python
with model.generate(max_new_tokens=3) as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        steps = nnsight.save([])
        for step in tracer.iter[:]:
            steps.append(model.transformer.h[-1].output[:, -1, :])
    with tracer.invoke():
        ids = tracer.result.save()       # its own worker; runs
```

Plain Python that only needs values already saved can also move below the `with`
block, where it is outside the trace entirely. `tracer.result` cannot: reading it
after the block raises ``ValueError: Cannot access `result` outside of
interleaving``, because it is served during the run like any other location.

### `max_new_tokens` is an upper bound

A generation that emits EOS stops there, so a loop bound to `max_new_tokens`
outruns it:

```python
# gpt2 continues " Paris, and the Eiff"; make the third token an EOS
with model.generate(prompt, max_new_tokens=6, eos_token_id=290) as tracer:
    for step in tracer.iter[:6]:
        ...
# OutOfOrderError — the run reached 3 occurrences, the loop asked for 6
```

`min_new_tokens=N` suppresses EOS until N tokens have been generated, which is
what makes a bound of N safe:

```python
with model.generate(
    prompt, max_new_tokens=6, min_new_tokens=6, eos_token_id=290,
) as tracer:
    per_step = nnsight.save([])
    for step in tracer.iter[:6]:
        per_step.append(model.transformer.h[6].output[:, -1, :].norm())
    ids = tracer.result.save()           # runs; len(per_step) == 6
```

`min_new_tokens` holds off EOS only. A `stop_strings=` criterion still ends the
run wherever it matches, so a loop over a run with stop strings should be open.

### Mitigation

- Read the count in the message: it names the occurrence asked for and the number
  the run reached, so the fix is usually to change one number.
- `tracer.all()` is `tracer.iter[:]` — same shape, warning instead of an error.

---

## An open loop whose body reads nothing never returns

### Symptom

The script sits at 100% CPU with no output, no warning and no timeout.

### Cause

An open loop ends when the model stops supplying steps, and the worker learns
that by parking on a request the model never serves. A body that touches no
module never parks, so nothing ever ends the loop and the index generator spins
forever.

### Wrong code

```python
with model.generate("Hello", max_new_tokens=2) as tracer:
    n = nnsight.save([0])
    for step in tracer.all():
        n[0] = step                       # no .input / .output anywhere
```

### Right code

Bound the loop, or read something in the body:

```python
with model.generate("Hello", max_new_tokens=2) as tracer:
    n = nnsight.save([0])
    for step in tracer.iter[:2]:
        n[0] = step
```

---

## An open loop past step 0 lands writes one step late

### Symptom

Interventions inside the loop appear to work — no error inside the body — but the
first step comes back unmodified and the effect is one step behind throughout.

### Cause

Inside the loop, `iteration` is pinned to the step for the body's *first*
request and then relaxes, so a later request resolves to the next occurrence the
model has not handled yet. For a module the model already ran this step, that is
next step's occurrence. Reading layer 6 and then writing layer 2 therefore parks
the write on the following step.

Most shapes of this surface on their own: at step 0 the request raises
`OutOfOrderError` immediately, and a loop that runs to the end of the generation
raises when the last parked write outruns the run. It stays silent in the one
case where every skewed write still has a step to land on — an open
`tracer.iter[a:]` with `a > 0`, or a bound that stops short of the last step.

### Wrong code

```python
# baseline norms of h[6] over 4 steps: [92.638, 100.614, 81.072, 86.349]
with model.generate(prompt, max_new_tokens=4) as tracer:
    got = nnsight.save([])
    for step in tracer.iter[1:3]:
        got.append(model.transformer.h[6].output[:, -1, :].norm())
        model.transformer.h[2].output[:] = 0   # below h[6]: parks to the next step
# got == [100.614, 51.358] — step 1 is the unmodified baseline
```

### Right code

Write before you read anything further down the stack, so every request in the
body is in forward-pass order:

```python
with model.generate(prompt, max_new_tokens=4) as tracer:
    got = nnsight.save([])
    for step in tracer.iter[1:3]:
        model.transformer.h[2].output[:] = 0
        got.append(model.transformer.h[6].output[:, -1, :].norm())
# got == [54.054, 53.875] — both steps modified
```

### Mitigation

- Order the body the way the forward runs: layer 2 before layer 6, a submodule
  before the block containing it. See
  [order-and-deadlocks.md](order-and-deadlocks.md).
- Compare step 0's value against an unmodified run. A first step that matches the
  baseline is the tell.

---

## `tracer.iter[N]` counts occurrences, not always generation steps

### Symptom

You expect `tracer.iter[2]` to mean "the 3rd generation step" for every module,
but for a module called several times per forward it targets a different call.

### Cause

Each visit to a location is tagged with its occurrence index, and `tracer.iter[N]`
binds the request to occurrence `N` (`iterator.py`, `Iterations`). A top-level
transformer block fires once per generation step, so occurrence `N` is step `N`.
A module called `k` times within one forward — a recurrent inner module, an
expert loop — fires `k` occurrences per step, so occurrence `N` lands somewhere
inside a step.

### Right code

```python
# a top-level block, once per step — iter[2] is generation step 2
with model.generate("Hello", max_new_tokens=3) as tracer:
    for step in tracer.iter[:3]:
        if step == 2:
            model.transformer.h[0].output[:] = 0
```

### Mitigation

For inner modules, count calls per forward (`print(parent.source)`) to translate
steps into occurrences.

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

Negative indices raise `ValueError: tracer.iter step cannot be negative: -1` —
there is no "last step" shorthand.

---

## The deprecated `with tracer.iter[...]:` form

### Symptom

```
NNsightDeprecationWarning: The `with tracer.iter[...]:` / `with tracer.all():`
block form is deprecated; use `for step in tracer.iter[...]:` instead.
```

### Cause / fix

The `with`-block form re-runs the captured block once per step; the `for` form is
a plain loop over the body. They differ in one visible way: because the block
form owns its loop, it catches its own over-run, truncates to the steps that ran,
and lets the code after the block run without a word. The `for` form raises
there. Prefer the `for` form and a bound the run meets.

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

`model.iter` and `model.all()` are deprecated the same way — use `tracer.iter` /
`tracer.all()`.

---

## Related
- [docs/usage/iter-all-next.md](../usage/iter-all-next.md) — full `tracer.iter[...]` / `.all()` reference.
- [docs/usage/generate.md](../usage/generate.md) — multi-token generation.
- [docs/gotchas/order-and-deadlocks.md](order-and-deadlocks.md) — module access order rules within a step.
