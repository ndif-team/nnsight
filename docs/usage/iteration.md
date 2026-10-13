---
title: Iteration — iter / all
one_liner: Target intervention code at specific generation steps with `for step in tracer.iter[...]` and `tracer.all()`.
tags: [usage, generation, iteration]
related: [docs/usage/generate.md, docs/errors/out-of-order-error.md, docs/errors/module-never-fired.md, docs/usage/source.md, docs/usage/invoke-and-batching.md]
sources: [src/nnsight/intervention/iterator.py, src/nnsight/intervention/tracer.py, src/nnsight/intervention/interleaver.py]
---

# Iteration — `tracer.iter` / `tracer.all()`

In `model.generate(...)` the model runs forward once per generated token, so each
module is reached once per step. `for step in tracer.iter[...]:` binds the loop
body to the selected steps: every `.output` / `.input` read inside binds to the
current step, and `step` is the real integer index. `tracer.all()` is
`tracer.iter[:]`. A single forward (`model.trace(x)`) has only step 0.

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

## Selecting steps

The selector is a slice, an int, or a list; only the loop header changes:

```python
with model.generate("Hello", max_new_tokens=5, do_sample=False) as tracer:
    out = nnsight.save([])
    for step in tracer.iter[1:3]:      # steps 1 and 2
    # for step in tracer.iter[0]:          only the prefill step
    # for step in tracer.iter[[0, 2, 4]]:  those steps only
        out.append(model.lm_head.output)
```

`step` is a plain int, so `if step == 2:` inside the body targets one step of a
wider loop. Negative steps raise `ValueError` (`tracer.iter step cannot be
negative: -1`); there is no "last step" shorthand.

`with tracer.iter[...]:` (the block form) is deprecated
(`NNsightDeprecationWarning`); unlike the loop it silently truncates an over-run
and lets the code after it run. Use `for step in tracer.iter[...]:`.
`model.iter` / `model.all()` are deprecated too.

## The one rule

**A loop must not ask for a step the run does not make.** A bound the run meets
is fine and the code after the loop runs. A loop that outruns the run is cut
short there with a warning — what it saved is kept, and the statements after it
do not run, so the result looks complete while being shorter than the bound.
Check the `len()` of what you collected, or hold the run to the count:

```python
with model.generate("Hello", max_new_tokens=3, do_sample=False) as tracer:
    zeroed = nnsight.save([])
    for step in tracer.iter[:10]:                # 10 steps of a 3-step run
        model.transformer.h[0].output[:] = 0
        zeroed.append(step)
# len(zeroed) == 3, and this warning:
```

```
UserWarning: 'model.transformer.h.0.output.i3' was never reached: the loop asked
for a step the run did not make, so it was cut short — values saved inside the
loop are kept, and the statements after it did not run. ...
```

The occurrence it names (`.i3`, the fourth ask) is the first step the run did not
make, which tells you how far the loop got.

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

`min_new_tokens` holds off the end-of-sequence token and nothing else. A
`stop_strings=` criterion still ends the run where it matches, and so does a
`tracer.stop()` in any block of the same trace — pair either with an open loop.

The rule is about the run, not about generation: a `tracer.iter[:2]` inside a
single-forward `model.trace()` outruns it the same way, and every runtime that
serves a trace applies the same policy.

### When the step count is unknown

`tracer.iter[:]` / `tracer.all()` end *by* asking for a step the run does not
make — the same warning and the same unwind, which still discards the
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

Values saved inside the loop survive either way. `tracer.result` cannot move below
the `with` block instead: it is served during the run, so reading it afterwards
raises ``ValueError: Cannot access `result` outside of interleaving``.

## Pitfalls

### An out-of-order body shifts by a step

Inside the loop, the step is pinned for the body's *first* request and then
relaxes, so a later request resolves to the next occurrence the model has not
handled yet. A request for a location the model already ran past this step
therefore binds to the **next** step instead of raising. Reading layer 6 and then
writing layer 2 parks the write on the following step; reading layer 6 and then
layer 0 returns step *k+1*'s value under step *k*'s label:

```python
# baseline norms of h[6] over 4 steps: [92.638, 100.614, 81.072, 86.349]
with model.generate(prompt, max_new_tokens=4, min_new_tokens=4) as tracer:
    got = nnsight.save([])
    for step in tracer.iter[1:3]:
        got.append(model.transformer.h[6].output[:, -1, :].norm())
        model.transformer.h[2].output[:] = 0   # below h[6]: parks to the next step
# got == [100.614, 51.358] — step 1 is the unmodified baseline
```

Swap the two lines — write `h[2]` before reading `h[6]` — and both steps are
modified (`[54.054, 53.875]`). Whether anything reports the shift depends on whether
the last shifted request still has a step to land on:

| Loop over an out-of-order body | Result |
|---|---|
| includes step 0 (`iter[:N]`, `iter[0:N]`) | raises `OutOfOrderError` (`'…i0' was requested but the model already ran past it`) |
| bounded, last selected step **is** the run's last (`iter[1:4]` over 4 steps) | warns, naming an occurrence past the loop's selection; statements after the loop do not run |
| bounded, stops **short** of the run's last step (`iter[1:3]` over 4 steps) | **silent** — values one step late, trailing code runs |
| open (`iter[1:]`, `tracer.all()`) | warns — as every open loop does when it ends — with writes one step late and the last dropped |

So the "never reached" warning is ambiguous: it reads as a loop bound the run did
not meet, pointing at `min_new_tokens=`, when the real cause is a backwards access.
A step list in descending order (`tracer.iter[[3, 1]]`) is cut short the same way.
If the run made the steps you asked for, look for a backwards access in the body
before touching the bound. The tell for the silent row is a first selected step that
matches an unmodified baseline — compare per step against a no-write run.

### A `.source` op first touched inside the loop shifts even at step 0

A module's `.source` is set up the first time it is touched. When that first touch
is inside the loop, after the op already ran on the current step, the read binds to
the op's next call — the next step — even in a loop that starts at 0:

```python
attn = model.transformer.h[3].attn            # attn_implementation="eager"
attn.source   # touch before the trace, so a wrong-order read raises instead of shifting
with model.generate(prompt, max_new_tokens=4) as tracer:
    pat = nnsight.save([])
    for step in tracer.iter[:4]:
        pat.append(attn.source.attention_interface_1.output[1])   # op first...
        out = attn.output[0]                                      # ...then its module
# len(pat) == 4; key lengths [7, 8, 9, 10] are steps 0-3
```

With the module read first and no prior touch, `pat` holds steps 1–3 (one entry
short) and the run ends with the "never reached" warning. A first touch inside
`with tracer.iter[1]:` after a `with tracer.iter[0]:` block likewise returns step 2.
Touch `.source` before the trace, and assert parallel lists have equal lengths.

### `tracer.iter[N]` counts occurrences

`tracer.iter[N]` binds to occurrence `N` of each location. A top-level block fires
once per step, so occurrence `N` is step `N`. A module or `.source` op called `k`
times within one forward (a recurrent inner module, an MoE expert loop) fires `k`
occurrences per step, so occurrence `N` lands inside a step. Count calls per
forward with `print(parent.source)` to translate steps into occurrences.

### Nested loops

A read placed after a nested inner loop binds to the step the *inner* loop left the
model on, not the outer step it is written under. Nest loops only when the outer
body reads nothing after the inner one.

## Gotchas

- **An open loop whose body reads no module never returns.** Nothing parks the
  worker, so it spins with no warning and no timeout. Bound the loop, or read
  something inside it.
- **Regular-module access after the loop is out of order.** Those forward passes
  are done, so a module's `.output`/`.input` after the loop raises
  `OutOfOrderError`.
