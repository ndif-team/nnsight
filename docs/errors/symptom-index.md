---
title: Symptom Index
one_liner: Start from what you observed — including the failures that raise nothing at all — and get to the page that explains it.
tags: [error, index, silent-failure]
related: [docs/errors/index.md, docs/usage/save.md, docs/usage/iteration.md]
sources: [src/nnsight/tracing/tracer.py, src/nnsight/intervention/interleaver.py, src/nnsight/intervention/envoy.py]
---

# Symptom Index

Up: [docs/index.md](../index.md) — all sections.

Start here when something broke and nothing raised. For an exception or warning,
look its message up in [index.md](index.md).

## Nothing raised — the result is wrong or missing

| What you saw | What it is | Where |
|---|---|---|
| The intervention changed nothing | The write rebound a local instead of going through the property | [Writing to a name instead of the model](#writing-to-a-name-instead-of-the-model) |
| The whole block did nothing, no output at all | `model.trace(x)` / `model.scan(x)` called without `with` | [A run method called without `with`](#a-run-method-called-without-with) |
| Only one forward ran when you asked for several | A generation kwarg passed to `trace()` | [Generation kwargs on trace()](#generation-kwargs-on-trace) |
| A name after the block is undefined (`UnboundLocalError`, `NameError`) | Not saved, or the block unwound before reaching it | [docs/usage/save.md](../usage/save.md), [module-never-fired.md](module-never-fired.md) |
| A saved list is empty, but only when running remotely | The elements were saved, not the container | [docs/usage/save.md](../usage/save.md) |
| `.grad` is `None` | Read outside a `with metric.backward():` block | [.grad outside a backward block](#grad-outside-a-backward-block) |
| Per-step writes in a loop land one step late, or a per-step list is one entry short with shapes one step ahead and a "was never reached" warning | A `tracer.iter` body that reads a later location before an earlier one (or a `.source` op first touched inside the loop) | [docs/usage/iteration.md](../usage/iteration.md#pitfalls) |
| A per-step list is shorter than the loop bound, and code after the loop never ran | The loop asked for steps the run did not make (EOS, stop string) | [docs/usage/iteration.md](../usage/iteration.md#the-one-rule) |
| A cross-invoke patch raises nothing and changes nothing | A name from the enclosing scope shadows the one the producer invoke binds | [docs/usage/invoke-and-batching.md](../usage/invoke-and-batching.md#writing-a-value-use-a-barrier) |
| An in-place edit does nothing (or a backward then fails on an in-place version check) | The write came after a later module was already read | [docs/usage/backward-and-grad.md](../usage/backward-and-grad.md#in-place-writes-after-the-forward-has-moved-on) |
| Every later trace is changed, not just the one that edited | A write to a `.weight` is permanent | [docs/usage/access-and-modify.md](../usage/access-and-modify.md#weight-edits-are-permanent) |
| Shapes are plausible but the numbers are for one batch row | `.output[0]` on a module whose output is a plain tensor | [docs/usage/access-and-modify.md](../usage/access-and-modify.md#tuple-outputs) |
| A name you never saved comes back bound anyway | `save` marks by object identity | [save-outside-trace.md](save-outside-trace.md) |

## Writing to a name instead of the model

Reading a location gives you the real tensor. Rebinding the name that holds it
changes the name, not the run:

```python
# no effect — `acts` is rebound; the model keeps the value it already produced
with model.trace(prompt):
    acts = model.transformer.h[5].output
    acts = acts * 0
    unchanged = model.output.logits[0, -1].argmax().save()
```

Write in place through the tensor, or assign to the property:

```python
with model.trace(prompt):
    model.transformer.h[5].output[:] = 0                                 # in place
    # or: model.transformer.h[5].output = model.transformer.h[5].output * 0   # swap
    changed = model.output.logits[0, -1].argmax().save()
```

The tell is a metric that does not move. Before trusting any intervention
result, check that an absurd version of it (zero the whole residual stream)
changes the output — see [docs/usage/access-and-modify.md](../usage/access-and-modify.md).

## A run method called without `with`

`model.trace(x)` builds a tracer; the `with` statement is what captures the body
and runs the model. Called on its own it returns an `InterleavingTracer` and runs
nothing:

```python
result = model.trace(prompt)      # an InterleavingTracer — no forward pass happened
```

`model.scan(prompt)` behaves the same way. `model.generate(...)` and
`model.pipe(...)` do run and return their result. If you want a plain forward with
no intervention, pass `trace=False`.

## Generation kwargs on `trace()`

`trace` runs exactly one forward. Kwargs it does not consume are forwarded to
that forward call, which ignores generation settings:

```python
with model.trace(prompt, max_new_tokens=5) as tracer:
    hidden = model.transformer.h[0].output.save()
print(hidden.shape)     # torch.Size([1, 7, 768]) — one pass over the prompt
```

Use `model.generate(prompt, max_new_tokens=5)` for multiple steps. See
[docs/usage/generate.md](../usage/generate.md).

## `.grad` outside a backward block

`.grad` on a captured activation is the plain torch attribute until a backward
pass is running, and torch returns `None` for a non-leaf tensor:

```python
with model.trace(prompt):
    hidden = model.transformer.h[-1].output
    grad = hidden.grad.save()       # None, plus torch's own non-leaf warning
```

Read it inside the backward block that produces it:

```python
with model.trace(prompt):
    hidden = model.transformer.h[-1].output
    metric = model.output.logits.sum()
    with metric.backward():
        grad = hidden.grad.clone().save()
```

See [docs/usage/backward-and-grad.md](../usage/backward-and-grad.md).

## Related

- [index.md](index.md) — exceptions and warnings, keyed by message.
- [debug-mode.md](debug-mode.md) — when the traceback itself is the problem.
