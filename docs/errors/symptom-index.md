---
title: Symptom Index
one_liner: Start from what you observed — including the failures that raise nothing at all — and get to the page that explains it.
tags: [error, index, silent-failure]
related: [docs/errors/index.md, docs/gotchas/index.md, docs/usage/save.md, docs/usage/iter-all-next.md]
sources: [src/nnsight/tracing/tracer.py, src/nnsight/intervention/interleaver.py, src/nnsight/intervention/envoy.py]
---

# Symptom Index

[index.md](index.md) is keyed by exception class, which only helps once you have
an exception. This page is keyed by what you *saw*. The first table is the one
worth reading: those failures raise nothing, so nothing routes you to a page.

## Nothing raised — the result is wrong or missing

| What you saw | What it is | Where |
|---|---|---|
| The intervention changed nothing | The write rebound a local instead of going through the property | [Writing to a name instead of the model](#writing-to-a-name-instead-of-the-model) |
| The whole block did nothing, no output at all | `model.trace(x)` / `model.scan(x)` called without `with` | [A run method called without `with`](#a-run-method-called-without-with) |
| Only one forward ran when you asked for several | A generation kwarg passed to `trace()` | [Generation kwargs on trace()](#generation-kwargs-on-trace) |
| A name after the block is undefined (`UnboundLocalError`, `NameError`) | Not saved, or the block unwound before reaching it | [docs/usage/save.md](../usage/save.md), [value-was-not-provided.md](value-was-not-provided.md) |
| A saved list is empty, but only when running remotely | The elements were saved, not the container | [docs/usage/save.md](../usage/save.md) |
| `.grad` is `None` | Read outside a `with metric.backward():` block | [.grad outside a backward block](#grad-outside-a-backward-block) |
| Per-step writes in a loop land one step late | A `tracer.iter` body that reads a later location before an earlier one — silent unless a shifted request runs off the end of the run | [out-of-order-error.md](out-of-order-error.md) |
| Shapes are plausible but the numbers are for one batch row | `.output[0]` on a module whose output is a plain tensor | [docs/gotchas/types-and-values.md](../gotchas/types-and-values.md) |
| A name you never saved comes back bound anyway | `save` marks by object identity | [save-outside-trace.md](save-outside-trace.md) |

## Something raised

| What you saw | Where |
|---|---|
| `OutOfOrderError: '<location>.i0' was requested but the model already ran past it` | [out-of-order-error.md](out-of-order-error.md) |
| `UserWarning: '<location>.iN' was never reached: the loop asked for a step the run did not make …` | [value-was-not-provided.md](value-was-not-provided.md) |
| `ValueError: Cannot access '<location>' outside of interleaving` | [cannot-access-outside-interleaving.md](cannot-access-outside-interleaving.md) |
| `ValueError: trace() needs an input, or at least one 'with tracer.invoke(...)' block` | [cannot-access-outside-interleaving.md](cannot-access-outside-interleaving.md) |
| `ValueError: save() was called outside a trace. …` | [save-outside-trace.md](save-outside-trace.md) |
| `ValueError: Cannot invoke while the model is already running.` | [invoke-during-execution.md](invoke-during-execution.md) |
| ``ValueError: A traced `with` block cannot start with `try:` …`` | [A trace body that starts with try:](#a-trace-body-that-starts-with-try) |
| ``ValueError: The body of a traced `with` must start on its own line …`` | [A body on the with line](#a-body-on-the-with-line) |
| `ValueError: A barrier was never reached by every block it waits for …` | [docs/usage/barrier.md](../usage/barrier.md) |
| `WithBlockNotFoundError` with no message | [with-block-not-found.md](with-block-not-found.md) |
| `SyntaxError: 'return' outside function` | [return inside a trace body](#return-inside-a-trace-body) |
| `NotImplementedError: <Class> does not support batching multiple invokes` | [batching-not-implemented.md](batching-not-implemented.md) |
| `AttributeError: module 'nnsight' has no attribute 'list' / 'apply' / 'session' / …` | [docs/reference/version-history.md](../reference/version-history.md) |
| `AttributeError: 'Tensor' object has no attribute 'value'` | [docs/reference/version-history.md](../reference/version-history.md) |

---

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

Write through the property, or in place through the tensor it returns:

```python
with model.trace(prompt):
    model.transformer.h[5].output[:] = 0            # in place, through the live tensor
    changed = model.output.logits[0, -1].argmax().save()
```

```python
with model.trace(prompt):
    acts = model.transformer.h[5].output
    model.transformer.h[5].output = acts * 0        # a swap: assign to the property
    also_changed = model.output.logits[0, -1].argmax().save()
```

The tell is a metric that does not move. Before trusting any intervention
result, check that an absurd version of it (zero the whole residual stream)
changes the output — see [docs/gotchas/modification.md](../gotchas/modification.md).

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

## A trace body that starts with `try:`

```
ValueError: A traced `with` block cannot start with `try:`; nnsight intercepts the
body at its first line, and a `try` there is the one statement Python gives it no
way back out of. Put any statement above the `try`, or move the `try` outside the
block.
```

nnsight runs the block's body itself, so it has to stop the interpreter from
running it inline first. It does that by raising at the body's first line
(`skip_context`, `src/nnsight/tracing/tracer.py`) and catching that raise in the
`with`. CPython puts the `try` keyword's line on an instruction that no
exception-table entry covers, so a raise delivered there unwinds the frame
outright instead of reaching the `with`. The shape is refused rather than
silently losing the whole block.

Either statement below fixes it:

```python
with model.trace(prompt):
    hidden = model.transformer.h[0].output      # any statement above the try
    try:
        head = hidden[0, -1]
    except IndexError:
        head = None
    kept = nnsight.save(head)
```

```python
try:                                            # or wrap the whole with
    with model.trace(prompt):
        hidden = model.transformer.h[0].output.save()
except Exception as error:
    print(error)
```

Only the *first* statement is the cue — a `try` anywhere else in the body is
ordinary Python and catches what it is written to catch.

## A body on the `with` line

```
ValueError: The body of a traced `with` must start on its own line; nnsight runs
the body itself, and can only intercept it at the start of a line.
```

Same mechanism, same fix: give the body its own indented line.

```python
with model.trace(prompt): out = model.output.logits.save()      # refused

with model.trace(prompt):
    out = model.output.logits.save()                            # fine
```

## `return` inside a trace body

The body is compiled on its own, outside any function, so a `return` in it is a
syntax error at capture time:

```
SyntaxError: 'return' outside function
```

Return the saved value *after* the block:

```python
def hidden_state(prompt):
    with model.trace(prompt):
        hidden = model.transformer.h[0].output.save()
    return hidden
```

## Related

- [index.md](index.md) — the same errors keyed by exception class.
- [docs/gotchas/index.md](../gotchas/index.md) — the traps that are not errors.
- [debug-mode.md](debug-mode.md) — when the traceback itself is the problem.
