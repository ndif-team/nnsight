---
title: Session
one_liner: `model.session()` wraps several traces in one scope so values pass between them without `.save()`.
tags: [usage, session, multi-trace]
related: [docs/usage/trace.md, docs/usage/save.md, docs/usage/invoke-and-batching.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/tracer.py]
---

# Session — share values across traces

A **session** is a scope around several traces. A value read in one trace is
available in a later trace **without** an explicit `.save()`, because the *session*
— not each individual trace — is the boundary back to your code. Only values you
mark with `.save()` survive past the session itself.

```python
import torch
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)
P = "The Eiffel Tower is in the city of"

with model.session():
    with model.trace(P):
        h0 = model.transformer.h[0].output       # captured — no .save() needed
    with model.trace(P):
        diff = (model.transformer.h[0].output - h0).abs().sum().save()

print(diff.item())     # 0.0  (same input -> identical activations)
```

`h0` flows from the first trace into the second on its own; only `diff` is
`.save()`d, so only `diff` comes back out.

## Why a session

Outside a session, each `with model.trace(...)` is its own boundary: values don't
cross from one trace to the next, and everything you want to keep needs `.save()`.
A session removes the per-trace boundary so you can:

- reuse a value captured in one forward inside a later one,
- run ordinary Python — loops, conditionals, building lists — around the traces,

all in one block, and only pay the `.save()` filter once, at the session edge.

## Ordinary Python around the traces

The session body is real Python; the nested traces run as they're reached.

```python
with model.session():
    norms = []
    for layer in range(3):
        with model.trace(P):
            norms.append(model.transformer.h[layer].output.norm())
    stacked = torch.stack(norms).save()

print([round(x, 1) for x in stacked.tolist()])   # [213.6, 658.5, 2570.4]
```

Each trace contributes one value to `norms`; only the final `stacked` tensor is
saved and returned.

## Save is still required for what you want back

`.save()` inside a session marks a value to survive **the session**. A value that
is never saved is usable in later traces but does not reach your code once the
session exits: touching `h0` after the block above raises `UnboundLocalError:
cannot access local variable 'h0' where it is not associated with a value`. The
same two traces written without a session give `NameError` instead, at the second
trace rather than after it. Calling `save()` with no session or trace active
raises as well (see [save.md](save.md)).

A value bound inside an **invoke** carries just as far as one bound in a plain
trace:

```python
with model.session():
    with model.trace() as tracer:
        with tracer.invoke(P):
            h = model.transformer.h[0].output
    with model.trace(P):
        diff = (model.transformer.h[0].output - h).abs().sum().save()
# diff == 0.0
```

## Remote sessions

A whole session runs as one remote job with `remote=True` on the session (not on
the inner traces) — the inner `with model.trace(...)` blocks execute against the
server's model when it runs the session body:

```python
with model.session(remote=True):
    with model.trace(P):
        h = model.transformer.h[0].output
    with model.trace(P):
        out = (model.transformer.h[0].output - h).abs().sum().save()
```

Use `remote="local"` to dry-run the serialize/deserialize path offline. See
[../remote/remote-session.md](../remote/remote-session.md).

## Notes

- `model.session()` returns a plain tracer that captures the block, runs it as real
  Python (nested traces execute as reached), and gates saves at its own outermost
  boundary — there is no separate session state.
- There is no top-level `nnsight.session(...)`; open one from a model:
  `model.session()`.
