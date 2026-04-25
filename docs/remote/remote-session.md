---
title: Remote Session
one_liner: Bundle multiple traces into a single NDIF job to share one queue wait and pass values directly between traces.
tags: [remote, ndif, session]
related: [docs/remote/remote-trace.md, docs/remote/non-blocking-jobs.md, docs/remote/index.md]
sources: [src/nnsight/modeling/mixins/remoteable.py:76, src/nnsight/intervention/tracing/base.py:1, src/nnsight/intervention/backends/remote.py:421]
---

# Remote Session

## What this is for

A session bundles several traces into a **single** NDIF job. The whole session is serialized as one request, queued once, executed contiguously on the server, and the saved values are returned together. This is the way to run a multi-step experiment (clean run, corrupted run, patched run) without paying the queue cost three times.

## When to use / when not to use

- Use whenever you have two or more remote traces with related interventions.
- Use to share Python values between traces without round-tripping through `.save()` and a result download.
- Don't put `remote=True` on the inner `model.trace(...)` calls — `remote=True` belongs on `model.session(...)`. The session backend wraps everything inside.
- Don't use a session if there's only one trace; just call `model.trace(..., remote=True)`.

## Canonical pattern

Activation patching across three traces, one job:

```python
from nnsight import LanguageModel, CONFIG

CONFIG.set_default_api_key("YOUR_KEY")

model = LanguageModel("meta-llama/Llama-3.1-70B")

with model.session(remote=True):
    # Trace 1: capture clean hidden state. No .save() needed; the value is
    # used in a later trace within the same session.
    with model.trace("Megan Rapinoe plays the sport of"):
        hs = model.model.layers[5].output[0][:, -1, :]

    # Trace 2: clean baseline.
    with model.trace("Shaquille O'Neal plays the sport of"):
        clean = model.lm_head.output[0][-1].argmax(dim=-1).save()

    # Trace 3: patched. Reuses 'hs' captured in Trace 1.
    with model.trace("Shaquille O'Neal plays the sport of"):
        model.model.layers[5].output[0][:, -1, :] = hs
        patched = model.lm_head.output[0][-1].argmax(dim=-1).save()

print("clean:  ", model.tokenizer.decode(clean))
print("patched:", model.tokenizer.decode(patched))
```

What happens (`src/nnsight/modeling/mixins/remoteable.py:76`):

1. `model.session(remote=True)` builds a `RemoteBackend(model.to_model_key(), blocking=True)`.
2. The session collects all inner traces into a single intervention graph.
3. On `__exit__`, the entire session is serialized and submitted as one HTTP POST.
4. One WebSocket, one queue wait, one result download.

## Cross-trace values don't need .save() — but final results do

Inside a session, traces share Python state directly. You only call `.save()` on the values you want returned to your local environment.

```python
with model.session(remote=True):
    with model.trace("Hello"):
        hs = model.transformer.h[0].output[0]   # captured but not transmitted

    with model.trace("World"):
        model.transformer.h[0].output[0] = hs   # used directly — no save() round-trip
        out = model.lm_head.output.save()       # this one IS returned

print(out.shape)
```

This is the main reason to use a session over a sequence of separate `remote=True` traces. The `hs` tensor never leaves the server.

## Don't put remote=True on inner traces

```python
# WRONG — double-remote, will probably error or hang
with model.session(remote=True):
    with model.trace("Hello", remote=True):
        out = model.lm_head.output.save()

# CORRECT
with model.session(remote=True):
    with model.trace("Hello"):
        out = model.lm_head.output.save()
```

`session(remote=True)` already provides the remote backend; inner traces inherit it.

## Saving collections built across traces

Build the collection inside the session and append to it:

```python
with model.session(remote=True):
    layer_logits = list().save()        # save once at session scope

    for i in range(12):
        with model.trace("Hello"):
            layer_logits.append(model.lm_head(model.transformer.ln_f(
                model.transformer.h[i].output[0]
            )).argmax(dim=-1))

print(len(layer_logits))   # 12
```

The list is created on the server, mutated by each trace, then transmitted back at the end.

## Non-blocking sessions

`blocking=False` works with sessions too:

```python
with model.session(remote=True, blocking=False) as session:
    with model.trace("Hello"):
        out = model.lm_head.output.save()

print(session.backend.job_id)
print(session.backend.job_status)
# Poll with session.backend() — see non-blocking-jobs.md
```

See [non-blocking-jobs.md](./non-blocking-jobs.md) for the polling pattern.

## Gotchas

- `.save()` *is* still required for any value you want returned to the user process. Cross-trace sharing inside the session is free; cross-process (server -> user) is not.
- Sessions don't make the server faster — they cut **queue** and **transport** overhead. A 5-minute session is still 5 minutes of GPU time.
- Variables defined outside the session can't be referenced inside it (they aren't serialized into the request). Build everything from scratch inside the session.
- When using `for layer in model.transformer.h:` inside a session, the loop runs once during code extraction — module access is just normal Python, not a server-side iteration. This is the same as local tracing.

## Related

- [remote-trace.md](./remote-trace.md) — single-trace remote runs.
- [non-blocking-jobs.md](./non-blocking-jobs.md) — submit a session and poll for the result later.
- [ndif-overview.md](./ndif-overview.md) — request lifecycle.
