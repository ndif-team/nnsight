---
title: Remote Session
one_liner: Bundle multiple traces into a single NDIF job to share one queue wait and pass values directly between traces.
tags: [remote, ndif, session]
related: [docs/remote/remote-trace.md, docs/remote/non-blocking-jobs.md, docs/remote/index.md]
sources: [src/nnsight/modeling/mixins/remotable.py:35, src/nnsight/intervention/backends/remote.py:39]
---

# Remote Session

## What this is for

A session bundles several traces into a **single** NDIF job. The whole session block is serialized as one request, queued once, executed contiguously on the server, and the saved values come back together. This is the way to run a multi-step experiment (clean run, corrupted run, patched run) without paying the queue cost three times.

## When to use / when not to use

- Use whenever you have two or more remote traces with related interventions.
- Use to share Python values between traces without round-tripping through `.save()` and a result download.
- `remote=True` goes on `model.session(...)`, **not** on the inner `model.trace(...)` calls — the whole session runs as one remote job, so the inner traces stay local (they run against the server's model when it runs the session body).
- Don't use a session if there's only one trace; just call `model.trace(..., remote=True)`.

## Canonical pattern

Activation patching across three traces, one job:

```python
from nnsight import TransformersModel, CONFIG

CONFIG.set_default_api_key("YOUR_KEY")

model = TransformersModel("meta-llama/Llama-3.1-70B")

with model.session(remote=True):
    # Trace 1: capture a clean hidden state. No .save() needed; the value is
    # used in a later trace within the same session.
    with model.trace("Megan Rapinoe plays the sport of"):
        hs = model.model.layers[5].output[:, -1, :]

    # Trace 2: clean baseline.
    with model.trace("Shaquille O'Neal plays the sport of"):
        clean = model.lm_head.output[0][-1].argmax(dim=-1).save()

    # Trace 3: patched. Reuses 'hs' captured in Trace 1.
    with model.trace("Shaquille O'Neal plays the sport of"):
        model.model.layers[5].output[:, -1, :] = hs
        patched = model.lm_head.output[0][-1].argmax(dim=-1).save()

print("clean:  ", model.tokenizer.decode(clean))
print("patched:", model.tokenizer.decode(patched))
```

What happens (`src/nnsight/modeling/mixins/remotable.py:35`):

1. `model.session(remote=True)` builds a `RemoteBackend(model.to_model_key(), blocking=True)`.
2. The session collects all inner traces into one serialized block.
3. On `__exit__`, the entire session is submitted as one request — one websocket, one queue wait, one result download.

## Cross-trace values don't need .save() — but final results do

Inside a session, traces share Python state directly. You only call `.save()` on the values you want returned to your local process.

```python
with model.session(remote=True):
    with model.trace("Hello"):
        hs = model.transformer.h[0].output       # captured but not transmitted

    with model.trace("World"):
        model.transformer.h[0].output = hs       # used directly — no save() round-trip
        out = model.lm_head.output.save()         # this one IS returned

print(out.shape)
```

This is the main reason to use a session over a sequence of separate `remote=True` traces — `hs` never leaves the server.

## Don't put remote=True on inner traces

```python
# WRONG — remote belongs on the session
with model.session(remote=True):
    with model.trace("Hello", remote=True):
        out = model.lm_head.output.save()

# CORRECT
with model.session(remote=True):
    with model.trace("Hello"):
        out = model.lm_head.output.save()
```

`session(remote=True)` already provides the remote backend; the inner traces run inside it.

## Saving collections built across traces

Build the collection inside the session with `nnsight.save(...)` and append to it:

```python
import nnsight

with model.session(remote=True):
    means = nnsight.save([])              # save the accumulator at session scope

    for i in range(12):
        with model.trace("Hello"):
            means.append(model.transformer.h[i].output.mean())

print(len(means))   # 12
```

The list is created on the server, mutated by each trace, then transmitted back at the end. A plain client-side list wouldn't come back — it has to be a saved value.

## Non-blocking sessions

`blocking=False` works with sessions too:

```python
with model.session(remote=True, blocking=False) as session:
    with model.trace("Hello"):
        out = model.lm_head.output.save()

print(session.backend.job_id)
print(session.backend.status)      # last Status seen
result = session.backend()         # poll: None until COMPLETED, then the saves dict
```

See [non-blocking-jobs.md](./non-blocking-jobs.md) for the polling loop.

## Gotchas

- `.save()` is still required for any value you want returned to your process. Cross-trace sharing inside the session is free; cross-process (server → you) is not.
- Sessions cut **queue** and **transport** overhead, not GPU time. A 5-minute session is still 5 minutes of compute.
- Variables defined outside the session can't be referenced inside it — build everything from scratch inside the session block.
- Iterating `for layer in model.transformer.h:` inside a session is ordinary Python during block capture, same as local tracing — not a server-side loop.
- **One trace fails → the whole session aborts.** If any inner trace raises, the job ends and no further traces run. Structure fault-tolerant pipelines as separate jobs.

## Related

- [remote-trace.md](./remote-trace.md) — single-trace remote runs.
- [non-blocking-jobs.md](./non-blocking-jobs.md) — submit a session and poll later.
- [ndif-overview.md](./ndif-overview.md) — request lifecycle.
