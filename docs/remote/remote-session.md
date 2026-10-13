---
title: Remote Session
one_liner: Bundle multiple traces into a single NDIF job to share one queue wait and pass values directly between traces.
tags: [remote, ndif, session]
related: [docs/remote/remote-trace.md, docs/remote/non-blocking-jobs.md, docs/remote/index.md]
sources: [src/nnsight/modeling/mixins/remotable.py, src/nnsight/intervention/backends/remote.py]
---

# Remote Session

## What this is for

A session bundles several traces into a **single** NDIF job: the whole session block is serialized as one request, queued once, executed contiguously on the server, and the saved values come back together. This is the way to run a multi-step experiment (clean run, corrupted run, patched run) without paying the queue cost three times.

## When to use / when not to use

- Use whenever you have two or more remote traces with related interventions.
- Don't use a session for a single trace; call `model.trace(..., remote=True)`.

## Canonical pattern

Activation patching across three traces, one job:

```python
from nnsight import TransformersModel, CONFIG

CONFIG.set_default_api_key("YOUR_KEY")

model = TransformersModel("meta-llama/Llama-3.1-70B")

with model.session(remote=True):
    with model.trace("Megan Rapinoe plays the sport of"):
        hs = model.model.layers[5].output[:, -1, :]          # used below; not saved

    with model.trace("Shaquille O'Neal plays the sport of"):
        clean = model.lm_head.output[0][-1].argmax(dim=-1).save()

    with model.trace("Shaquille O'Neal plays the sport of"):
        model.model.layers[5].output[:, -1, :] = hs
        patched = model.lm_head.output[0][-1].argmax(dim=-1).save()

print("clean:  ", model.tokenizer.decode(clean))
print("patched:", model.tokenizer.decode(patched))
```

`model.session(remote=True)` builds one `RemoteBackend`; on `__exit__` the entire session is submitted as one request — one websocket, one queue wait, one result.

## Cross-trace values don't need .save() — results do

Traces inside a session share Python state directly on the server. `hs` above is read in trace 1 and written in trace 3 without ever leaving the server — the main reason to use a session over separate `remote=True` traces. Only values you want back in your local process need `.save()`.

## remote=True goes on the session, not the inner traces

```python
# WRONG
with model.session(remote=True):
    with model.trace("Hello", remote=True):
        out = model.lm_head.output.save()

# CORRECT — the inner traces run inside the session's remote job
with model.session(remote=True):
    with model.trace("Hello"):
        out = model.lm_head.output.save()
```

## Saving collections built across traces

Create the collection inside the session with `nnsight.save(...)` and append to it:

```python
import nnsight

with model.session(remote=True):
    means = nnsight.save([])              # save the accumulator at session scope

    for i in range(12):
        with model.trace("Hello"):
            means.append(model.model.layers[i].output.mean())

print(len(means))   # 12
```

The list is created on the server, mutated by each trace, and transmitted back at the end. A plain client-side list wouldn't come back.

`blocking=False` works on sessions too — see [non-blocking-jobs.md](./non-blocking-jobs.md#sessions).

## Gotchas

- Sessions cut **queue** and **transport** overhead, not GPU time. A 5-minute session is still 5 minutes of compute.
- **Sessions have a server-side time limit.** A session that runs past it is killed, so split a long sweep into several sessions.
- Names the block reads from the enclosing scope **are** captured: every one is pickled into the request payload. They must be picklable, they arrive on CPU in their client dtype, and their size is added to your upload — the usual reason a sweep's payload gets large. Build data inside the block when you want it *off* the wire.
- Iterating `for layer in model.model.layers:` inside a session is ordinary Python during block capture, same as local tracing — not a server-side loop.
- **One trace fails → the whole session aborts.** If any inner trace raises, the job ends and no further traces run. Structure fault-tolerant pipelines as separate jobs.

## Related

- [remote-trace.md](./remote-trace.md) — single-trace remote runs.
- [non-blocking-jobs.md](./non-blocking-jobs.md) — submit a session and poll later.
- [docs/usage/session.md](../usage/session.md) — sessions in general.
