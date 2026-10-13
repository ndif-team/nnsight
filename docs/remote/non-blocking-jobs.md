---
title: Non-Blocking Remote Jobs
one_liner: Submit a remote trace or session, get the job ID immediately, and poll for the result later.
tags: [remote, ndif, async]
related: [docs/remote/remote-trace.md, docs/remote/remote-session.md, docs/remote/remote-async.md, docs/remote/index.md]
sources: [src/nnsight/intervention/backends/remote.py]
---

# Non-Blocking Remote Jobs

## What this is for

By default, `remote=True` blocks: the client holds a websocket open and waits for `COMPLETED`. `blocking=False` swaps to fire-and-poll — the trace submits the request and returns immediately; you fetch the result later by calling the backend.

## When to use / when not to use

- Use when the job runs long enough that you want the client process free.
- Use to manage many concurrent remote jobs (one backend per job, poll them in any order).
- Don't use for small jobs — the polling overhead exceeds the cost of just blocking.
- To wait on a job from an async event loop instead of polling, use `AsyncRemoteBackend` — see [remote-async.md](./remote-async.md).

## Canonical pattern

```python
import time
from nnsight import TransformersModel

model = TransformersModel("meta-llama/Llama-3.1-70B")

with model.trace("Hello", remote=True, blocking=False) as tracer:
    embeds = model.model.embed_tokens.output.save()
    logits = model.lm_head.output.save()

backend = tracer.backend
print(backend.job_id)     # 'a3e1…' — id assigned by NDIF
print(backend.status)     # Status.RECEIVED

# backend() returns None until COMPLETED, then the saves dict.
while (result := backend()) is None:
    print(f"status: {backend.status.name}")
    time.sleep(1)

result["embeds"].shape    # keyed by the saved variable's name
result["logits"].shape
```

How it works:

- The trace's `__exit__` calls the backend once. With `job_id` still `None`, that runs `submit()` — a POST to `{HOST}/request` **without** a websocket. It stores the returned `job_id` and returns `None`.
- Each later `backend()` call runs `poll()`: a GET against `{HOST}/response/{job_id}`. It returns the saves dict on `COMPLETED`, raises `RemoteError` on `ERROR`, and returns `None` while the job is still running (or before its first status lands). Saved values are **not** pushed back into your variables — read them from the returned dict.

## Reattaching to an existing job

If you've stored the job id, construct a poll-only backend later and fetch the result without resubmitting:

```python
from nnsight.intervention.backends.remote import RemoteBackend

backend = RemoteBackend(model.to_model_key(), blocking=False, job_id="a3e1…")

result = backend()        # polls, since job_id is set; None until COMPLETED
```

## Sessions

`blocking=False` works on `model.session(...)` the same way:

```python
with model.session(remote=True, blocking=False) as session:
    with model.trace("Hello"):
        out = model.lm_head.output.save()

print(session.backend.job_id)
result = session.backend()   # poll: None until done, then the saves dict
```

## Gotchas

- **Status advances only when you poll.** There's no background thread: each `backend()` call (one HTTP round-trip) fetches whatever status the server last recorded and updates `backend.status`. Call once after a long wait and you may jump straight from `RECEIVED` to the result. Sleep between polls; don't hammer it.
- `print(...)` output from the block streams live only in blocking mode; non-blocking, you see only whatever status the last poll fetched.
- Completed-but-unfetched results may be garbage-collected server-side after a TTL. Fetch and store the result promptly.

## Related

- [remote-trace.md](./remote-trace.md) — blocking-mode default.
- [remote-session.md](./remote-session.md) — bundling multiple traces.
- [remote-async.md](./remote-async.md) — await a job from an event loop instead of polling.
