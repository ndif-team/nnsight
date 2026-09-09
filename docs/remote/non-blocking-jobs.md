---
title: Non-Blocking Remote Jobs
one_liner: Submit a remote trace, get the job ID immediately, and poll for the result later.
tags: [remote, ndif, async]
related: [docs/remote/remote-trace.md, docs/remote/remote-session.md, docs/remote/remote-async.md, docs/remote/index.md]
sources: [src/nnsight/intervention/backends/remote.py:93, src/nnsight/intervention/backends/remote.py:263, src/nnsight/intervention/backends/remote.py:280]
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
    output = model.lm_head.output.save()

# The trace context submitted the job and exited. tracer.backend is the
# RemoteBackend, now holding the job id.
backend = tracer.backend
print(backend.job_id)     # 'a3e1…' — id assigned by NDIF
print(backend.status)     # Status.RECEIVED

# Poll until done. backend() returns None until COMPLETED, then the saves dict.
while True:
    result = backend()
    if result is not None:
        break
    print(f"status: {backend.status.name}")
    time.sleep(1)

print(result.keys())            # dict_keys(['output'])
print(result['output'].shape)
```

How it works (`src/nnsight/intervention/backends/remote.py:93`):

- The trace's `__exit__` calls the backend once with the tracer. With `job_id` still `None`, that runs `submit()` — a POST to `{HOST}/request` **without** a websocket; the server records each status to its object store. It stores the returned `job_id` and returns `None`.
- Each later `backend()` call runs `poll()`: an HTTP GET against `{HOST}/response/{job_id}` (`src/nnsight/intervention/backends/remote.py:280`). It returns the deserialized saves dict on `COMPLETED`, raises `RemoteError` on `ERROR`, and returns `None` while the job is still running (or before its first status lands — a 404).

## Result shape

The returned dict is keyed by the saved variable's name in your trace:

```python
with model.trace("Hello", remote=True, blocking=False) as tracer:
    embeds = model.transformer.wte.output.save()
    logits = model.lm_head.output.save()

backend = tracer.backend
# ...later...
result = backend()
result['embeds'].shape
result['logits'].shape
```

## Reattaching to an existing job

If you've stored the job id, construct a poll-only backend later and fetch the result without resubmitting:

```python
from nnsight.intervention.backends.remote import RemoteBackend

backend = RemoteBackend(
    model.to_model_key(),
    blocking=False,
    job_id="a3e1…",       # the stored job id
)

result = backend()        # GET /response/{job_id}; None until COMPLETED, then the saves
```

Because `job_id` is already set, the first `backend()` polls rather than submitting. `backend.poll()` is the underlying call (`src/nnsight/intervention/backends/remote.py:280`).

## Sessions also support blocking=False

```python
with model.session(remote=True, blocking=False) as session:
    with model.trace("Hello"):
        out = model.lm_head.output.save()

print(session.backend.job_id)
result = session.backend()   # poll: None until done, then the saves dict
```

## Gotchas

- **`backend()` does not advance status on its own** — there's no background polling thread. Each call fetches whatever status the server last recorded. Call once after a long wait and you may jump straight from `RECEIVED` to the completed result, observing no intermediate states.
- `backend.status` is updated only by a `backend()` (poll) call; it holds the most recent `Status`.
- `backend()` blocks for one HTTP round-trip per call. Sleep between polls; don't hammer it.
- `print(...)` statements stream live over the websocket in blocking mode, but in non-blocking mode you only see whatever status the last poll fetched.
- Completed-but-unfetched results may be garbage-collected server-side after a TTL. Fetch and store the result promptly.

## Related

- [remote-trace.md](./remote-trace.md) — blocking-mode default.
- [remote-session.md](./remote-session.md) — bundling multiple traces; also supports `blocking=False`.
- [remote-async.md](./remote-async.md) — await a job from an event loop instead of polling.
- [ndif-overview.md](./ndif-overview.md) — full lifecycle including the HTTP/websocket split.
