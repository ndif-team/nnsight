---
title: Non-Blocking Remote Jobs
one_liner: Submit a remote trace, get the job ID immediately, and poll for the result later.
tags: [remote, ndif, async]
related: [docs/remote/remote-trace.md, docs/remote/remote-session.md, docs/remote/index.md]
sources: [src/nnsight/intervention/backends/remote.py:421, src/nnsight/intervention/backends/remote.py:990, src/nnsight/intervention/backends/remote.py:696]
---

# Non-Blocking Remote Jobs

## What this is for

By default, `remote=True` blocks: the client opens a WebSocket and waits for `COMPLETED`. Sometimes you want to fire-and-poll instead — submit the request, walk away, and check on it later. `blocking=False` swaps to that mode. The trace returns immediately after submission and you fetch results via `tracer.backend()`.

## When to use / when not to use

- Use when the job will take long enough that you want the client process free.
- Use to manage many concurrent remote jobs (one tracer per job, poll them in any order).
- Use with webhooks (`callback=` URL) so a downstream service is notified when the job completes.
- Don't use for small jobs — the polling overhead exceeds the cost of just blocking.

## Canonical pattern

```python
import time
from nnsight import LanguageModel

model = LanguageModel("meta-llama/Llama-3.1-70B")

with model.trace("Hello", remote=True, blocking=False) as tracer:
    output = model.lm_head.output.save()

# Trace context exits immediately after submission. tracer.backend is the
# RemoteBackend instance with state for polling.
backend = tracer.backend
print(backend.job_id)        # 'a3e1...' — UUID assigned by NDIF
print(backend.job_status)    # ResponseModel.JobStatus.RECEIVED

# Poll until done. backend() returns None until COMPLETED, then the dict of saves.
while True:
    result = backend()
    if result is not None:
        break
    print(f"status: {backend.job_status.name}")
    time.sleep(1)

print(result.keys())          # dict_keys(['id', 'output'])
print(result['output'].shape)
```

How polling works (`src/nnsight/intervention/backends/remote.py:990`):

- First `backend()` call: `job_id is None`, so it submits the request via HTTP POST and stores the job ID. Returns nothing.
- `blocking=False` skips the WebSocket — instead, each subsequent `backend()` calls `get_response()` which does an HTTP GET against `{HOST}/response/{job_id}` (`src/nnsight/intervention/backends/remote.py:696`).
- That endpoint returns the latest `ResponseModel`. If status is `COMPLETED`, the result URL is downloaded and returned. Otherwise `handle_response` returns `None` and you poll again.

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

The `'id'` key is the job ID, included automatically.

## Submit-and-forget with a webhook

Provide a `callback` URL when constructing `RemoteBackend` directly. NDIF will POST to that URL when the job completes, so you don't need to poll:

```python
from nnsight.intervention.backends.remote import RemoteBackend

backend = RemoteBackend(
    model_key=model.to_model_key(),
    blocking=False,
    callback="https://my-service.example.com/webhooks/ndif",
)

with model.trace("Hello", backend=backend) as tracer:
    out = model.lm_head.output.save()

print(tracer.backend.job_id)   # save this; your webhook handler can fetch the result by ID
```

The `callback` value is forwarded as the `ndif-callback` header (`src/nnsight/intervention/backends/remote.py:537`). NDIF will POST status-update objects (`RECEIVED`, `QUEUED`, `RUNNING`, `COMPLETED`, etc.) to that URL — not the result itself. Your webhook handler should call `get_response(job_id)` after seeing `COMPLETED` to download the actual saves.

## Reattaching to an existing job

If you've stored the job ID, you can construct a backend later and fetch the result without resubmitting:

```python
backend = RemoteBackend(
    model_key=model.to_model_key(),
    blocking=False,
    job_id="a3e1...",     # the stored job ID
)

result = backend.get_response()       # HTTP GET /response/{job_id}
```

`backend.get_response()` is the underlying call (`src/nnsight/intervention/backends/remote.py:696`); it returns `None` if the job isn't ready and the deserialized result if it is.

## Sessions also support blocking=False

```python
with model.session(remote=True, blocking=False) as session:
    with model.trace("Hello"):
        out = model.lm_head.output.save()

print(session.backend.job_id)
result = session.backend()   # poll
```

Same shape as a single trace: `session.backend()` returns `None` until done, then the saves dict.

## Gotchas

- `tracer.backend.job_status` updates only when you call `backend()` again — there's no background polling thread. The status is whatever the most recent HTTP response said. **`backend()` does not advance status itself**; it just fetches the latest response object from NDIF's object store. If you wait long enough and call once, you may go straight from `RECEIVED` to `COMPLETED` with no intermediate states observed.
- `backend()` blocks for one HTTP round-trip per call. Don't hammer it; sleep between polls.
- **Server-side TTL is 24 hours.** Completed-but-unfetched results are garbage-collected after 24h — store the result locally as soon as you fetch it.
- `blocking=False` with `verbose=True` can produce noisy status output every poll. The standard status display is designed for the WebSocket flow; non-blocking polls re-display each status.
- `print(...)` statements in the trace are streamed over the WebSocket in blocking mode but only appear in the **final** response in non-blocking mode (you'll see them all at once when the job completes).
- The `callback` URL receives **status-update notifications** (`RECEIVED`, `QUEUED`, `RUNNING`, `COMPLETED`, etc.) — not the actual result payload. Your webhook handler should fetch the result with `get_response(job_id)` after seeing `COMPLETED`.

## Related

- [remote-trace.md](./remote-trace.md) — blocking-mode default.
- [remote-session.md](./remote-session.md) — bundling multiple traces, also supports `blocking=False`.
- [ndif-overview.md](./ndif-overview.md) — full lifecycle including HTTP/WebSocket split.
