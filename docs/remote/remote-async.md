---
title: Async Remote Jobs
one_liner: Await a remote NDIF job from an event loop, or async-iterate its raw status updates.
tags: [remote, ndif, async, asyncio]
related: [docs/remote/remote-trace.md, docs/remote/non-blocking-jobs.md, docs/remote/index.md]
sources: [src/nnsight/intervention/backends/remote.py:339, src/nnsight/intervention/backends/remote.py:394, src/nnsight/intervention/backends/remote.py:417]
---

# Async Remote Jobs

## What this is for

`AsyncRemoteBackend` waits for a remote job on an `asyncio` event loop instead of blocking a thread (`RemoteBackend`, blocking mode) or polling by hand (`blocking=False`). Submission is still synchronous — only the *waiting* is async, so the event loop stays free while the job runs on NDIF. Use it to run several remote jobs concurrently, or inside an async app.

## When to use / when not to use

- Use inside async code (an event loop is already running) or to await many jobs concurrently with `asyncio.gather`.
- Use `async for` when you want to react to each raw status update yourself (custom progress UI, logging, metrics).
- For ordinary blocking runs, plain `remote=True` is simpler — see [remote-trace.md](./remote-trace.md).
- To fire-and-poll without an event loop, use `blocking=False` — see [non-blocking-jobs.md](./non-blocking-jobs.md).

## Canonical pattern: await the saves dict

Pass an `AsyncRemoteBackend` as the trace's `backend`. The trace body runs the same as always; the backend is fired synchronously on `__exit__` (subscribe, take the session id, POST the payload). Then `await` it for the saves dict:

```python
import asyncio
from nnsight import TransformersModel
from nnsight.intervention.backends.remote import AsyncRemoteBackend

model = TransformersModel("meta-llama/Llama-3.1-70B")

async def main():
    backend = AsyncRemoteBackend(model.to_model_key())

    with model.trace("The Eiffel Tower is in the city of", backend=backend):
        logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

    result = await backend                  # wait for COMPLETED, get the saves
    print(model.tokenizer.decode(result["logit"]))

asyncio.run(main())
```

`await backend` renders the status display and raises `RemoteError` on a server error, exactly like the blocking parent (`src/nnsight/intervention/backends/remote.py:398`). The result is a dict keyed by your saved variable names.

## Result comes out of the await, not your frame

By the time the job completes the trace block has long since exited, so — unlike blocking `remote=True` — the saved values are **not** pushed back into your `logit` variable. Read them from the awaited dict (`result["logit"]`), same as the non-blocking poll.

## Running several jobs concurrently

```python
async def run(prompt):
    backend = AsyncRemoteBackend(model.to_model_key())
    with model.trace(prompt, backend=backend):
        out = model.lm_head.output[0][-1].argmax(dim=-1).save()
    result = await backend
    return model.tokenizer.decode(result["out"])

async def main():
    prompts = ["The Eiffel Tower is in", "The capital of Japan is"]
    return await asyncio.gather(*(run(p) for p in prompts))
```

Each backend holds its own websocket; the blocking `recv` is run through `asyncio.to_thread` so the loop keeps making progress across all of them (`src/nnsight/intervention/backends/remote.py:437`).

## Async-iterate the raw status updates

`async for update in backend` yields each `ResponseModel` as it arrives, then the **saves dict as the final item** once the job completes. This form does *not* touch the status display and does *not* raise on `ERROR` — it hands you each update raw, to do with as you like; an `ERROR` update simply ends the stream (inspect it and raise yourself if you want) (`src/nnsight/intervention/backends/remote.py:417`):

```python
async def main():
    backend = AsyncRemoteBackend(model.to_model_key())

    with model.trace("Hello", backend=backend):
        out = model.lm_head.output.save()

    result = None
    async for update in backend:
        if isinstance(update, dict):
            result = update                 # the saves dict, yielded last
        else:
            print(update.status, update.description)   # raw ResponseModel
    print(result["out"].shape)
```

Distinguish the two kinds of item by type: every status update is a `ResponseModel`; the single final item is a plain `dict` of saves.

## Gotchas

- `AsyncRemoteBackend` is always effectively blocking-style over one websocket — the `blocking`/`job_id` non-blocking poll path doesn't apply; you await or iterate, you don't call `backend()`.
- The websocket is opened and the request POSTed synchronously inside the trace's `__exit__`; only the status stream is awaited. Construct the backend and enter the trace on the same thread that later awaits it.
- `await backend` renders the display and raises on `ERROR`; `async for` does neither. Pick the form that matches whether you want nnsight's handling or your own.
- The connection is closed automatically when the await resolves or the async iterator finishes.

## Related

- [remote-trace.md](./remote-trace.md) — blocking `remote=True`.
- [non-blocking-jobs.md](./non-blocking-jobs.md) — submit and poll without an event loop.
- [ndif-overview.md](./ndif-overview.md) — request lifecycle and status values.
