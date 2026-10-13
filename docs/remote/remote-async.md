---
title: Async Remote Jobs
one_liner: Await a remote NDIF job from an event loop, or async-iterate its raw status updates.
tags: [remote, ndif, async, asyncio]
related: [docs/remote/remote-trace.md, docs/remote/non-blocking-jobs.md, docs/remote/index.md]
sources: [src/nnsight/intervention/backends/remote.py]
---

# Async Remote Jobs

## What this is for

`AsyncRemoteBackend` waits for a remote job on an `asyncio` event loop instead of blocking a thread (`RemoteBackend`, blocking mode) or polling by hand (`blocking=False`). Submission is still synchronous — only the *waiting* is async, so the event loop stays free while the job runs on NDIF.

## When to use / when not to use

- Use inside async code, or to await many jobs concurrently with `asyncio.gather`.
- Use `async for` to react to each raw status update yourself (custom progress UI, logging, metrics).
- For ordinary blocking runs, plain `remote=True` is simpler — see [remote-trace.md](./remote-trace.md).
- To fire-and-poll without an event loop, use `blocking=False` — see [non-blocking-jobs.md](./non-blocking-jobs.md).

## Canonical pattern: await the saves dict

Pass an `AsyncRemoteBackend` as the trace's `backend`. On `__exit__` it subscribes, takes the session id and POSTs the payload synchronously; then `await` it:

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

`await backend` renders the status display and raises `RemoteError` on a server error, like the blocking backend. It returns a dict keyed by your saved variable names; the values are **not** pushed back into `logit`, since the block exited long before the job completed.

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

Each backend holds its own websocket; its blocking `recv` runs through `asyncio.to_thread`, so the loop makes progress across all of them.

## Async-iterate the raw status updates

`async for update in backend` yields each `ResponseModel` as it arrives, then the **saves dict as the final item**. This form does *not* touch the status display and does *not* raise on `ERROR` — an `ERROR` update simply ends the stream; inspect it and raise yourself if you want:

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

## Gotchas

- The non-blocking `blocking`/`job_id` poll path doesn't apply: you await or iterate, you don't call `backend()`.
- Construct the backend, enter the trace, and await it on the same thread — the websocket is opened inside the trace's `__exit__`.
- The connection closes automatically when the await resolves or the iterator finishes.

## Related

- [remote-trace.md](./remote-trace.md) — blocking `remote=True`.
- [non-blocking-jobs.md](./non-blocking-jobs.md) — submit and poll without an event loop.
- [ndif-overview.md](./ndif-overview.md) — status values.
