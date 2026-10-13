---
title: NDIF Overview
one_liner: What NDIF is, why the client model is on the meta device, job status values, what a job cost, and dry-running the remote path with remote="local".
tags: [remote, ndif, architecture]
related: [docs/remote/api-key-and-config.md, docs/remote/remote-trace.md, docs/remote/status-and-availability.md]
sources: [src/nnsight/modeling/mixins/remotable.py, src/nnsight/intervention/backends/remote.py, src/nnsight/schema/response.py, src/nnsight/ndif.py]
---

# NDIF Overview

## What this is for

NDIF (National Deep Inference Fabric) is a hosted service that runs nnsight intervention code on shared GPUs. You build the model wrapper locally on the meta device (no GPU needed), and `remote=True` sends the traced block to NDIF, which runs it against the real weights and returns your saved values. The model itself is never sent — it is identified by its model key and must already be deployed (see [status-and-availability.md](./status-and-availability.md)). The first run and the request lifecycle are in [remote-trace.md](./remote-trace.md).

## When to use / when not to use

- Use NDIF when the model doesn't fit on local hardware (Llama-3.1-70B/405B, DeepSeek, etc.), or to skip GPU setup and model downloads.
- Don't use it for tight inner loops over tiny models — local execution has no queue and no network round-trip.
- Don't use it for code you can't share. The traced block, every function it references, and any registered local modules are serialized (source-based) and sent to the server.

## The client model is on the meta device

`TransformersModel("meta-llama/Llama-3.1-70B")` without `dispatch=True` builds the architecture on `torch.device("meta")`: every module path (`model.model.layers[0].output`) is a real envoy, but no weights are allocated. That is what lets a machine with no GPU write intervention code against a 70B model. `dispatch=True` allocates real weights locally, which is rarely useful with `remote=True`.

```python
model = TransformersModel("meta-llama/Llama-3.1-70B")
print(next(model._module.parameters()).device)   # meta
```

## Job status values

Status updates are `ResponseModel` objects carrying one of these (`Status` in `src/nnsight/schema/response.py`):

| Status | Meaning |
|--------|---------|
| `RECEIVED` | Request validated and accepted. |
| `QUEUED` | Waiting in this model's queue. |
| `PROVISIONING` | Bringing capacity up for the model. |
| `DEPLOYING` | The model deployment is coming up. |
| `DISPATCHED` | Forwarded to a model deployment; about to run. |
| `RUNNING` | Your block is executing. |
| `LOG` | A `print(...)` inside your block; a transient message, not a lifecycle stage. |
| `COMPLETED` | Saves are ready; the client receives and loads them. |
| `ERROR` | Server-side exception; the client raises `RemoteError` with the remote traceback. |

The client renders these as a single in-place status line (a spinner in terminals, an HTML element in Jupyter); see `StatusDisplay` in `src/nnsight/intervention/backends/display.py`.

## What the job cost

The `COMPLETED` response carries `meta`, a plain dict of what the run cost on the server. Read it off `tracer.backend.meta` after the block exits (the backend runs in `__exit__`, so it is `None` inside):

```python
with model.trace("Hello", remote=True) as tracer:
    out = model.lm_head.output.save()

meta = tracer.backend.meta
meta["runtime"]             # 0.42
meta["max_mem_by_gpu"]      # {'0': 2147483648}
```

| Key | Meaning |
|---|---|
| `runtime` | Wall-clock seconds the block ran on the server. |
| `max_mem_by_gpu` | Peak bytes the block drove *on top of the resident weights*, per card — not the card's total usage. |
| `max_mem_pct_by_gpu` | `max_mem_by_gpu` against the headroom the job had (its share of the card, less the weights), as a percentage. 100 means it filled everything left for it. |
| `max_memory_usage` | `max_mem_by_gpu` on the worst-pressured card. |
| `alloc_shortfall_by_gpu` | Only on an out-of-memory failure; see below. |

GPU maps are keyed by device id as a **string** (`["0"]`, not `[0]`). Every key is optional — an older server sends no `meta` — so use `.get()`.

A **failed** job reports its cost too; the backend records it before raising:

```python
from nnsight.intervention.backends.remote import RemoteError

try:
    with model.trace(prompt, remote=True) as tracer:
        acts = model.model.layers[-1].output.save()
except RemoteError:
    print(tracer.backend.meta.get("alloc_shortfall_by_gpu"))   # {'0': 1310000000}
```

`alloc_shortfall_by_gpu` gives, per card that ran out, the part of the refused allocation that would not fit — how much you need to free, which neither the allocation size nor the traceback tells you. On a sharded model, *which* card is itself the finding. It is approximate, and `None` when the server could not tell which card refused. Non-blocking `poll()` and `AsyncRemoteBackend` record `meta` the same way, on the backend you hold.

## Dry-run the remote path with remote="local"

`remote="local"` exercises the whole serialize → deserialize → execute path in-process — no server, no key. It serializes the block exactly as `RemoteBackend` would, deserializes it *with your non-installed modules hidden* (mimicking a server that lacks your source files), and runs it against your real, dispatched local model. A local function or class that wasn't shipped by value raises `ModuleNotFoundError` at the deserialize step, just as the server would.

```python
model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("The Eiffel Tower is in the city of", remote="local"):
    logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

print(model.tokenizer.decode(logit))   # ' Paris'
```

A passing `remote="local"` run is strong evidence the real `remote=True` run will serialize and ship correctly. See `LocalSimulationBackend` (`src/nnsight/intervention/backends/local.py`).

## Gotchas

- The model identifier you instantiate locally must match an NDIF deployment. Confirm with `nnsight.is_model_running("...")` — see [status-and-availability.md](./status-and-availability.md).
- Helpers and classes used in the block must be importable on the server or shipped by value — see [register-local-modules.md](./register-local-modules.md).
- "Works locally, fails remotely" is almost always an environment mismatch — see [env-comparison.md](./env-comparison.md).

## Related

- [remote-trace.md](./remote-trace.md) — first remote run and the request lifecycle.
- [api-key-and-config.md](./api-key-and-config.md) — auth and config.
- [non-blocking-jobs.md](./non-blocking-jobs.md), [remote-async.md](./remote-async.md) — submit and poll, or await from an event loop.
- https://discuss.ndif.us/ — service forum and outage announcements.
