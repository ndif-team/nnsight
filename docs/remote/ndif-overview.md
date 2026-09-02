---
title: NDIF Overview
one_liner: How nnsight talks to NDIF — request lifecycle, meta-device models, and where execution actually happens.
tags: [remote, ndif, architecture]
related: [docs/remote/api-key-and-config.md, docs/remote/remote-trace.md, docs/remote/status-and-availability.md]
sources: [src/nnsight/modeling/mixins/remotable.py:19, src/nnsight/intervention/backends/remote.py:39, src/nnsight/schema/response.py:13, src/nnsight/ndif.py:197]
---

# NDIF Overview

## What this is for

NDIF (National Deep Inference Fabric) is a hosted service that runs nnsight intervention code on shared GPU pods. You build a model wrapper locally on the meta device (no GPU needed) and submit the serialized traced block to NDIF; the service deserializes it on a server that holds the real weights, runs the forward pass with your interventions, and streams results back.

## When to use / when not to use

- Use NDIF when the model doesn't fit on local hardware (Llama-3.1-70B/405B, DeepSeek, etc.).
- Use it when you want uniform infrastructure (no GPU setup, no model downloads).
- Don't use it for tight inner loops over tiny models — local execution is faster (no queue, no network round-trip).
- Don't use it for code you can't share with the server. The traced block, every function it references, and any registered local modules are serialized (source-based) and sent over the wire.

## Architecture

```
User process                                                NDIF
+----------------------------------+                        +-------------------------+
| TransformersModel(...)           |                        | scheduler / queue       |
|   -> meta device (no weights)    |                        |   RECEIVED -> QUEUED    |
|                                  |                        |   -> DISPATCHED         |
| with model.trace(..., remote=T): |   WS /subscribe        +-----------+-------------+
|     ...save()                    | <===== session_id =====>           |
|                                  |                                    |
|     RemoteBackend                |   POST /request (blob + env)       |
|       RequestModel.serialize()   | ---------------------------------> | model worker
|                                  |   status updates over websocket    |   pulls request
|       websocket recv loop        | <===============================>  |   runs interleaver
|       torch.load(result)         |   COMPLETED + presigned result url |   collects saves
+----------------------------------+                                    +-------------------------+
```

Source map:

- `src/nnsight/modeling/mixins/remotable.py:19` — `trace(remote=True)` builds `RemoteBackend(self.to_model_key(), host=..., env=..., blocking=..., ...)` via `_remote_backend`.
- `src/nnsight/intervention/backends/remote.py:304` — `request()` is the blocking path: subscribe, POST, then a websocket recv loop.
- `src/nnsight/schema/request.py:37` — `RequestModel.serialize(tracer, compress)` reduces the block to source + referenced globals/locals and pickles it (zstd if `COMPRESS`).
- `src/nnsight/intervention/backends/remote.py:144` — `download_result()` streams the presigned URL, decompresses, and `torch.load`s the saves.
- `src/nnsight/ndif.py:197` — `status()` queries `{HOST}/status` and lists deployed models.

## Blocking request lifecycle

The default (`blocking=True`) path, `RemoteBackend.request` (`src/nnsight/intervention/backends/remote.py:304`):

1. `trace(remote=True)` builds a `RemoteBackend`; the block is captured on `__exit__`.
2. Local (non-installed) modules are registered for by-value pickling (`pull_env`), then the block is serialized to bytes (`RequestModel.serialize`).
3. A websocket connects to `{HOST}/subscribe` and receives a `session_id` (subscribe before sending so no update is missed).
4. The payload is POSTed to `{HOST}/request` (JSON routing metadata as a form field, serialized block as a file blob). The initial `RECEIVED` response comes back over HTTP.
5. The client reads `ResponseModel` status updates off the websocket until `COMPLETED` or `ERROR`.
6. On `COMPLETED`, the result is downloaded from a presigned URL on `response.data`, decompressed, deserialized with `torch.load(..., map_location="cpu")`, and pushed back into your local frame so your `h = ...save()` variables populate.

The model itself is never serialized — it's identified by `model_key` and must already be deployed on NDIF (see [status-and-availability.md](./status-and-availability.md)).

## Job status values

Status updates are `ResponseModel` objects carrying one of these (`src/nnsight/schema/response.py:13`):

| Status | Meaning |
|--------|---------|
| `RECEIVED` | Request validated and accepted. |
| `QUEUED` | Waiting in this model's queue. |
| `PROVISIONING` | Bringing capacity up for the model. |
| `DEPLOYING` | The model deployment is coming up. |
| `DISPATCHED` | Forwarded to a model deployment; about to run. |
| `RUNNING` | Forward pass executing on the GPU pod. |
| `LOG` | A `print(...)` inside your block; a transient message, not a lifecycle stage. |
| `COMPLETED` | Saves are ready; the client downloads and loads them. |
| `ERROR` | Server-side exception; the client raises `RemoteError` with the remote traceback. |

The client renders these as a single in-place status line (animated spinner in terminals, an in-place HTML element in Jupyter). See `StatusDisplay` (`src/nnsight/intervention/backends/display.py:58`). There is no `STREAM` status — the old `tracer.local()` hybrid-streaming path does not exist here.

## What the job cost

The `COMPLETED` response also carries `meta_data` — what the run cost on the server. The backend keeps the last one it saw, and the tracer keeps the backend it ran on, so read it after the block exits (the backend runs in `__exit__`, so it is still `None` inside):

```python
with model.trace("Hello", remote=True) as tracer:
    out = model.lm_head.output.save()

print(tracer.backend.meta_data)
# {'runtime': 0.42,                      # wall-clock seconds on the server
#  'max_memory_usage': 2147483648,       # peak bytes on the worst-pressured card
#  'max_mem_by_gpu': {'0': 2147483648},  # ...per card
#  'max_mem_pct_by_gpu': {'0': 20.0}}    # ...against the headroom the job had
```

The memory figures are what *your block* drove on top of the resident weights, not the card's total usage — the weights are the server's, and they are already there before your job starts. GPU keys are strings.

It is a plain dict, not a model: a server can report more than the client knows about, and an older one that reports nothing leaves `meta_data` as `None`. Don't assume a key is present — `meta_data` is populated only on `COMPLETED`, so it stays `None` for a job that errored, and for a job that hasn't finished.

A non-blocking job's `poll()` and an `AsyncRemoteBackend` record it the same way, on the backend you already hold.

## What "meta device" means client-side

When you instantiate `TransformersModel("meta-llama/Llama-3.1-70B")` without `dispatch=True`, the model is built on `torch.device("meta")` — the architecture is constructed (so `model.transformer.h[0].output` is a real envoy path) but no weights are allocated. This is what lets a machine with no GPU write intervention code against a 70B model.

```python
from nnsight import TransformersModel

model = TransformersModel("meta-llama/Llama-3.1-70B")
print(next(model._module.parameters()).device)   # meta

with model.trace("Hello", remote=True):        # works — runs on NDIF
    out = model.lm_head.output.save()
```

The meta build lives in `MetaDevice` / `Meta` (`src/nnsight/modeling/mixins/meta.py`); `Remotable` (`src/nnsight/modeling/mixins/remotable.py`) is layered on top.

## Dry-run the remote path locally

`remote="local"` exercises the entire serialize → deserialize → execute path in-process — no server, no network. It serializes the block exactly as `RemoteBackend` would, then deserializes it *with your non-installed modules hidden* (mimicking a server that doesn't have your source files) and runs the deserialized block against the real, dispatched local model. If the block references a local function or class that wasn't shipped by value, the deserialize step raises `ModuleNotFoundError`, just as the server would.

```python
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("The Eiffel Tower is in the city of", remote="local"):
    logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

print(model.tokenizer.decode(logit))   # ' Paris'
```

A passing `remote="local"` run is strong evidence the real `remote=True` run will work. It's the recommended way to validate a remote script offline. See `LocalSimulationBackend` (`src/nnsight/intervention/backends/local.py`).

## Canonical pattern

```python
from nnsight import TransformersModel, CONFIG

CONFIG.set_default_api_key("YOUR_KEY")

model = TransformersModel("meta-llama/Llama-3.1-70B")

with model.trace("The Eiffel Tower is in the city of", remote=True):
    logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

print(model.tokenizer.decode(logit))   # ' Paris'
```

## Gotchas

- `remote=True` and `dispatch=True` are independent. `dispatch=True` allocates real weights locally; that's rarely useful with `remote=True` (you'd pay memory for nothing).
- The model identifier you instantiate locally must match an NDIF deployment. Use `nnsight.is_model_running("...")` to confirm — see [status-and-availability.md](./status-and-availability.md).
- Anything used inside the block — helper functions, custom classes — must be importable on the server or shipped by value. Local-only modules are auto-registered by `pull_env`, or register them yourself with `nnsight.register(...)` — see [register-local-modules.md](./register-local-modules.md).
- "Works locally, fails remotely" is almost always an env mismatch. See [env-comparison.md](./env-comparison.md).

## Related

- [api-key-and-config.md](./api-key-and-config.md) — auth and config.
- [remote-trace.md](./remote-trace.md) — minimal `remote=True` recipe.
- [non-blocking-jobs.md](./non-blocking-jobs.md) — submit and poll instead of blocking.
- [remote-async.md](./remote-async.md) — await a job from an event loop.
- https://discuss.ndif.us/ — service forum and outage announcements.
