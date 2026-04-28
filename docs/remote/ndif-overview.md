---
title: NDIF Overview
one_liner: How nnsight talks to NDIF — request lifecycle, meta-device models, and where execution actually happens.
tags: [remote, ndif, architecture]
related: [docs/remote/api-key-and-config.md, docs/remote/remote-trace.md, docs/remote/status-and-availability.md]
sources: [src/nnsight/modeling/mixins/remoteable.py:31, src/nnsight/intervention/backends/remote.py:421, src/nnsight/ndif.py:247, src/nnsight/schema/response.py:13]
---

# NDIF Overview

## What this is for

NDIF (Neural Network Distributed Inference Framework) is a hosted service that runs nnsight intervention code on shared GPU pods. Users build model wrappers locally on the meta device (no GPU needed) and submit serialized intervention graphs to NDIF; the service deserializes them on a server with the real weights, runs the forward pass with hooks, and streams results back.

## When to use / when not to use

- Use NDIF when the model doesn't fit on local hardware (Llama-3.1-70B/405B, DeepSeek, etc.).
- Use it when you want a uniform infrastructure (no GPU setup, no model downloads).
- Don't use it for tight inner loops over tiny models — local execution is faster (no queue, no network round-trip).
- Don't use it for code you can't share with the server. Mediator code, all referenced functions, and any registered modules are serialized via cloudpickle and sent over the wire.

## Architecture

```
User process                                                NDIF
+----------------------------------+                        +-------------------------+
| LanguageModel(...)               |                        | scheduler / queue       |
|   -> meta device (no weights)    |                        |   RECEIVED -> QUEUED    |
|                                  |                        |   -> DISPATCHED         |
| with model.trace(..., remote=T): |                        +-----------+-------------+
|     ...save()                    |                                    |
|                                  |   POST /request (compressed bytes) |
|     RemoteBackend                | ---------------------------------> |
|       RequestModel.serialize()   |                                    |
|                                  |   WebSocket /ws/socket.io          |
|       socketio.SimpleClient      | <===============================>  | model worker
|                                  |   ResponseModel: status updates    |   pulls request
|       _decompress_and_load()     |   COMPLETED + result URL/bytes     |   runs interleaver
+----------------------------------+                                    |   collects saves
                                                                        +-------------------------+
```

Source map:

- `src/nnsight/modeling/mixins/remoteable.py:31` — `trace(remote=True)` constructs `RemoteBackend(self.to_model_key(), blocking=blocking)`.
- `src/nnsight/intervention/backends/remote.py:421` — `RemoteBackend` orchestrates HTTP submit + WebSocket polling.
- `src/nnsight/intervention/backends/remote.py:514` — `request()` calls `RequestModel(...).serialize(self.compress)` (zstd).
- `src/nnsight/intervention/backends/remote.py:869` — `blocking_request` is the default path (WebSocket loop, status updates, result download).
- `src/nnsight/ndif.py:247` — `status()` queries `{HOST}/status`; lists which models are running.

## Job lifecycle

The server emits `ResponseModel` objects with these statuses (`src/nnsight/schema/response.py:17`):

| Status | Meaning |
|--------|---------|
| `RECEIVED` | API key accepted, request validated. |
| `QUEUED` | Waiting in this model's queue (other users ahead of you, or the model is warming). |
| `DISPATCHED` | Forwarded to a model deployment; about to run. |
| `RUNNING` | Forward pass executing on the GPU pod. |
| `LOG` | A `print(...)` inside your trace; appears as a single log line in the status display. |
| `COMPLETED` | Saves are ready; client downloads them and unpickles. |
| `STREAM` | Server pushed a function for hybrid local/remote execution (rare, used by `tracer.local()`). |
| `ERROR` | Server-side exception; client raises `RemoteException` with the remote traceback. |

The client renders these as a single-line spinner in terminals or an in-place HTML element in Jupyter. See `JobStatusDisplay` (`src/nnsight/intervention/backends/remote.py:92`) for the renderer.

## What "meta device" means client-side

When you instantiate `LanguageModel("meta-llama/Llama-3.1-70B")` without `dispatch=True`, the model is built on `torch.device("meta")` — the architecture is constructed (so `model.transformer.h[0].output` is a real envoy path) but no weights are allocated. This is what lets users with no GPU still write intervention code against a 70B model.

```python
model = LanguageModel("meta-llama/Llama-3.1-70B")
print(model.device)  # meta

with model.trace("Hello", remote=True):       # works — runs on NDIF
    out = model.lm_head.output.save()

with model.trace("Hello"):                    # fails — no weights locally
    out = model.lm_head.output.save()
```

The meta device behavior comes from `MetaMixin._load_meta` (`src/nnsight/modeling/mixins/meta.py:66`). `RemoteableMixin` is built on top.

## Canonical pattern

```python
from nnsight import LanguageModel, CONFIG

CONFIG.set_default_api_key("YOUR_KEY")

model = LanguageModel("meta-llama/Llama-3.1-70B")

with model.trace("The Eiffel Tower is in the city of", remote=True):
    logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

print(model.tokenizer.decode(logit))   # 'Paris'
```

## Gotchas

- `remote=True` and `dispatch=True` are independent. `dispatch=True` allocates real weights locally; rarely useful with `remote=True` (you'd be paying memory for nothing).
- The model identifier you instantiate locally must match an NDIF deployment. Use `nnsight.ndif_status()` to confirm. See [status-and-availability.md](./status-and-availability.md).
- Anything used inside the trace body — helper functions, custom modules, classes — must be importable on the server. Local-only modules need `nnsight.register(...)`. See [register-local-modules.md](./register-local-modules.md).
- "Works locally, fails remotely" is almost always an env mismatch. See [env-comparison.md](./env-comparison.md).

## Related

- [api-key-and-config.md](./api-key-and-config.md) — auth and config.
- [remote-trace.md](./remote-trace.md) — minimal `remote=True` recipe.
- [non-blocking-jobs.md](./non-blocking-jobs.md) — long-running jobs without blocking.
- https://discuss.ndif.us/ — service forum and outage announcements.
