---
title: Remote Execution on NDIF
one_liner: Index for the docs/remote/ folder; pick a guide based on what you want to do with NDIF.
tags: [remote, ndif, index]
related: [docs/usage/trace.md, docs/usage/generate.md]
sources: [src/nnsight/modeling/mixins/remotable.py:19, src/nnsight/intervention/backends/remote.py:39, src/nnsight/ndif.py:197]
---

# Remote Execution on NDIF

## What this is for

These docs cover running nnsight interventions on NDIF (National Deep Inference Fabric) — shared GPU infrastructure that hosts large models (Llama-3.1-70B/405B and similar). You write the same `with model.trace(...)` blocks, add `remote=True`, and the traced block is serialized, queued, executed on a remote pod, and streamed back.

NDIF service docs live at https://nnsight.net and https://login.ndif.us. These docs cover the client-side surface only.

## Decision tree

| Goal | Doc |
|------|-----|
| What is NDIF and what happens when I set `remote=True`? | [ndif-overview.md](./ndif-overview.md) |
| Get an API key, set HOST/key, configure logging | [api-key-and-config.md](./api-key-and-config.md) |
| Run a single trace remotely (most common starting point) | [remote-trace.md](./remote-trace.md) |
| Bundle several traces into one request to skip multiple queue waits | [remote-session.md](./remote-session.md) |
| Submit a job and poll for it later instead of blocking | [non-blocking-jobs.md](./non-blocking-jobs.md) |
| Await a remote job (or async-iterate its status) from an event loop | [remote-async.md](./remote-async.md) |
| Use local helper modules in remote intervention code | [register-local-modules.md](./register-local-modules.md) |
| Check if a model is up before submitting | [status-and-availability.md](./status-and-availability.md) |
| Debug a "works locally, fails remotely" mismatch | [env-comparison.md](./env-comparison.md) |

## Recipe at a glance

```python
from nnsight import TransformersModel, CONFIG

CONFIG.set_default_api_key("YOUR_KEY")            # one-time setup

model = TransformersModel("meta-llama/Llama-3.1-70B")   # builds on meta device

with model.trace("The Eiffel Tower is in the city of", remote=True):
    logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

print(model.tokenizer.decode(logit))    # ' Paris'
```

## Test the remote path without a server

`remote="local"` runs the whole serialize → deserialize → execute path in-process (no NDIF, no network), hiding your non-installed modules exactly as the server would. A passing `remote="local"` run is strong evidence the real `remote=True` run will work. See [ndif-overview.md](./ndif-overview.md#dry-run-the-remote-path-locally).

```python
with model.trace("The Eiffel Tower is in the city of", remote="local"):
    logit = model.lm_head.output[0][-1].argmax(dim=-1).save()
```

## Related

- [docs/usage/trace.md](../usage/trace.md) — local tracing patterns; the same code runs remotely with `remote=True`.
- [docs/usage/generate.md](../usage/generate.md) — multi-token generation; combine with `remote=True` for hosted models.
