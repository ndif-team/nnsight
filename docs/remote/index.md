---
title: Remote Execution on NDIF
one_liner: Index for the docs/remote/ folder; pick a guide based on what you want to do with NDIF.
tags: [remote, ndif, index]
related: [docs/index.md, docs/usage/trace.md, docs/usage/generate.md]
sources: [src/nnsight/modeling/mixins/remotable.py:19, src/nnsight/intervention/backends/remote.py:39, src/nnsight/ndif.py:197]
---

# Remote Execution on NDIF

Up: [docs/index.md](../index.md) — all sections.

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
| Test the remote path offline, without a server (`remote="local"`) | [remote-trace.md](./remote-trace.md#try-it-offline-first-with-remotelocal) |
| Debug a "works locally, fails remotely" mismatch | [env-comparison.md](./env-comparison.md) |
| Sweep a dataset, or train an adapter/probe, server-side | [remote-dataset-sweep.md](../patterns/remote-dataset-sweep.md), [remote-training.md](../patterns/remote-training.md) |

## Related

- [docs/usage/trace.md](../usage/trace.md) — local tracing patterns; the same code runs remotely with `remote=True`.
- [docs/usage/generate.md](../usage/generate.md) — multi-token generation; combine with `remote=True` for hosted models.
