---
title: Remote Execution on NDIF
one_liner: Index for the docs/remote/ folder; pick a guide based on what you want to do with NDIF.
tags: [remote, ndif, index]
related: [docs/index.md, docs/usage/trace.md, docs/usage/generate.md]
sources: [src/nnsight/modeling/mixins/remotable.py:19, src/nnsight/intervention/backends/remote.py:39, src/nnsight/ndif.py:197]
---

# Remote Execution on NDIF

Up: [docs/index.md](../index.md) — all sections.

Running nnsight interventions on NDIF (National Deep Inference Fabric), the shared service that hosts large models: the same `with model.trace(...)` blocks with `remote=True`, executed on NDIF. These pages cover the client side; service docs are at https://nnsight.net and https://login.ndif.us.

| Goal | Doc |
|------|-----|
| Run a single trace remotely (start here); what happens on `remote=True` | [remote-trace.md](./remote-trace.md) |
| What NDIF is, job status values, what a job cost | [ndif-overview.md](./ndif-overview.md) |
| Test the remote path offline, without a server (`remote="local"`) | [ndif-overview.md](./ndif-overview.md#dry-run-the-remote-path-with-remotelocal) |
| Get an API key, set the host, configure logging | [api-key-and-config.md](./api-key-and-config.md) |
| Bundle several traces into one request to skip multiple queue waits | [remote-session.md](./remote-session.md) |
| Submit a job and poll for it later instead of blocking | [non-blocking-jobs.md](./non-blocking-jobs.md) |
| Await a remote job (or async-iterate its status) from an event loop | [remote-async.md](./remote-async.md) |
| Use local helper modules in remote intervention code | [register-local-modules.md](./register-local-modules.md) |
| Check if a model is up before submitting | [status-and-availability.md](./status-and-availability.md) |
| Debug a "works locally, fails remotely" mismatch | [env-comparison.md](./env-comparison.md) |
| Sweep a dataset server-side | [patterns/remote-dataset-sweep.md](../patterns/remote-dataset-sweep.md) |
| Train an adapter or probe server-side | [patterns/remote-training.md](../patterns/remote-training.md) |

## Related

- [docs/usage/trace.md](../usage/trace.md) — local tracing; the same code runs remotely with `remote=True`.
- [docs/usage/generate.md](../usage/generate.md) — multi-token generation.
