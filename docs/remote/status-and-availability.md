---
title: NDIF Status and Availability
one_liner: Check the NDIF service state and whether a specific model is currently running before submitting.
tags: [remote, ndif, status]
related: [docs/remote/index.md, docs/remote/api-key-and-config.md]
sources: [src/nnsight/ndif.py:247, src/nnsight/ndif.py:326, src/nnsight/ndif.py:72]
---

# NDIF Status and Availability

## What this is for

Models on NDIF can be in different deployment states (running, deploying, scheduled-but-cold, down). Submitting a request to a non-running model still works — it gets queued — but it may take a long time to dispatch. These functions tell you what's actually live so you can fail fast or pick a different revision.

## Canonical pattern

```python
import nnsight

print(nnsight.status())   # formatted table of every deployment
nnsight.is_model_running("meta-llama/Llama-3.1-70B")    # -> True / False
```

## status() — full deployment table

```python
import nnsight

s = nnsight.status()
print(s)
```

Output (from the docstring example, `src/nnsight/ndif.py:247`):

```
NDIF Service: Up
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┓
┃ Model Class   ┃ Repo ID                    ┃ Revision ┃ Type      ┃ Status  ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━┩
│ LanguageModel │ meta-llama/Llama-3.1-70B   │ main     │ Dedicated │ RUNNING │
└───────────────┴────────────────────────────┴──────────┴───────────┴─────────┘
```

`status()` returns an `NdifStatus` object that subclasses `dict` (`src/nnsight/ndif.py:72`), so you can also iterate it programmatically:

```python
s = nnsight.status()

for repo_id, info in s.items():
    print(repo_id, info['model_class'], info['type'].value, info['state'].value)
```

`status(raw=True)` returns the raw API JSON without formatting.

`nnsight.ndif_status()` is the deprecated alias kept for backwards compatibility. Use `status()` going forward.

## NdifStatus enums

`NdifStatus.Status` (overall service):

| Value | Meaning |
|-------|---------|
| `UP` | At least one model is `RUNNING`. |
| `REDEPLOYING` | No model is running, but at least one is `DEPLOYING` or `NOT_DEPLOYED`. |
| `DOWN` | API request failed or no models are running or being deployed. |

`NdifStatus.ModelStatus` (per-model):

| Value | Meaning |
|-------|---------|
| `RUNNING` | Fully deployed; accepting requests immediately. |
| `DEPLOYING` | Coming up; new requests will queue. |
| `NOT_DEPLOYED` | Configured but not started; may take longer to become available. |
| `DOWN` | Deployment failed or unavailable. |

`NdifStatus.DeploymentType`:

| Value | Meaning |
|-------|---------|
| `DEDICATED` | Permanent deployment. |
| `PILOT_ONLY` | Available only to pilot users. |
| `SCHEDULED` | Runs on a schedule (e.g., specific hours). |

Sources: `src/nnsight/ndif.py:90`, `src/nnsight/ndif.py:113`, `src/nnsight/ndif.py:138`.

## is_model_running

For a yes/no answer about a single model:

```python
if nnsight.is_model_running("meta-llama/Llama-3.1-70B"):
    with model.trace("Hello", remote=True):
        out = model.lm_head.output.save()
else:
    print("Model not currently running on NDIF — request will queue and warm.")
```

Implementation (`src/nnsight/ndif.py:326`):
- Resolves the repo_id via `HfApi().model_info(repo_id).id` (handles redirects/aliases).
- Walks `response["deployments"]` and matches on `repo_id` and `revision` (defaults to `"main"`).
- Returns `True` only if `application_state == "RUNNING"`.

Custom revision:

```python
nnsight.is_model_running("meta-llama/Llama-3.1-70B", revision="my-finetune-branch")
```

## "Deployed but not running" — what it means

Some models are listed in the status table but not actively serving:

- **`DEPLOYING`** — the deployment is starting up. Requests submitted now will queue and start running shortly.
- **`NOT_DEPLOYED`** — the deployment is configured but cold. Submitting a request may trigger a warm-up depending on NDIF's policy; this can take significant time.
- **`SCHEDULED` type** — only runs at certain times (e.g., overnight). Outside its window, requests queue until the next start.

For real-time status outside Python, NDIF maintains a status page and the `/status` endpoint at `{CONFIG.API.HOST}/status`. The API URL is set via `CONFIG.API.HOST` (`src/nnsight/ndif.py:191`).

## Custom HOST

If you've pointed `CONFIG.API.HOST` at a self-hosted or staging deployment, `status()` and `is_model_running()` automatically query that host:

```python
from nnsight import CONFIG
CONFIG.API.HOST = "https://staging.api.ndif.us"

nnsight.status()    # queries staging
```

## Gotchas

- `status()` fails gracefully — on a network error it prints to stderr and returns an empty dict (`src/nnsight/ndif.py:275`). Check `s.status` if you need a hard signal.
- `is_model_running` returns `False` on network failure too. Distinguish "false-because-down" from "false-because-no-network" by also calling `status()`.
- The repo ID lookup uses HuggingFace; an unauthenticated rate limit can delay results if you call `is_model_running` in a tight loop. Cache the answer.
- `RUNNING` doesn't mean *no* queue — it means the deployment is live. Other users may be ahead of you.

## Related

- https://discuss.ndif.us/ — outage reports.
- [api-key-and-config.md](./api-key-and-config.md) — `CONFIG.API.HOST`.
- [ndif-overview.md](./ndif-overview.md) — what happens when a job is submitted.
