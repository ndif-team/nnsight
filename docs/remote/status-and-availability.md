---
title: NDIF Status and Availability
one_liner: Check the NDIF service state and whether a specific model is currently running before submitting.
tags: [remote, ndif, status]
related: [docs/remote/index.md, docs/remote/api-key-and-config.md]
sources: [src/nnsight/ndif.py:197, src/nnsight/ndif.py:249, src/nnsight/ndif.py:128]
---

# NDIF Status and Availability

## What this is for

Models on NDIF can be in different deployment states (running, deploying, unhealthy). Submitting a request to a non-running model still works — it queues — but may take a while to dispatch. These functions tell you what's live so you can fail fast or pick a different revision.

## Canonical pattern

```python
import nnsight

print(nnsight.status())                                    # formatted table
nnsight.is_model_running("meta-llama/Llama-3.1-70B")       # -> True / False
```

## status() — deployment table

```python
import nnsight

s = nnsight.status()
print(s)
```

Output:

```
NDIF Service: Up 🟢

Model Class        Repo ID                   Revision  Level  State
-----------------  ------------------------  --------  -----  ---------
TransformersModel  meta-llama/Llama-3.1-70B  main      HOT    RUNNING
TransformersModel  openai-community/gpt2     main      WARM   DEPLOYING
```

Only **deployed** models appear — those at level `HOT` or `WARM` (`COLD`, i.e. downloaded but not up, is filtered out; `src/nnsight/ndif.py:222`).

`status()` returns an `NdifStatus` (`src/nnsight/ndif.py:128`). It's a dict-like view over `deployments`, so you can inspect it programmatically:

```python
s = nnsight.status()

print(s.status)                      # NdifStatus.Status.UP
for repo_id in s:                    # iterates deployment repo ids
    info = s[repo_id]
    print(repo_id, info["model_class"], info["level"], info["state"])
```

Each `info` dict has `model_class`, `repo_id`, `revision`, `level`, `state`. `NdifStatus` supports `s[key]`, `key in s`, `len(s)`, `s.keys()`, and iteration.

`status(raw=True)` returns the raw `/status` JSON instead of an `NdifStatus`.

`nnsight.ndif_status()` is a deprecated alias that warns and forwards to `status()`.

## Service status values

`NdifStatus.status` is one of (`src/nnsight/ndif.py:136`):

| Value | Meaning |
|-------|---------|
| `UP` | At least one model's state is `RUNNING`. |
| `REDEPLOYING` | None running, but at least one is `DEPLOYING`. |
| `DOWN` | Nothing running or deploying, or the service was unreachable. |

Per-model `state` values seen in the table: `RUNNING` (live), `DEPLOYING` (coming up; requests queue), `UNHEALTHY` (deployment problem). `level` is `HOT` or `WARM`.

## is_model_running

For a yes/no answer about a single model (`src/nnsight/ndif.py:249`):

```python
if nnsight.is_model_running("meta-llama/Llama-3.1-70B"):
    with model.trace("Hello", remote=True):
        out = model.lm_head.output.save()
else:
    print("Model not currently running on NDIF — a request will queue and warm.")
```

It canonicalizes the repo id via the Hub (`HfApi().model_info(repo_id).id`, handling aliases/redirects), matches on `repo_id` and `revision`, and returns `True` only if `application_state == "RUNNING"`.

Custom revision:

```python
nnsight.is_model_running("meta-llama/Llama-3.1-70B", revision="my-finetune-branch")
```

## Custom HOST

If you've pointed `CONFIG.API.HOST` at a self-hosted or staging deployment, `status()` and `is_model_running()` query that host automatically (both go through `{CONFIG.API.HOST}/status`):

```python
from nnsight import CONFIG
CONFIG.API.HOST = "https://staging.api.ndif.us"

nnsight.status()    # queries staging
```

## Gotchas

- Both fail gracefully on a network error: `status()` prints a DOWN message to stderr and returns an empty `NdifStatus` (whose `.status` is `DOWN`); `is_model_running` returns `False`. So a `False` can mean "not running" *or* "couldn't reach NDIF" — check `status().status` to distinguish.
- `is_model_running` makes a Hub call to canonicalize the repo id; an unauthenticated rate limit can slow a tight loop. Cache the answer.
- `RUNNING` doesn't mean *no* queue — it means the deployment is live. Other users may be ahead of you.

## Related

- [api-key-and-config.md](./api-key-and-config.md) — `CONFIG.API.HOST`.
- [ndif-overview.md](./ndif-overview.md) — what happens when a job is submitted.
- https://discuss.ndif.us/ — outage reports.
