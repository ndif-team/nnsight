---
title: NDIF Status and Availability
one_liner: Check the NDIF service state and whether a specific model is currently running before submitting.
tags: [remote, ndif, status]
related: [docs/remote/index.md, docs/remote/api-key-and-config.md]
sources: [src/nnsight/ndif.py]
---

# NDIF Status and Availability

## What this is for

Models on NDIF can be in different deployment states (running, deploying, unhealthy). Submitting to a model that isn't running still works — the request queues — but may take a while to dispatch. These functions tell you what's live so you can fail fast or pick a different revision.

## Canonical pattern

```python
import nnsight

print(nnsight.status())                                    # deployment table
nnsight.is_model_running("meta-llama/Llama-3.1-70B")       # -> True / False
```

```
NDIF Service: Up 🟢

Model Class        Repo ID                   Task             Remote Code  Revision  Level  State      HOT Replicas
-----------------  ------------------------  ---------------  -----------  --------  -----  ---------  ------------
TransformersModel  meta-llama/Llama-3.1-70B  text-generation  -            main      HOT    RUNNING    2
TransformersModel  openai-community/gpt2     text-generation  -            main      WARM   DEPLOYING  0
```

Only **deployed** models appear — level `HOT` or `WARM`; `COLD` (downloaded but not up) is filtered out. Per-model `state` is `RUNNING` (live), `DEPLOYING` (coming up; requests queue) or `UNHEALTHY` (deployment problem).

## status() programmatically

`status()` returns an `NdifStatus`, a dict-like view over the deployments (`status(raw=True)` returns the raw `/status` JSON instead):

```python
s = nnsight.status()

print(s.status)                      # NdifStatus.Status.UP
for repo_id in s:                    # iterates deployment keys
    info = s[repo_id]
    print(repo_id, info["model_class"], info["level"], info["state"])
```

Each `info` dict has `model_class`, `repo_id`, `task`, `trust_remote_code`, `revision`, `level`, `state` and `hot_replicas` (how many HOT replicas serve it). `NdifStatus` supports `s[key]`, `key in s`, `len(s)`, `s.keys()` and iteration. Replicas of one deployment fold into a single entry (healthiest state wins; `hot_replicas` counts the HOT ones). A checkpoint deployed under several pipeline tasks is several deployments, each keyed `"repo_id (task)"`; when the tasks tie, the revision joins the key, and `trust_remote_code` when those tie too.

`NdifStatus.status` is one of:

| Value | Meaning |
|-------|---------|
| `UP` | At least one model is `RUNNING`. |
| `REDEPLOYING` | None running, but at least one is `DEPLOYING`. |
| `DOWN` | Nothing running or deploying, or the service was unreachable. |

`nnsight.ndif_status()` is a deprecated alias that warns and forwards to `status()`.

## is_model_running

```python
if nnsight.is_model_running("meta-llama/Llama-3.1-70B"):
    with model.trace("Hello", remote=True):
        out = model.lm_head.output.save()
else:
    print("Model not currently running on NDIF — a request will queue and warm.")
```

The repo id is canonicalized via the Hugging Face Hub (aliases and redirects match), and `revision` (default `"main"`) and, when given, `task` must match. Custom revision, or one task when the checkpoint is up under several:

```python
nnsight.is_model_running("meta-llama/Llama-3.1-70B", revision="my-finetune-branch")
nnsight.is_model_running("openai-community/gpt2", task="text-generation")
nnsight.is_model_running("x/custom", trust_remote_code=True)   # the key carries it too
```

Both functions query `{CONFIG.API.HOST}/status`, so they follow a custom host — see [api-key-and-config.md](./api-key-and-config.md#pointing-at-a-different-host).

## Gotchas

- Both fail gracefully on a network error: `status()` prints a DOWN message to stderr and returns an empty `NdifStatus` (whose `.status` is `DOWN`); `is_model_running` returns `False`. So a `False` can mean "not running" *or* "couldn't reach NDIF" — check `status().status` to distinguish.
- `is_model_running` makes a Hub call to canonicalize the repo id; an unauthenticated rate limit can slow a tight loop. Cache the answer.
- `RUNNING` doesn't mean *no* queue — it means the deployment is live. Other users may be ahead of you.

## Related

- [api-key-and-config.md](./api-key-and-config.md) — `CONFIG.API.HOST`.
- [ndif-overview.md](./ndif-overview.md) — job status values once a request is submitted.
- https://discuss.ndif.us/ — outage reports.
