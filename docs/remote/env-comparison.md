---
title: Local vs Remote Environment Comparison
one_liner: Diff your local Python environment against NDIF's to debug "works locally, fails remotely" issues.
tags: [remote, ndif, debugging, environment]
related: [docs/remote/register-local-modules.md, docs/remote/ndif-overview.md]
sources: [src/nnsight/ndif.py]
---

# Local vs Remote Environment Comparison

## What this is for

The most common source of remote failures is a Python or package version mismatch: your block serializes against your local interpreter, and the server unpickles and runs it against its own. `nnsight.compare()` diffs what's installed locally against NDIF, highlighting the critical packages.

## When to use / when not to use

- Use when a remote trace fails with a deserialization error, an `AttributeError` on a model module, or a pickling mismatch.
- Use before a long-running job to catch torch/transformers drift up front.
- Don't use for routine runs — it makes a network call.

## Canonical pattern

```python
import nnsight

print(nnsight.compare())    # returns an EnvComparison; without print you see nothing
```

Output shape (versions illustrative):

```
Python Version:
  Local:  3.X.Y
  Remote: 3.X.Y ✓

Package       Local Version  Remote Version  Status
------------  -------------  --------------  ----------
nnsight       <local>        <remote>        ⚠ CRITICAL
torch         <same>         <same>          ✓
transformers  <same>         <same>          ✓
numpy         <local>        <remote>        ≠
```

| Status | Meaning |
|--------|---------|
| `✓` | Versions match exactly. |
| `≠` | Mismatch on a non-critical package. Usually safe; investigate if you depend on that package. |
| `⚠ CRITICAL` | Mismatch on `nnsight`, `transformers` or `torch` (`CRITICAL_PACKAGES`) — the packages most likely to break serialization or change model behavior. |

Fix critical mismatches before continuing — pin your local install to the server's version, or wait for the server to update.

## Comparing against a specific server

`compare()` reads the configured host (`CONFIG.API.HOST`). A job sent with a per-call host runs on that server's packages, so compare against the same host:

```python
HOST = "http://localhost:5001"

print(nnsight.compare(HOST))

with model.trace("The Eiffel Tower is in", remote=HOST):
    ...
```

Each server's environment is fetched once per process and cached under its URL; `"http://localhost:5001/"` is the same server.

## Inspecting the result programmatically

`EnvComparison` exposes structured fields:

```python
cmp = nnsight.compare()

cmp.python_matches            # bool
cmp.mismatches                # {pkg: {"local", "remote", "match", "critical"}} — differing only
cmp.critical_mismatches       # the critical subset of mismatches
cmp.packages["torch"]         # {'local': ..., 'remote': ..., 'match': ..., 'critical': ...}
cmp.local, cmp.remote         # the raw env dicts
```

`packages` (and the printed table) is keyed by the packages the **server** has: a local-only package can't cause a remote mismatch, so it doesn't appear. Use `get_local_env()` to see those.

## Raw environment helpers

For the underlying data (e.g. for CI):

```python
from nnsight import ndif

local = ndif.get_local_env()          # {'python_version': ..., 'packages': {...}}
remote = ndif.get_remote_env()        # GET {CONFIG.API.HOST}/env, cached per host
remote = ndif.get_remote_env("http://localhost:5001")   # another server
remote = ndif.get_remote_env(force_refresh=True)   # re-fetch after the server updates
```

`get_local_env()` enumerates installed distributions (by import name) plus modules importable from your working tree; modules outside `site-packages`/`dist-packages` get the version string `"local"` and are auto-registered for by-value serialization — see [register-local-modules.md](./register-local-modules.md).

## Common mismatches and fixes

- **nnsight / torch / transformers**: `pip install <pkg>==<remote-version>`. torch versions affect tensor pickling and dtype handling; transformers releases change module structure, so a block that references modules absent server-side fails with an `AttributeError` from the worker. Pin to the server's version, or ask in https://discuss.ndif.us/ whether the server can be upgraded.
- **Local-only packages** (version `"local"`, or not on the server): candidates for `nnsight.register(...)` — see [register-local-modules.md](./register-local-modules.md).

## Gotchas

- An editable install of a package shows a version mismatch even when the code is identical, because only the `.dist-info` metadata differs.
- Introspection reflects the *active* interpreter; a different venv shows different results.
- Some packages have separate distribution and import names (e.g. `pillow`/`PIL`); the diff keys by import name where possible, so a package can look missing when it's only differently named.

## Related

- [register-local-modules.md](./register-local-modules.md) — fix `ModuleNotFoundError` for local-only modules.
- [ndif-overview.md](./ndif-overview.md) — what's executed where.
- https://discuss.ndif.us/ — request server-side updates.
