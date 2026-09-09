---
title: Local vs Remote Environment Comparison
one_liner: Diff your local Python environment against NDIF's to debug "works locally, fails remotely" issues.
tags: [remote, ndif, debugging, environment]
related: [docs/remote/register-local-modules.md, docs/remote/ndif-overview.md]
sources: [src/nnsight/ndif.py:424, src/nnsight/ndif.py:364, src/nnsight/ndif.py:275, src/nnsight/ndif.py:307]
---

# Local vs Remote Environment Comparison

## What this is for

The most common source of remote failures is a Python or package version mismatch: your block serializes against your local interpreter, and the server unpickles and runs against its own. `nnsight.compare()` diffs what's installed locally vs on NDIF, highlighting the critical packages.

## When to use / when not to use

- Use when a remote trace fails with a deserialization error, an `AttributeError` on a transformer module, or a pickling mismatch.
- Use before a long-running job to catch torch/transformers drift up front.
- Don't use for routine runs — it makes a network call.

## Canonical pattern

```python
import nnsight

print(nnsight.compare())
```

Output (truncated):

```
Python Version:
  Local:  3.11.9
  Remote: 3.11.9 ✓

Package       Local Version  Remote Version  Status
------------  -------------  --------------  ----------
nnsight       0.5.1          0.5.0           ⚠ CRITICAL
torch         2.4.0          2.4.0           ✓
transformers  4.45.0         4.45.0          ✓
numpy         1.26.4         1.26.0          ≠
```

`compare()` returns an `EnvComparison` object (`src/nnsight/ndif.py:364`) — `print` it for the table, or inspect it programmatically.

## Reading the diff

Status legend:

| Status | Meaning |
|--------|---------|
| `✓` | Versions match exactly. |
| `≠` | Mismatch on a non-critical package. Usually safe; investigate if you depend on that package. |
| `⚠ CRITICAL` | Mismatch on `nnsight`, `transformers`, or `torch` — the packages most likely to break serialization or change model behavior. |

The critical set is hardcoded (`src/nnsight/ndif.py:25`):

```python
CRITICAL_PACKAGES = {"nnsight", "transformers", "torch"}
```

Fix critical mismatches before continuing — pin your local install to the server's version, or wait for the server to update.

## Inspecting the result programmatically

`EnvComparison` exposes structured fields (no parsing the table):

```python
import nnsight

cmp = nnsight.compare()

cmp.python_matches            # bool
cmp.mismatches                # {pkg: {"local", "remote", "match", "critical"}} — differing only
cmp.critical_mismatches       # the critical subset of mismatches
cmp.packages["torch"]         # {'local': ..., 'remote': ..., 'match': ..., 'critical': ...}
cmp.local, cmp.remote         # the raw env dicts
```

`packages` is keyed by the packages the **server** has (a local-only package can't cause a remote mismatch, so it isn't a key here).

## Raw environment helpers

For the underlying data (e.g. for CI):

```python
from nnsight import ndif

local = ndif.get_local_env()
remote = ndif.get_remote_env()        # cached after first call

print(local.keys())                   # dict_keys(['python_version', 'packages'])
print(remote["python_version"])
print(local["packages"]["torch"])
```

`get_local_env()` (`src/nnsight/ndif.py:275`) enumerates installed distributions (by import name) plus modules importable from your working tree; modules outside `site-packages`/`dist-packages` get the version string `"local"` (these are also auto-registered for by-value serialization — see [register-local-modules.md](./register-local-modules.md)).

`get_remote_env(force_refresh=False)` (`src/nnsight/ndif.py:307`) calls `GET {CONFIG.API.HOST}/env` and caches the result in a module global; pass `force_refresh=True` to re-fetch:

```python
ndif.get_remote_env(force_refresh=True)
```

## Common mismatches and fixes

### nnsight / torch / transformers

```bash
pip install nnsight==<remote-version>
pip install torch==<remote-version>
pip install transformers==<remote-version>
```

torch versions affect tensor pickling and dtype handling; transformers releases change module structure (e.g. `transformer.h` vs `model.layers`), so a block that references modules absent server-side fails with an `AttributeError` from the worker. Pin to the server's version, or ask in https://discuss.ndif.us/ whether the server can be upgraded.

### Local-only packages

A package that's local (or version `"local"`) and not on the server is a candidate for `nnsight.register(...)`. See [register-local-modules.md](./register-local-modules.md).

## Suppressing color

Tables use ANSI color when writing to a TTY. Set `NO_COLOR=1` to disable or `FORCE_COLOR=1` to force it on (`src/nnsight/ndif.py:70`).

## Gotchas

- `get_remote_env()` caches its response. After the server updates, call `force_refresh=True` to see new versions.
- Introspection reflects the *active* interpreter; a different venv shows different results.
- Some packages have separate distribution and import names (e.g. `pillow`/`PIL`); the diff keys by import name where possible, so a package can look missing when it's only differently named.
- The table only shows packages the server has; local-only packages don't appear — use `get_local_env()` to see them.

## Related

- [register-local-modules.md](./register-local-modules.md) — fix `ModuleNotFoundError` for local-only modules.
- [ndif-overview.md](./ndif-overview.md) — what's executed where.
- https://discuss.ndif.us/ — request server-side updates.
