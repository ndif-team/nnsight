---
title: Local vs Remote Environment Comparison
one_liner: Diff your local Python environment against NDIF's to debug "works locally, fails remotely" issues.
tags: [remote, ndif, debugging, environment]
related: [docs/remote/register-local-modules.md, docs/remote/ndif-overview.md]
sources: [src/nnsight/ndif.py:520, src/nnsight/ndif.py:365, src/nnsight/ndif.py:417, src/nnsight/ndif.py:69]
---

# Local vs Remote Environment Comparison

## What this is for

The most common source of remote failures is a Python or package version mismatch. Your trace serializes against your local interpreter; the server unpickles and runs against its own. `nnsight.compare()` prints a side-by-side table of what's installed locally vs on NDIF, with the critical packages highlighted.

## When to use / when not to use

- Use when a remote trace fails with a deserialization error, an `AttributeError` on a transformer module, or a pickling mismatch.
- Use before submitting a long-running job, to catch torch / transformers drift up front.
- Don't use for routine runs — it makes a network call and prints a noisy table.

## Canonical pattern

```python
import nnsight

nnsight.compare()
```

Output (truncated):

```
Python Version:
  Local:  3.11.9
  Remote: 3.11.9 ✓

NDIF Environment Comparison
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Package          ┃ Local Version ┃ Remote Version ┃ Status       ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ nnsight          │ 0.5.1         │ 0.5.0          │ ⚠ CRITICAL  │
│ torch            │ 2.4.0         │ 2.4.0          │ ✓            │
│ transformers     │ 4.45.0        │ 4.45.0         │ ✓            │
│ numpy            │ 1.26.4        │ 1.26.0         │ ≠            │
│ ...              │               │                │              │
└──────────────────┴───────────────┴────────────────┴──────────────┘
```

## Reading the diff

Status legend (`src/nnsight/ndif.py:464`):

| Status | Meaning |
|--------|---------|
| `✓` | Versions match exactly. |
| `≠` | Mismatch on a non-critical package. Usually safe; investigate if you're using a feature that depends on this package. |
| `⚠ CRITICAL` | Mismatch on `nnsight`, `transformers`, or `torch`. These are the packages most likely to break serialization or change model behavior. |

The set of critical packages is hardcoded at `src/nnsight/ndif.py:69`:

```python
CRITICAL_PACKAGES = {"nnsight", "transformers", "torch"}
```

Mismatches on critical packages should be fixed before continuing — either pin your local install to the server's version, or wait for the server to update.

## Underlying functions

If you need the raw data (e.g., for tooling or CI), use the lower-level helpers:

```python
import nnsight

local = nnsight.get_local_env()
remote = nnsight.get_remote_env()      # cached after first call

print(local.keys())     # dict_keys(['python_version', 'packages'])
print(remote['python_version'])
print(local['packages']['torch'])
```

`get_local_env()` (`src/nnsight/ndif.py:365`) walks `importlib.metadata.distributions()` and `pkgutil.iter_modules()` to enumerate everything importable. Modules outside site-packages get the version string `"local"` (these are also auto-registered for cloudpickle by-value serialization — see [register-local-modules.md](./register-local-modules.md)).

`get_remote_env(force_refresh=False)` (`src/nnsight/ndif.py:417`) calls `GET {CONFIG.API.HOST}/env`. The result is cached in a module-level `NDIF_ENV` global; pass `force_refresh=True` to re-fetch:

```python
nnsight.get_remote_env(force_refresh=True)
```

## Common mismatches and fixes

### nnsight version mismatch

```bash
pip install nnsight==<remote-version>
```

The remote version is what's running on the server; pin to it locally. Alternatively, ask in https://discuss.ndif.us/ whether the server can be upgraded.

### torch version mismatch

torch versions affect tensor pickling, CUDA kernel availability, and dtype handling. Pin locally:

```bash
pip install torch==<remote-version>
```

If the remote torch is newer than what's available on your platform, try matching minor versions only.

### transformers version mismatch

Different transformers releases can change module structure (e.g., `model.transformer.h` vs `model.model.layers`). If your trace references modules that don't exist server-side, you'll see `AttributeError: 'X' object has no attribute 'Y'` from the worker.

```bash
pip install transformers==<remote-version>
```

### Local-only packages

If a package shows up in your local list but not in remote (or as version `"local"`), it's a candidate for `nnsight.register(...)`. See [register-local-modules.md](./register-local-modules.md).

## Suppressing color

The table uses rich/ANSI by default. Set `NO_COLOR=1` in the environment to disable, or `FORCE_COLOR=1` to force on (`src/nnsight/ndif.py:442`).

## Gotchas

- `get_remote_env()` caches the response. After the server updates, call `force_refresh=True` to see the new versions.
- The local environment introspection counts the *active* interpreter. Running inside a different venv shows different results.
- Some packages have separate distribution and import names (e.g., `pillow` / `PIL`). The diff uses import names where possible, falling back to distribution names. A package may appear missing when it's only differently named.
- `compare()` only displays packages that exist on the server. Local-only packages don't appear in the table — use `get_local_env()` directly to see them.

## Related

- [register-local-modules.md](./register-local-modules.md) — fix `ModuleNotFoundError` for local-only modules.
- [ndif-overview.md](./ndif-overview.md) — what's executed where.
- https://discuss.ndif.us/ — request server-side updates.
