---
title: API Key and Configuration
one_liner: Set the NDIF API key and tune host/logging/debug flags via CONFIG, env vars, or Colab userdata.
tags: [remote, ndif, config]
related: [docs/remote/index.md, docs/remote/remote-trace.md]
sources: [src/nnsight/schema/config.py:12, src/nnsight/schema/config.py:74, src/nnsight/intervention/backends/remote.py:47]
---

# API Key and Configuration

## What this is for

Every remote request to NDIF is keyed against an API key. This doc covers the ways to set the key, the relevant `CONFIG` knobs, and where the on-disk config file lives.

## Get a key

Sign in at https://login.ndif.us/ to register and copy your API key. A free pilot tier is available; check https://nnsight.net for current details.

## Canonical pattern

```python
from nnsight import CONFIG

CONFIG.set_default_api_key("YOUR_API_KEY")
# Persists to ~/.config/nnsight/config.yaml so you don't have to set it again.
```

`CONFIG.set_default_api_key` sets `CONFIG.API.APIKEY` and calls `CONFIG.save()` (`src/nnsight/schema/config.py:109`).

## Setting the key — three options

Config is layered: shipped defaults < user config file < environment (`src/nnsight/schema/config.py:74`).

### Option 1: persistent (recommended once per machine)

```python
from nnsight import CONFIG
CONFIG.set_default_api_key("...")    # writes the user config file
```

Or, HuggingFace-style, use the `login` helper — it prompts for the key with
`getpass` (never echoed), verifies it against the service (best-effort — the key is
still saved, with a note, if the service can't be reached or doesn't recognize it),
and persists it:

```python
from nnsight import login, whoami
login()                              # prompts; or login("YOUR_KEY") to skip the prompt
whoami()                             # {"email": ..., "tags": [...]} for the stored key
```

```bash
nnsight login                        # the same, from a terminal
nnsight whoami
```

### Option 2: environment variable

```bash
export NDIF_API_KEY="..."
```

`Config._from_env()` reads `NDIF_API_KEY` on import and overrides the on-disk value (`src/nnsight/schema/config.py:87`).

`RemoteBackend` resolves the key at construction with this precedence (`src/nnsight/intervention/backends/remote.py:74`):

```python
self.api_key = api_key or CONFIG.API.APIKEY or ""
```

`api_key` is the explicit kwarg to `RemoteBackend`; `CONFIG.API.APIKEY` already reflects the env var (or Colab secret) because `_from_env()` folded it in at import.

### Option 3: Google Colab

In Colab, store the key as a Userdata secret named `NDIF_API_KEY`. When `NDIF_API_KEY` isn't set in the environment and no key is in the config file, `_from_env()` reads the secret via `google.colab.userdata.get("NDIF_API_KEY")` (`src/nnsight/schema/config.py:45`).

## Other settings

Compression, the remote status display and debug output are `CONFIG` fields; every
field, its default and the config-file location are in
[config.md](../reference/config.md). Set one and persist it with `CONFIG.save()`.

## Pointing at a different host

To use a different NDIF deployment (a self-hosted server, say), three options:

```python
# Persistent
CONFIG.API.HOST = "https://ndif.example.com"
CONFIG.save()

# Per-process, via env var
#   export NDIF_HOST="https://ndif.example.com"

# Per-call: pass the host URL as `remote=`
with model.trace("...", remote="https://ndif.example.com"):
    out = model.lm_head.output.save()
```

When `remote` is a string other than `"local"`, `Remotable.trace` treats it as a host URL overriding `CONFIG.API.HOST` for that call (`src/nnsight/modeling/mixins/remotable.py:68`). The URL must start with `http://` or `https://`; `RemoteBackend.__init__` raises `ValueError` otherwise (`src/nnsight/intervention/backends/remote.py:70`).

## Gotchas

- `NDIF_HOST` and `NDIF_API_KEY` env vars override the on-disk values, so a stale env var can mask the key you just saved.
- `CONFIG.APP.DEBUG = True` prints payload/result sizes and a per-status timeline every run. Turn it off for clean output.
- `nnsight.ndif_status()` is deprecated in favor of `nnsight.status()`.

## Related

- [remote-trace.md](./remote-trace.md) — first remote run after configuring.
- [status-and-availability.md](./status-and-availability.md) — verify the host/key are working.
- https://login.ndif.us/ — issuing/rotating keys.
