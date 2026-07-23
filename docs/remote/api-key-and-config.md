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

## CONFIG settings

`CONFIG.API` (`src/nnsight/schema/config.py:12`):

| Field | Default | Purpose |
|-------|---------|---------|
| `HOST` | `"https://api.ndif.us"` | Base URL for all NDIF requests. The websocket URL is derived (`https://…` → `wss://…`). Override with the `NDIF_HOST` env var or by assigning directly. |
| `APIKEY` | `None` | Set via `set_default_api_key`, `NDIF_API_KEY`, or a Colab secret. |
| `COMPRESS` | `True` | zstd-compress the request payload and decompress the result. |

`CONFIG.APP` (`src/nnsight/schema/config.py:18`):

| Field | Default | Purpose |
|-------|---------|---------|
| `DEBUG` | `False` | Verbose remote diagnostics: `[remote]` payload/result byte-size prints, plus each status printed on its own line (a persisted timeline) instead of one in-place spinner. Also settable via the `NNSIGHT_DEBUG` env var. |
| `REMOTE_LOGGING` | `True` | Show the status display (spinner + RECEIVED/QUEUED/RUNNING/…) and the download progress bar. Set `False` for silent runs. |
| `PYMOUNT` | `True` | Mount `.save()` onto every object so `value.save()` works in a trace. If `False` (or the optional C extension didn't build), use `nnsight.save(value)` instead. |

Toggle and persist:

```python
from nnsight import CONFIG

CONFIG.APP.REMOTE_LOGGING = False
CONFIG.APP.DEBUG = True
CONFIG.save()                       # write changes to the user config file
```

## Pointing at a different host

NDIF has staging and self-hosted deployments. Three options:

```python
# Persistent
CONFIG.API.HOST = "https://staging.api.ndif.us"
CONFIG.save()

# Per-process, via env var
#   export NDIF_HOST="https://staging.api.ndif.us"

# Per-call: pass the host URL as `remote=`
with model.trace("...", remote="https://self-hosted.example.com"):
    out = model.lm_head.output.save()
```

When `remote` is a string other than `"local"`, `Remotable.trace` treats it as a host URL overriding `CONFIG.API.HOST` for that call (`src/nnsight/modeling/mixins/remotable.py:68`). The URL must start with `http://` or `https://`; `RemoteBackend.__init__` raises `ValueError` otherwise (`src/nnsight/intervention/backends/remote.py:70`).

## Where the config file lives

`CONFIG.save()` writes to the user config file, resolved by `_user_config_path()` (`src/nnsight/schema/config.py:37`):

1. `$NNSIGHT_CONFIG` if set (a full path), else
2. `$XDG_CONFIG_HOME/nnsight/config.yaml` (default `~/.config/nnsight/config.yaml`).

Contents look like:

```yaml
API:
  APIKEY: <your-key>
  COMPRESS: true
  HOST: https://api.ndif.us
APP:
  DEBUG: false
  PYMOUNT: true
  REMOTE_LOGGING: true
```

The shipped defaults live at `<nnsight-package-dir>/config.yaml` and are merged under your user file, so upgrading nnsight no longer clobbers your saved key — it's in `~/.config`, separate from the package.

## Gotchas

- `NDIF_HOST` and `NDIF_API_KEY` env vars override the on-disk values, so a stale env var can mask the key you just saved.
- `CONFIG.APP.DEBUG = True` prints payload/result sizes and a per-status timeline every run. Turn it off for clean output.
- The legacy `CROSS_INVOKER`, `CACHE_DIR`, and `TRACE_CACHING` config fields are gone. `nnsight.ndif_status()` still exists but is deprecated in favor of `nnsight.status()`.

## Related

- [remote-trace.md](./remote-trace.md) — first remote run after configuring.
- [status-and-availability.md](./status-and-availability.md) — verify the host/key are working.
- https://login.ndif.us/ — issuing/rotating keys.
