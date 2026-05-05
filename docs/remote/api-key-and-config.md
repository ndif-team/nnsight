---
title: API Key and Configuration
one_liner: Set the NDIF API key and tune host/logging/debug flags via CONFIG, env vars, or Colab userdata.
tags: [remote, ndif, config]
related: [docs/remote/index.md, docs/remote/remote-trace.md]
sources: [src/nnsight/schema/config.py:1, src/nnsight/config.yaml:1, src/nnsight/intervention/backends/remote.py:491]
---

# API Key and Configuration

## What this is for

Every remote request to NDIF is keyed against an API key. This doc covers the three ways to set the key, the relevant `CONFIG` knobs, and where the on-disk config file lives.

## Get a key

Sign in at https://login.ndif.us/ to register and copy your API key. A free pilot tier is available; check https://nnsight.net for current details.

## Canonical pattern

```python
from nnsight import CONFIG

CONFIG.set_default_api_key("YOUR_API_KEY")
# Persists to <site-packages>/nnsight/config.yaml so you don't have to set it again.
```

`CONFIG.set_default_api_key` writes to `CONFIG.API.APIKEY` and calls `CONFIG.save()` (`src/nnsight/schema/config.py:87`).

## Setting the key — three options

### Option 1: persistent (recommended once per machine)

```python
from nnsight import CONFIG
CONFIG.set_default_api_key("...")    # writes config.yaml on disk
```

### Option 2: environment variable

```bash
export NDIF_API_KEY="..."
```

`CONFIG.from_env()` reads `NDIF_API_KEY` on import (`src/nnsight/schema/config.py:60`). The env var wins over the on-disk value.

`RemoteBackend` also re-reads the env var on every request, with this precedence (`src/nnsight/intervention/backends/remote.py:491`):

```python
self.api_key = (
    api_key                                    # explicit kwarg to RemoteBackend
    or os.environ.get("NDIF_API_KEY", None)    # env var
    or CONFIG.API.APIKEY                       # on-disk config
)
```

### Option 3: Google Colab

In Colab, store the key as a Userdata secret named `NDIF_API_KEY`. `from_env()` reads it via `google.colab.userdata.get("NDIF_API_KEY")` automatically (`src/nnsight/schema/config.py:67`).

## Other CONFIG settings

`CONFIG.API` (`src/nnsight/schema/config.py:9`):

| Field | Default | Purpose |
|-------|---------|---------|
| `HOST` | `"https://api.ndif.us"` | Base URL for all NDIF requests. WebSocket URL is derived (`https://...` -> `wss://...`). Override with the `NDIF_HOST` env var or by assigning directly. |
| `COMPRESS` | `True` | zstd-compress the request body and decompress the response. |
| `APIKEY` | `None` | Set via `set_default_api_key`, env var, or Colab userdata. |

`CONFIG.APP` (`src/nnsight/schema/config.py:15`):

| Field | Default | Purpose |
|-------|---------|---------|
| `REMOTE_LOGGING` | `True` | Show the spinning status display (RECEIVED/QUEUED/RUNNING/...). Set to `False` for silent runs. |
| `DEBUG` | `False` | Verbose tracebacks and `[RemoteBackend]` payload-size prints. Also turns on per-status-line history (verbose mode in `JobStatusDisplay`). |
| `CACHE_DIR` | `"~/.cache/nnsight/"` | Local cache directory (not specific to remote). |

Toggle and persist:

```python
from nnsight import CONFIG

CONFIG.APP.REMOTE_LOGGING = False
CONFIG.APP.DEBUG = True
CONFIG.save()                       # write changes to config.yaml
```

`CONFIG.set_default_app_debug(True)` is a shortcut that sets `DEBUG` and saves (`src/nnsight/schema/config.py:93`).

## Pointing at a different host

NDIF has staging and self-hosted deployments. Two options:

```python
# Persistent
CONFIG.API.HOST = "https://staging.api.ndif.us"
CONFIG.save()

# Per-request
with model.trace("...", backend="https://self-hosted.example.com"):
    ...
```

When `backend` is a string, `RemoteableMixin.trace` builds a `RemoteBackend` against that URL (`src/nnsight/modeling/mixins/remoteable.py:65`). The string must include `http://` or `https://`; `RemoteBackend.__init__` raises `ValueError` otherwise (`src/nnsight/intervention/backends/remote.py:486`).

## Where the config file lives

`CONFIG.save()` writes to `<nnsight-package-dir>/config.yaml`. Today's contents look like:

```yaml
API:
  APIKEY: <your-key>
  COMPRESS: true
  HOST: https://api.ndif.us
APP:
  CACHE_DIR: ~/.cache/nnsight/
  CROSS_INVOKER: true
  DEBUG: false
  PYMOUNT: true
  REMOTE_LOGGING: true
  TRACE_CACHING: false
```

`PATH = os.path.dirname(os.path.abspath(__file__))` in `src/nnsight/__init__.py:44` defines the path; on-disk format is `src/nnsight/config.yaml`.

## Gotchas

- After upgrading nnsight via `pip install -U nnsight` the on-disk `config.yaml` is overwritten with defaults. Re-run `CONFIG.set_default_api_key(...)` once after upgrades.
- `NDIF_HOST` and `NDIF_API_KEY` env vars override the on-disk values, so a stale env var can mask the key you just saved.
- `CONFIG.APP.DEBUG = True` enables verbose status logging *and* full internal tracebacks. Disable for clean stack traces if you're investigating user-code errors.

## Related

- [remote-trace.md](./remote-trace.md) — first remote run after configuring.
- [status-and-availability.md](./status-and-availability.md) — verifies the host/key are working.
- https://login.ndif.us/ — issuing/rotating keys.
