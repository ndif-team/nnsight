---
title: API Key and Configuration
one_liner: Set the NDIF API key and tune host/logging/debug flags via CONFIG, env vars, or Colab userdata.
tags: [remote, ndif, config]
related: [docs/remote/index.md, docs/remote/remote-trace.md]
sources: [src/nnsight/schema/config.py, src/nnsight/intervention/backends/remote.py]
---

# API Key and Configuration

## What this is for

Every remote request to NDIF is keyed against an API key. This page covers the ways to set the key and point nnsight at a different host.

## Get a key

Sign in at https://login.ndif.us/ to register and copy your API key. A free pilot tier is available; check https://nnsight.net for current details.

## Setting the key

Config is layered: shipped defaults < user config file < environment (`Config` in `src/nnsight/schema/config.py`).

### Persistent (once per machine)

```python
from nnsight import CONFIG

CONFIG.set_default_api_key("YOUR_API_KEY")   # sets CONFIG.API.APIKEY and saves the user config file
```

Or, HuggingFace-style, use `login` — it prompts for the key with `getpass` (never echoed), verifies it against the service (best-effort: the key is still saved, with a note, if the service can't be reached or doesn't recognize it), and persists it:

```python
from nnsight import login, whoami
login()                              # prompts; or login("YOUR_KEY") to skip the prompt
whoami()                             # {"email": ..., "tags": [...]} for the stored key
```

```bash
nnsight login                        # the same, from a terminal
nnsight whoami
```

### Environment variable

```bash
export NDIF_API_KEY="..."
```

Read on import; it overrides the on-disk value. An `api_key=` passed to `RemoteBackend` overrides both.

### Google Colab

Store the key as a Userdata secret named `NDIF_API_KEY`. When the env var isn't set and no key is in the config file, nnsight reads the secret via `google.colab.userdata.get("NDIF_API_KEY")`.

## Other settings

Compression, the remote status display and debug output are `CONFIG` fields; every field, its default and the config-file location are in [config.md](../reference/config.md). Set one and persist it with `CONFIG.save()`.

## Pointing at a different host

To use a different NDIF deployment (a self-hosted server, say):

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

A `remote` string other than `"local"` is a host URL overriding `CONFIG.API.HOST` for that call. It must start with `http://` or `https://`, or `RemoteBackend` raises `ValueError`.

## Gotchas

- `NDIF_HOST` and `NDIF_API_KEY` env vars override the on-disk values, so a stale env var can mask the key you just saved.
- `CONFIG.APP.DEBUG = True` prints payload/result sizes and a per-status timeline every run. Turn it off for clean output.

## Related

- [remote-trace.md](./remote-trace.md) — first remote run after configuring.
- [status-and-availability.md](./status-and-availability.md) — verify the host and key are working.
- https://login.ndif.us/ — issuing and rotating keys.
