---
title: Configuration Reference
one_liner: Every CONFIG.API.* and CONFIG.APP.* setting, with type, default, and when to change.
tags: [reference, config]
---

# Configuration Reference

`nnsight.CONFIG` is a Pydantic `Config` singleton, built at import from `src/nnsight/schema/config.py`. Two sub-models:

- `CONFIG.API` — settings for talking to NDIF (host, key, compression).
- `CONFIG.APP` — local-runtime settings (debug, remote logging, `.save()` mount).

```python
from nnsight import CONFIG
print(CONFIG)
# API=ApiConfig(HOST='https://api.ndif.us', APIKEY=None, COMPRESS=True)
# APP=AppConfig(DEBUG=False, REMOTE_LOGGING=True, PYMOUNT=True, DISABLE_CPP_BACKTRACE=True)
```

## How config is loaded

Source of truth: `src/nnsight/schema/config.py` and the shipped `src/nnsight/config.yaml`.

Load order (`Config.load`), later wins:

1. **Shipped defaults** from the package `config.yaml`. (Don't edit it — it's overwritten on upgrade.)
2. **User file** — `~/.config/nnsight/config.yaml` (or `$XDG_CONFIG_HOME/nnsight/config.yaml`, or the path in `$NNSIGHT_CONFIG`), merged over the defaults.
3. **Environment** — `NDIF_API_KEY`, `NDIF_HOST`, `NNSIGHT_DEBUG`, `NNSIGHT_DISABLE_CPP_BACKTRACE` (see below). In Colab, a `NDIF_API_KEY` notebook secret is a fallback when no env var and no file key is set.

`CONFIG.save()` writes the current values to the **user** file (creating it if needed) — never to the shipped one.

```python
from nnsight import CONFIG

CONFIG.set_default_api_key("YOUR_NDIF_KEY")  # sets APIKEY + saves to user file
CONFIG.save()                                 # explicit save
```

## `CONFIG.API.*`

| Name | Type | Default | What it does | When to change |
|------|------|---------|--------------|----------------|
| `API.HOST` | `str` | `"https://api.ndif.us"` | Base URL for NDIF requests (status, env, job submission, results). | Point at an internal NDIF deployment; or override per call with `remote="<host url>"`, or globally via `NDIF_HOST`. |
| `API.APIKEY` | `Optional[str]` | `None` | NDIF API key sent with every remote request. | Set via `CONFIG.set_default_api_key("...")`, the `NDIF_API_KEY` env var, or a Colab secret. |
| `API.COMPRESS` | `bool` | `True` | zstandard-compress request payloads and result downloads for faster transfers. | Disable only when debugging the wire format / a compression issue. |

## `CONFIG.APP.*`

| Name | Type | Default | What it does | When to change |
|------|------|---------|--------------|----------------|
| `APP.DEBUG` | `bool` | `False` | When `True`, tracebacks keep nnsight's internal frames; when `False`, they are filtered to your own code. | Turn on to debug an exception you suspect is in nnsight internals. Also enabled by `NNSIGHT_DEBUG`. |
| `APP.REMOTE_LOGGING` | `bool` | `True` | Whether `print(...)` from a remote run is streamed back as log events. | Disable to silence remote logs. |
| `APP.PYMOUNT` | `bool` | `True` | Mount `.save()` onto every Python object (via the optional C extension) so `value.save()` works in a trace. When `False` (or the extension didn't build), use `nnsight.save(value)`. Mounting adds `.save` to all objects process-wide, so anything checking `hasattr(x, "save")` will see it. | Disable if you only use `nnsight.save()`, or a class's own `.save()` conflicts. |
| `APP.DISABLE_CPP_BACKTRACE` | `bool` | `True` | At import, neutralize glibc `backtrace()` so a torch C++ error raised inside an interleaving greenlet surfaces as a normal Python exception instead of segfaulting the process. Empties torch's (rarely used) *C++* backtrace string process-wide; Python tracebacks and error messages are unaffected. Only glibc/x86-64 Linux is touched. | Turn off (via `NNSIGHT_DISABLE_CPP_BACKTRACE=0`) only if you need torch's C++ stack traces and accept the crash risk. |


> **Why this exists.** nnsight runs each invoke's intervention code in a greenlet, which time-shares one OS stack by copying stack slices in and out. When a torch op raises a `c10` error while a worker greenlet is running, torch's error constructor captures a C++ backtrace with glibc `backtrace()`, which walks off the greenlet's stack slice into stale memory and crashes libgcc's unwinder — turning a plain shape error into a hard segfault. Emptying that one call removes the crash without changing any Python-visible behavior.

## Environment variables

| Variable | Effect |
|----------|--------|
| `NDIF_API_KEY` | Sets `CONFIG.API.APIKEY` at import. |
| `NDIF_HOST` | Sets `CONFIG.API.HOST` at import. |
| `NNSIGHT_DEBUG` | If set (any value), forces `CONFIG.APP.DEBUG = True`. |
| `NNSIGHT_DISABLE_CPP_BACKTRACE` | Overrides `CONFIG.APP.DISABLE_CPP_BACKTRACE`. Falsy values (`0`, `false`, `no`, `off`) turn the guard **off**; anything else keeps it on (the default). |
| `NNSIGHT_CONFIG` | Path to the user config file (overrides the `~/.config/nnsight/config.yaml` default). |
| `XDG_CONFIG_HOME` | Base dir for the default user config path. |

Pass `-v` (or `--verbose`) on the command line — e.g. `python train.py -v` — to turn on debug mode for that run (equivalent to `NNSIGHT_DEBUG=1`). Note it's a plain `sys.argv` scan, so any launcher that also uses `-v` (e.g. `pytest -v`) will enable it too.

## Programmatic usage

```python
from nnsight import CONFIG

# Read
print(CONFIG.APP.DEBUG, CONFIG.API.HOST)

# Modify in-process (does NOT persist)
CONFIG.APP.PYMOUNT = False

# Persist to the user config file
CONFIG.save()

# Set + persist API key in one call
CONFIG.set_default_api_key("YOUR_NDIF_KEY")
```
