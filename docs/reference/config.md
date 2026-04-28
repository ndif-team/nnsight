---
title: Configuration Reference
one_liner: Every CONFIG.APP.* and CONFIG.API.* setting, with type, default, and when to change.
tags: [reference, config]
---

# Configuration Reference

`nnsight.CONFIG` is a `ConfigModel` (Pydantic) singleton constructed at import from `src/nnsight/config.yaml`. It is composed of two sub-models:

- `CONFIG.APP` — local-runtime settings (debug, pymount, cross-invoker, cache).
- `CONFIG.API` — settings used to talk to NDIF (host, key, compression).

## How config is loaded

Source of truth: `src/nnsight/schema/config.py` and `src/nnsight/config.yaml`.

Load order (`ConfigModel.load`):

1. Read defaults from `config.yaml` (next to the package).
2. `from_env()` — `NDIF_API_KEY` env var (or Colab `userdata.NDIF_API_KEY`) overrides `CONFIG.API.APIKEY`; `NDIF_HOST` overrides `CONFIG.API.HOST`.
3. `from_cli()` — if `-d` or `--debug` is in `sys.argv`, force `CONFIG.APP.DEBUG = True`.

`CONFIG.save()` writes the current values back into `config.yaml`. The convenience helpers below also persist:

```python
from nnsight import CONFIG

CONFIG.set_default_api_key("YOUR_NDIF_KEY")  # writes APIKEY + saves
CONFIG.set_default_app_debug(True)            # writes APP.DEBUG + saves
CONFIG.save()                                 # explicit save
```

## `CONFIG.APP.*`

| Name | Type | Default | What it does | When to change |
|------|------|---------|--------------|----------------|
| `APP.DEBUG` | `bool` | `False` | When `True`, exceptions inside a trace include the full nnsight internal stack frames. When `False`, tracebacks are reconstructed to point at user code only. | Turn on while debugging an exception you suspect originates in nnsight internals; otherwise leave off. Also enabled by `python -d`. |
| `APP.REMOTE_LOGGING` | `bool` | `True` | Whether `print(...)` from a remote-trace worker is streamed back as `LOG` events. | Disable to silence remote logs (e.g. for noisy production scripts). |
| `APP.PYMOUNT` | `bool` | `True` | When `True`, the `py_mount.c` C extension injects `.save()` and `.stop()` onto every Python `object` while a trace is active, enabling `tensor.save()` syntax. When `False`, you must use `nnsight.save(obj)` instead. | Disable if (a) you only use `nnsight.save()`, or (b) you have classes whose own `.save()` method is being shadowed/conflicting. |
| `APP.CROSS_INVOKER` | `bool` | `True` | When `True`, variables assigned in one invoke are pushed/pulled across worker threads so later invokes can read them. | Disable to isolate invokes (e.g. while debugging a value-leakage bug) or for a small perf gain when you don't need cross-invoke sharing. |
| `APP.CACHE_DIR` | `str` | `"~/.cache/nnsight/"` | Path for cached artifacts (e.g. exported edits via `Envoy.export_edits`). | Point at a faster disk or a writable location in restricted environments. |
| `APP.TRACE_CACHING` | `bool` | `False` | **Deprecated.** Trace caching (source / AST / code-object caching) is now always on. Setting this to `True` warns and has no effect. | Don't change. Will be removed in a future version. |

## `CONFIG.API.*`

| Name | Type | Default | What it does | When to change |
|------|------|---------|--------------|----------------|
| `API.HOST` | `str` | `"https://api.ndif.us"` | Base URL for NDIF requests (status, env, job submission, results). | Point at an internal NDIF deployment, or override via `NDIF_HOST` env var. |
| `API.APIKEY` | `Optional[str]` | `None` (the shipped `config.yaml` may contain a placeholder) | NDIF API key sent with every remote request. | Set via `CONFIG.set_default_api_key("...")`, the `NDIF_API_KEY` env var, or Colab user data. |
| `API.COMPRESS` | `bool` | `True` | When `True`, request payloads and result downloads use zstandard compression for faster transfers. | Disable only if you suspect a compression-layer issue or are debugging the wire format. |

## Environment-variable shortcuts

| Variable | Effect |
|----------|--------|
| `NDIF_API_KEY` | Overrides `CONFIG.API.APIKEY` at import time. |
| `NDIF_HOST` | Overrides `CONFIG.API.HOST` at import time. |

In Colab, `google.colab.userdata.NDIF_API_KEY` is also picked up automatically.

## CLI shortcuts

| Flag | Effect |
|------|--------|
| `python -d ...` or `python --debug ...` | Forces `CONFIG.APP.DEBUG = True` for the run. |

## Programmatic usage

```python
from nnsight import CONFIG

# Read
print(CONFIG.APP.DEBUG, CONFIG.API.HOST)

# Modify in-process (does NOT persist)
CONFIG.APP.PYMOUNT = False
CONFIG.APP.CROSS_INVOKER = False

# Persist to config.yaml
CONFIG.save()

# Set + persist API key in one call
CONFIG.set_default_api_key("YOUR_NDIF_KEY")
```
