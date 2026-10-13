---
title: Register Local Modules
one_liner: Make local helper modules available inside remote intervention code by shipping their source with the request.
tags: [remote, ndif, serialization]
related: [docs/remote/remote-trace.md, docs/remote/env-comparison.md, docs/remote/ndif-overview.md]
sources: [src/nnsight/ndif.py, src/nnsight/intervention/backends/remote.py]
---

# Register Local Modules

## What this is for

The traced block is serialized (source-based) and sent to NDIF. If it references a class or function from a module that **isn't installed on the server**, deserialization raises `ModuleNotFoundError`. `nnsight.register(my_module)` tells the serializer to ship that module's classes and functions *by value* — their source travels in the request payload — so the server rebuilds them without importing.

Usually you don't call it by hand: the remote backend auto-registers your local modules before the first request (see [below](#auto-registration-of-local-modules)).

## When to use / when not to use

- Use for local utilities (a `helpers.py` next to your script, an internal package) that auto-registration doesn't pick up — notably editable installs.
- Use after upgrading a server-installed package locally to a not-yet-deployed version, to force your copy to ship.
- Don't use for huge dependencies — every byte of source is sent in every request.

## Canonical pattern

```python
# my_utils.py — local file, NOT installed on the NDIF server
def steer(hidden):
    return hidden / hidden.norm(dim=-1, keepdim=True)
```

```python
import nnsight
import my_utils
from nnsight import TransformersModel

nnsight.register(my_utils)        # BEFORE the trace; or nnsight.register("my_utils")

model = TransformersModel("meta-llama/Llama-3.1-70B")

with model.trace("Hello", remote=True):
    vec = my_utils.steer(model.model.layers[5].output).save()
```

`nnsight.register` wraps `cloudpickle.register_pickle_by_value`. To confirm a helper ships before submitting, dry-run with `remote="local"`, which hides your local modules during deserialization — see [ndif-overview.md](./ndif-overview.md#dry-run-the-remote-path-with-remotelocal).

## What "by value" means

By default the serializer records a function as `(module_path, name)` and re-imports it on the other side, which fails if the module isn't on the server. By value, the function's source, closure cells and referenced globals are inlined into the payload; the server rebuilds it from source (and registers that source in `linecache`, so tracebacks still show the offending line). The traced block itself ships the same way.

## Auto-registration of local modules

Before its first request, `RemoteBackend` calls `nnsight.ndif.pull_env()`, which registers every module `get_local_env()` marks `"local"` — importable from your working tree rather than from `site-packages`/`dist-packages`. So a helper file on `sys.path`, and functions defined in your top-level script, ship with no manual `register()` call. The scan runs once per process.

Call `nnsight.register(...)` explicitly when:

- The module is an editable install (`pip install -e .`) — it looks like an installed package, not `"local"`, so auto-registration skips it.
- The package lives somewhere other than your working tree or `site-packages`, where it isn't detected.

## Gotchas

- **Register BEFORE the trace.** The serializer decides how to pickle a function at pickle time; registering after the block is captured is too late.
- **Functions that close over file paths or local resources** still won't work server-side — registering ships source, not the filesystem. Pass inputs explicitly.
- **Imports inside a registered module** must exist on the server. If `my_utils.py` imports `some_obscure_library`, that library must be on NDIF (or registered too).

## Related

- [env-comparison.md](./env-comparison.md) — compare local and remote package versions; find what needs registering or pinning.
- [remote-trace.md](./remote-trace.md) — the `remote=True` invocation that triggers serialization.
