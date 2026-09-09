---
title: Register Local Modules
one_liner: Make local helper modules available inside remote intervention code by shipping their source with the request.
tags: [remote, ndif, serialization]
related: [docs/remote/remote-trace.md, docs/remote/env-comparison.md, docs/remote/ndif-overview.md]
sources: [src/nnsight/ndif.py:31, src/nnsight/ndif.py:318, src/nnsight/intervention/backends/remote.py:249]
---

# Register Local Modules

## What this is for

When you submit a remote trace, the traced block is serialized (source-based) and sent to NDIF. If your block references a class or function from a module that **isn't installed on the NDIF server**, deserialization on the worker raises `ModuleNotFoundError`. `nnsight.register(my_module)` tells the serializer to ship that module's classes and functions *by value* — their source travels in the request payload — so the worker rebuilds them without importing from PyPI.

In practice you rarely call it by hand: the remote backend auto-registers your local modules before the first request (see [Auto-registration](#auto-registration-of-local-modules)).

## When to use / when not to use

- Use for your own local utilities (a `helpers.py` next to your script, an internal package, an experiment file) when auto-registration doesn't pick them up (e.g. an editable install).
- Use after upgrading a server-installed package locally to a not-yet-deployed version, to force your copy to ship.
- Don't use for huge dependencies — every byte of source is sent in every request. Keep registered modules small.

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

nnsight.register(my_utils)        # register BEFORE using anything from my_utils

model = TransformersModel("meta-llama/Llama-3.1-70B")

with model.trace("Hello", remote=True):
    vec = my_utils.steer(model.transformer.h[5].output).save()

print(vec.shape)
```

`nnsight.register` is a thin wrapper over `cloudpickle.register_pickle_by_value` (`src/nnsight/ndif.py:31`). Pass the module object or its name:

```python
nnsight.register(my_utils)        # module object
nnsight.register("my_utils")      # name
```

## Check it works offline with remote="local"

`remote="local"` serializes the block and deserializes it **with your local modules hidden**, mimicking a server that lacks your source files. If a helper wasn't shipped by value, the deserialize step raises exactly as the real server would — so a passing `remote="local"` run confirms your local code ships correctly:

```python
from nnsight import TransformersModel
import my_utils

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("Hello", remote="local"):     # raises if my_utils didn't ship by value
    vec = my_utils.steer(model.transformer.h[5].output).save()
```

## What "by value" means

By default the serializer records a function as `(module_path, name)` and re-imports it on the other side — which fails if the module isn't on the server's `sys.path`. Registering by value instead inlines the function's source, closure cells, and referenced globals into the payload; the server rebuilds it from source (and registers that source in `linecache`, so tracebacks can still show the offending line even though your file isn't present). This is the same mechanism used for the traced block itself — registered modules just opt into it.

## Auto-registration of local modules

Before its first request, `RemoteBackend` calls `pull_env()` (`src/nnsight/ndif.py:318`, invoked from `_serialize`, `src/nnsight/intervention/backends/remote.py:249`). `pull_env` walks the local environment via `get_local_env()`, finds every module marked with the version string `"local"` (importable from your working tree, not a pip install), and registers each one automatically:

```python
def pull_env():
    if _PULLED_ENV:
        return
    for package, version in get_local_env().get("packages", {}).items():
        if version == "local":
            register(package)
    _PULLED_ENV = True
```

So if your helper file is on `sys.path`, it's registered the first time you submit. The remote test suite relies on this: local helper modules and even functions defined in your top-level script ship without any manual `register()` call. `pull_env` is cached per process (`_PULLED_ENV`), so the local-env scan runs only once.

Call `nnsight.register(...)` explicitly when:

- The module is an editable install (`pip install -e .`) — it looks like a normal package, not `"local"`, so auto-registration skips it.
- You want the registration guaranteed before any submission.

## Gotchas

- **Call `register` BEFORE the trace.** The serializer decides how to pickle a function based on the registry at pickle time; registering after the block is captured is too late.
- **Functions that close over file paths or local resources** still won't work server-side — registering ships source, not the filesystem. Refactor to take inputs explicitly.
- **Heavy imports inside a local module** must exist on the server. If `my_utils.py` does `import some_obscure_library`, that library needs to be on NDIF (or registered too).
- Modules outside `site-packages`/`dist-packages` are what `get_local_env` marks `"local"`; a package installed anywhere else won't be auto-detected.

## Related

- [env-comparison.md](./env-comparison.md) — compare local and remote package versions; find what needs registering or pinning.
- [ndif-overview.md](./ndif-overview.md) — where serialized requests go.
- [remote-trace.md](./remote-trace.md) — the `remote=True` invocation that triggers serialization.
