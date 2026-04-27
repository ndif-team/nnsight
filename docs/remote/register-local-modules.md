---
title: Register Local Modules
one_liner: Make local helper modules available inside remote intervention code by registering them for source-based serialization.
tags: [remote, ndif, serialization]
related: [docs/remote/remote-trace.md, docs/remote/env-comparison.md, docs/remote/ndif-overview.md]
sources: [src/nnsight/ndif.py:22, src/nnsight/intervention/serialization.py:682, src/nnsight/intervention/backends/remote.py:387]
---

# Register Local Modules

## What this is for

When you submit a remote trace, the intervention code is serialized via cloudpickle and sent to NDIF. If your trace references a class or function from a module that **isn't installed on the NDIF server**, deserialization on the worker raises `ModuleNotFoundError`. `nnsight.register(my_module)` tells cloudpickle to serialize that module's contents *by value* — its source code travels in the request payload — so the worker can rebuild it without ever importing it from PyPI.

## When to use / when not to use

- Use for your own local utilities (a `helpers.py` next to your script, an internal package, an experiment file).
- Use after upgrading a server-installed package locally to a not-yet-deployed version (otherwise the worker will use its older copy).
- Don't use for huge dependencies — every byte of source is sent in every request. Keep registered modules small.
- Don't use for top-level scripts (the main script is already handled). Use it for *imported* modules.

## Canonical pattern

```python
# my_utils.py — local file, NOT installed on the NDIF server
def my_steering_vector(model, layer):
    return model.transformer.h[layer].output[0].mean(dim=1)
```

```python
import nnsight
import my_utils
from nnsight import LanguageModel

nnsight.register(my_utils)        # register BEFORE using anything from my_utils

model = LanguageModel("meta-llama/Llama-3.1-70B")

with model.trace("Hello", remote=True):
    vec = my_utils.my_steering_vector(model, 5).save()

print(vec.shape)
```

`nnsight.register` is a thin wrapper around `cloudpickle.register_pickle_by_value` (`src/nnsight/ndif.py:22`):

```python
def register(module: types.ModuleType | str):
    if isinstance(module, str):
        _PICKLE_BY_VALUE_MODULES.add(module)
    else:
        register_pickle_by_value(module)
```

You can pass either the module object or the string name:

```python
nnsight.register(my_utils)        # module object
nnsight.register("my_utils")      # name
```

## What "by value" actually means

By default, cloudpickle records a function as `(module_path, name)` and re-imports it on the other side. That fails if the module isn't on `sys.path` server-side.

With `register_pickle_by_value`, cloudpickle inlines the function's source code, closure cells, and globals into the pickle stream. On the server side, NNsight's `CustomCloudPickler` / `CustomCloudUnpickler` (`src/nnsight/intervention/serialization.py:682`) rebuild the function from source and patch it back into a synthetic module. This is also what's used for the trace body itself — registered local modules just opt into the same mechanism.

## Auto-registration of local modules

`RemoteBackend.request` calls `pull_env()` (`src/nnsight/intervention/backends/remote.py:387`) before each submission. `pull_env()` walks the local environment, finds modules whose distribution version is the literal string `"local"` (i.e., not installed via pip — discovered as importable directories on `sys.path`), and registers them automatically:

```python
def pull_env():
    if not _PULLED_ENV:
        local_env = get_local_env()
        for package, version in local_env.get("packages", {}).items():
            if version == "local":
                register(package)
        _PULLED_ENV = True
```

`get_local_env()` (`src/nnsight/ndif.py:365`) marks any non-stdlib, non-site-packages module as `"local"`. So if your helper file is on `sys.path`, it should be auto-registered the first time you submit a remote request.

You should still call `nnsight.register(...)` explicitly when:

- The module is installed via `pip install -e .` (it shows up as a normal package, not `"local"`).
- You want to be sure of the registration before any submission.
- The auto-registration heuristic missed it (rare but possible).

## Registering a sub-package

`register_pickle_by_value` only handles the module you pass; submodules aren't picked up automatically. For a package like `myproj/`, register each submodule you actually use:

```python
nnsight.register("myproj.utils")
nnsight.register("myproj.steering")
```

Or register the whole top-level package and import submodules through it after registration.

## Transitive imports

If `my_utils.py` does `from helpers import something` where `helpers` is also a local module, **`pull_env()`'s auto-registration covers it** — anything detected as `version="local"` in the environment gets registered. If auto-registration fails or you've disabled it, you need to register both `my_utils` and `helpers` explicitly.

## Gotchas

- **Call `register` BEFORE the trace.** Cloudpickle decides how to serialize a function based on the registry at pickle time. Registering after the trace body has been captured is too late.
- **Functions that close over file paths or local resources** still won't work on the server — registering ships the source, not the filesystem. Refactor to take inputs explicitly.
- **Heavy dependencies in your local module's imports** will be required server-side. If `my_utils.py` does `import some_obscure_library`, that library needs to exist on NDIF (or be registered too).
- **Editable installs (`pip install -e .`)** are detected as installed packages, not `"local"`. Auto-registration won't pick them up; register manually.
- **Circular registrations** (`A imports B`, register both) generally work but produce larger payloads than necessary. Register only what you actually use.

## Related

- [env-comparison.md](./env-comparison.md) — compare local and remote package versions; identify what needs to be registered or pinned.
- [ndif-overview.md](./ndif-overview.md) — where serialized requests go.
- [remote-trace.md](./remote-trace.md) — the `remote=True` invocation that triggers serialization.
