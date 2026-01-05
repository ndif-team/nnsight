# Version Check Optimization for @remote Objects

## Not Yet Implemented

The `@remote(library="...", version="...")` decorator captures version metadata, but the server currently ignores it and always execs the transmitted source code. This document describes how we could skip source transmission when the server already has a compatible version of the library installed.

We're not implementing this yet because the bandwidth savings (~33KB per class) are negligible compared to model weights, and always transmitting source has real benefits: you know exactly what code ran, regardless of what's installed on the server.

## Current Behavior

```python
@nnsight.remote(library="nnterp", version="0.1.0")
class StandardizedTransformer:
    ...
```

This gets serialized with the version metadata included:

```json
{
  "remote_objects": {
    "StandardizedTransformer": {
      "source": {"code": "class StandardizedTransformer:...", "file": "...", "line": 42},
      "library": "nnterp",
      "version": "0.1.0",
      "module_refs": {},
      "closure_vars": {},
      "type": "class",
      "instances": {}
    }
  }
}
```

The server receives `library` and `version` but doesn't use them—it just execs the source. See `deserialize_source_based()` in `serialization_source.py`, around line 2051, where it iterates over `remote_objects` and unconditionally executes each one's source code.

## Design Options

### Option A: Server-Side Lazy Check (No Protocol Change)

The simplest approach. The server already receives version metadata in the payload, so it could check a local registry before exec'ing:

```python
INSTALLED_PACKAGES = {"nnterp": "0.1.5", "einops": "0.6.1"}

def deserialize_source_based(payload, model, ...):
    for obj_name, obj_data in data.get('remote_objects', {}).items():
        lib = obj_data.get('library')
        ver = obj_data.get('version')

        if lib and ver and version_compatible(INSTALLED_PACKAGES.get(lib), ver):
            # Import from installed package instead of exec'ing source
            module = importlib.import_module(lib)
            namespace[obj_name] = getattr(module, obj_name)
        else:
            # Current behavior: exec transmitted source
            exec_func(obj_data['source']['code'], namespace)
```

This doesn't save bandwidth (source is still sent), but avoids exec overhead when there's a match. The server controls what it trusts, and there's a clean fallback if the version doesn't match.

### Option B: Client-Side Preflight Query

Add a server endpoint exposing installed packages:

```
GET /api/packages
{"nnterp": "0.1.5", "einops": "0.6.1", ...}
```

Client queries this before serializing (and caches the result). When a match is found, send an import reference instead of source:

```json
{
  "remote_objects": {
    "StandardizedTransformer": {
      "import": "nnterp.StandardizedTransformer",
      "library": "nnterp",
      "version": "0.1.0"
    }
  }
}
```

This saves bandwidth but requires a new API endpoint and a protocol change to handle the `import` field.

### Option C: Optional Preflight with Server-Side Fallback

Combine Options A and B for defense in depth:

1. **Optional preflight**: Client may query `/api/packages` and cache the result. This is advisory—clients can skip it entirely and the system still works.

2. **Smart serialization**: If the preflight indicates the server has the package, client sends an import reference. Otherwise, it sends full source as usual.

3. **Server-side fallback**: Server checks its registry at deserialize time regardless of what the client sent. If it has a compatible version, it can use its local copy even if source was transmitted. If not, it execs the source.

```
Client                              Server
------                              ------

[Optional: GET /api/packages] ----> [Return installed packages]
<---- {"nnterp": "0.1.5", ...}

[Serialize request]
  - If preflight says server has
    nnterp>=0.1.0, send import ref
  - Otherwise, send full source

POST /request ------------------->  [Deserialize]
                                      For each remote_object:
                                      - import ref + have package: import it
                                      - source + have package: import it (ignore source)
                                      - source + no package: exec source
```

This design means:
- Old clients that don't do preflight still work (they send source, server execs it)
- New clients that do preflight save bandwidth when versions match
- Server always has final say on whether to trust its local copy
- If anything goes wrong, fall back to exec'ing source

## Version Compatibility Strategies

When checking if server version is compatible with client version:

| Strategy | Example | Trade-off |
|----------|---------|-----------|
| Exact match | Client 0.1.0 = Server 0.1.0 | Safest, but rejects compatible updates |
| Semver patch | Client 0.1.0 ~ Server 0.1.5 | Reasonable if libraries follow semver |
| Server newer OK | Client 0.1.0 <= Server 0.1.5 | Common case, minor risk of behavior change |
| Any match | Just check library exists | Fast but risky |

Start with exact match. If that proves too restrictive (e.g., server has 0.1.5 but everyone's clients say 0.1.0), relax to semver-compatible.

## Source Hash Verification

Even with matching version strings, a researcher might have local patches to the library. To handle this, we could hash the source at serialization time:

```json
{
  "library": "nnterp",
  "version": "0.1.0",
  "source_hash": "a1b2c3d4e5f6..."
}
```

Server compares the hash against its installed version. If they don't match, the client has a modified version, so exec the transmitted source instead of importing.

This adds complexity and computational overhead. The version string should be sufficient for most cases—local patches are rare, and researchers who do patch libraries can omit the `version` parameter to force source transmission.

## Why We're Not Implementing This Now

1. **Code is tiny relative to data**: A class definition is ~33KB. Model weights are gigabytes. The optimization saves negligible time.

2. **Reproducibility**: Always transmitting source means the trace is self-contained. You know exactly what code ran, regardless of server state. This matters for scientific reproducibility.

3. **Robustness**: No version compatibility edge cases, no cache invalidation bugs, no "it worked yesterday" mysteries when server packages update.

4. **Simplicity**: The current code path is straightforward. Adding version checks means more branches, more failure modes, more to debug.

If bandwidth does become a bottleneck (many large `@remote` classes, high-volume API usage), implementing Option C would be straightforward. The version metadata is already captured and transmitted; we'd just need to act on it.

## Implementation Notes

If we implement this later, the main changes would be:

**Server-side** (NDIF):
- Add `GET /api/packages` endpoint returning `{package: version}` for installed packages (could auto-detect via `importlib.metadata`)
- Modify `deserialize_source_based()` to check registry before exec
- Handle both `{"source": "..."}` and `{"import": "..."}` formats in `remote_objects`
- Decide on version compatibility logic (exact vs semver)

**Client-side** (nnsight):
- Add optional preflight query to `RemoteBackend`, with caching
- Modify `serialize_source_based()` to emit import references when preflight indicates a match
- Probably add a config option to disable the optimization (always send source) for debugging

**Testing considerations**:
- Version match: verify server imports from installed package
- Version mismatch: verify server execs transmitted source
- Missing preflight: verify server falls back correctly
- Local patches: if we implement hash verification, verify source hash mismatch triggers exec
- Backward compatibility: verify old clients (no preflight, always send source) still work

## Related Documentation

- [Source Serialization Design](./nnsight-source-serialization-design.md) — Main design document covering the serialization format
- [Source Serialization Tutorial](./source-serialization-tutorial.md) — Usage guide for the `@remote` decorator
