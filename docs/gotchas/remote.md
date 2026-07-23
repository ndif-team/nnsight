---
title: Remote Execution Pitfalls
one_liner: NDIF-specific traps — .save() is the transmission channel, local containers don't populate remotely, remote goes on the session, env mismatches, registration.
tags: [gotcha, remote, ndif]
related: [docs/remote/index.md, docs/remote/remote-session.md, docs/gotchas/save.md]
sources: [src/nnsight/intervention/backends/remote.py, src/nnsight/intervention/backends/local.py, src/nnsight/modeling/mixins/remotable.py]
---

# Remote Execution Pitfalls

## TL;DR
- `.save()` is the **only** channel that transmits values back from NDIF. Unsaved values are dropped on the server.
- An external list `.append`-ed inside a remote trace ends up empty — the appends happen server-side. Create the container **inside** the trace with `nnsight.save([])`.
- `.detach().cpu()` before `.save()` to shrink the download.
- Put `remote=True` on `model.session(...)`, **not** on the inner `model.trace(...)` calls — remote goes on the outermost context.
- `print(...)` inside a remote trace comes back as `LOG` status, not local stdout (gated by `CONFIG.APP.REMOTE_LOGGING`).
- Local helper functions/classes must be registered with `nnsight.register(...)` to run on the server.
- **Test offline with `remote="local"`** — it serializes/deserializes the trace (local modules hidden) and runs it locally, a strong dry run for the real remote path.

---

## Test the remote path offline with `remote="local"`

`remote=True` needs a live NDIF server. `remote="local"` (`LocalSimulationBackend`, `src/nnsight/intervention/backends/local.py`) serializes the trace exactly as the remote path would, deserializes it with local (non-installed) modules hidden, then runs it against your real model — so a passing `remote="local"` run is strong evidence the real one will work.

```python
with model.trace("The Eiffel Tower is in", remote="local"):
    hs = model.transformer.h[0].output.detach().cpu().save()
print(hs.shape)   # (1, 7, 768)
```

Model identity for a real remote request comes from `model.to_model_key()`, e.g.
`nnsight.modeling.transformers.TransformersModel:{"repo_id": "openai-community/gpt2", "revision": null}`. Deprecated aliases (`LanguageModel`) share `TransformersModel`'s key.

---

## `.save()` is the transmission channel

### Symptom
A remote request succeeds, but your variable comes back `None`/missing/unchanged. No error.

### Cause
The remote backend serializes only saved variables back to the client. Without `.save()` the value's id is never marked, so it is filtered before shipping.

### Wrong / Right
```python
with model.trace("Hello", remote=True):
    output = model.output.logits            # not saved
# nothing useful comes back

with model.trace("Hello", remote=True):
    output = model.output.logits.detach().cpu().save()
print(output.shape)
```

---

## External list `.append` doesn't populate remotely

### Symptom
Collecting into a list across steps works locally; remotely the list is empty.

### Cause
A list created *outside* the trace lives in the client's process. `.append(...)` inside a remote trace runs on the *server's* copy, which is discarded when the request returns. (This is the one place the local "external list + append saved tensors" pattern from [save.md](save.md) breaks.)

### Wrong / Right
```python
# wrong — client-side list
captured = []
with model.generate("Hello", max_new_tokens=5, remote=True) as tracer:
    with tracer.invoke("Hello"):
        for _ in tracer.iter[:5]:
            captured.append(model.output.logits.argmax(dim=-1))
print(captured)   # []

# right — server-side container, saved
with model.generate(max_new_tokens=5, remote=True) as tracer:
    with tracer.invoke("Hello"):
        captured = nnsight.save([])
        for _ in tracer.iter[:5]:
            captured.append(model.output.logits.argmax(dim=-1))
print(captured)
```

### Mitigation
- Any container you populate during a remote trace must be created inside it and saved.

---

## Move tensors to CPU before save

GPU tensors, full precision, and attached autograd graphs are heavy to transport.
```python
with model.trace("Hello", remote=True):
    hs = model.transformer.h[0].output.detach().cpu().save()
```
Cast to `bfloat16`/`float16` if precision allows.

---

## `remote=True` on the session, not inner traces

### Cause
A session bundles multiple traces into a **single** remote request; `remote=True` on `model.session(...)` opts the whole block in (`src/nnsight/modeling/mixins/remotable.py`). Put it on the inner traces and each becomes a separate request — extra queue waits, and no cross-trace variable flow.

### Right code
```python
with model.session(remote=True):
    with model.trace("A"):
        a = model.transformer.h[5].output        # no .save() needed within the session
    with model.trace("B"):
        model.transformer.h[5].output[:] = a
        result = model.output.logits.save()
```

---

## `print(...)` inside remote traces

`print` output is captured and shipped as `LOG` status messages on the request lifecycle (alongside `RECEIVED`/`QUEUED`/`RUNNING`), not to local stdout. Watch the status stream, or move the value out with `.save()`. Disable with `CONFIG.APP.REMOTE_LOGGING = False`.

---

## Environment mismatch warnings

Remote runs warn when Python/PyTorch/nnsight versions differ from the server's — pickled custom objects are version-sensitive. Match versions where possible; see [docs/remote/env-comparison.md](../remote/env-comparison.md).

---

## Local helpers need registration

A remote run that references a locally-defined function/class fails with `ModuleNotFoundError`/`AttributeError` on the server. Ship the source with `nnsight.register(module_or_callable)` (see [docs/remote/register-local-modules.md](../remote/register-local-modules.md)). Standard libraries (torch, numpy, transformers) are already installed and need no registration.

---

## Blocking, non-blocking, and async

- `RemoteBackend` blocks by default: it holds a websocket open until the job completes and pushes saved values back.
- Non-blocking (`blocking=False`) submits the job and lets you poll — the saved values are not pushed back into your frame automatically (the trace has already exited).
- `AsyncRemoteBackend` (`await backend` → saves dict; `async for update in backend` → status updates then the saves dict last) is the async variant. See [docs/remote/remote-async.md](../remote/remote-async.md).

Config: `CONFIG.API.HOST`, `CONFIG.API.APIKEY`, `CONFIG.API.COMPRESS`; `CONFIG.APP.REMOTE_LOGGING`, `CONFIG.APP.DEBUG`. Environment: `NDIF_API_KEY`, `NDIF_HOST`.

---

## Related
- [docs/remote/index.md](../remote/index.md) — remote execution overview.
- [docs/remote/remote-session.md](../remote/remote-session.md) — sessions over one request.
- [docs/gotchas/save.md](save.md) — `.save()` mechanics underlying all the above.
- [docs/gotchas/iteration.md](iteration.md) — unbounded `iter[:]` drops trailing code (easy to miss in a remote status stream).
