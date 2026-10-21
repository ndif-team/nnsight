---
title: Remote Trace
one_liner: Run a single forward-pass trace on NDIF by passing remote=True; the canonical first remote run and the request lifecycle.
tags: [remote, ndif, trace]
related: [docs/remote/index.md, docs/remote/remote-session.md, docs/remote/non-blocking-jobs.md, docs/usage/trace.md]
sources: [src/nnsight/modeling/mixins/remotable.py, src/nnsight/intervention/backends/remote.py]
---

# Remote Trace

## What this is for

`model.trace(input, remote=True)` runs an intervention on NDIF. It is the same `trace` you call locally; `remote=True` makes the backend serialize the traced block, send it to the server, and wait for the saved values.

## When to use / when not to use

- Use for one-off remote runs, or any model too large to load locally.
- For several traces in a row, use `model.session(remote=True)` to share one queue wait — see [remote-session.md](./remote-session.md).
- To submit and poll later, use `blocking=False` — see [non-blocking-jobs.md](./non-blocking-jobs.md). To await from an event loop, see [remote-async.md](./remote-async.md).
- To check the block serializes and ships before submitting, dry-run it with `remote="local"` — see [ndif-overview.md](./ndif-overview.md#dry-run-the-remote-path-with-remotelocal).

## Canonical pattern

```python
from nnsight import TransformersModel, CONFIG

CONFIG.set_default_api_key("YOUR_KEY")

model = TransformersModel("meta-llama/Llama-3.1-70B")   # builds on the meta device

with model.trace("The Eiffel Tower is in the city of", remote=True):
    logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

print(model.tokenizer.decode(logit))   # ' Paris'
```

## Request lifecycle

The blocking path (`RemoteBackend.request`):

1. `trace(remote=True)` builds `RemoteBackend(model.to_model_key(), ...)`; the block is captured on `__exit__`.
2. Local (non-installed) modules are registered for by-value pickling (`pull_env`), then the block is serialized, optionally compressed (`RequestModel.serialize`).
3. A websocket connects to `{HOST}/subscribe` and receives a session id — subscribing first means no update is missed.
4. The payload is POSTed to `{HOST}/request`; the initial `RECEIVED` response returns over HTTP.
5. The client reads status updates off the websocket until `COMPLETED` or `ERROR` (status values: [ndif-overview.md](./ndif-overview.md#job-status-values)).
6. On `COMPLETED`, the result arrives either inline on the websocket or, depending on its size, as a presigned URL the client downloads. Either way it is decompressed, loaded with `torch.load(..., map_location="cpu")`, and pushed back into your frame, so your saved variables populate — or, with a `RemoteBackend(..., inject_results=False)`, kept on `backend.results` instead (see [below](#keeping-the-saves-on-the-backend)).

## .save() is what comes back

Only values marked with `.save()` (or `nnsight.save(...)`) are returned; everything else stays on the server and is discarded. Your local variable name is matched to the saved value when the result is pushed back.

```python
with model.trace("Hello", remote=True):
    hidden = model.model.layers[5].output                          # not returned
    answer = model.lm_head.output[0][-1].argmax(dim=-1).save()     # returned
```

Saved tensors always land on CPU. Calling `.detach().cpu()` before `.save()` on a large activation still helps: it drops the autograd graph and converts server-side, where the tensor lives.

## Keeping the saves on the backend

A synchronous, blocking run's saves are never returned to you or stored anywhere: the backend runs as the block exits and what it returns is discarded, so injecting them into the block's variables is the only way they reach you. Non-blocking and async runs are different — they return their saves (from the poll or the await) and never inject.

To have a blocking run keep its saves instead, build the backend with `inject_results=False`: the variables stay unfilled and the saves land on `backend.results`, a dict keyed by the saved variable names. This is useful when the code that reads the result is not the code that wrote the block, such as a library that runs a trace on a caller's behalf:

```python
from nnsight.intervention.backends.remote import RemoteBackend

backend = RemoteBackend(model.to_model_key(), inject_results=False)

with model.trace("Hello", backend=backend):
    logits = model.lm_head.output.save()

backend.results["logits"]   # `logits` itself was never filled in
```

The same backend works for `model.session(backend=backend)`. To make it the default for every remote backend — including the one `model.trace(..., remote=True)` builds, read from `tracer.backend.results` — set `CONFIG.APP.INJECT_RESULTS = False`; an explicit `inject_results=` still wins. A blocking run that injects leaves `backend.results` as `None`. A non-blocking run also keeps its saves on `backend.results` once a poll completes; an async run hands them back from `await` / `async for`.

## Print statements appear as LOG status

`print(...)` inside the block runs on the server and is forwarded as a `LOG` update, rendered inline by the status display:

```python
with model.trace("Hello", remote=True):
    print(f"hidden mean: {model.model.layers[0].output.mean()}")   # ℹ [job-id] LOG  hidden mean: ...
    out = model.lm_head.output.save()
```

Set `CONFIG.APP.REMOTE_LOGGING = False` to silence remote logging and the spinner.

## Generation

`model.generate(..., remote=True)` returns **token ids** (prompt plus new tokens); read them off `tracer.result`:

```python
with model.generate("Hello", max_new_tokens=5, do_sample=False, remote=True) as tracer:
    output = tracer.result.save()

print(model.tokenizer.decode(output[0]))
```

For per-step interventions use `tracer.iter[:N]` — see [docs/usage/generate.md](../usage/generate.md).

## Custom host or pre-built backend

Pointing at another server (`remote="https://..."`, `CONFIG.API.HOST`, `NDIF_HOST`) is covered in [api-key-and-config.md](./api-key-and-config.md#pointing-at-a-different-host). To override host, key or verbosity for one call, build the backend yourself:

```python
from nnsight.intervention.backends.remote import RemoteBackend

backend = RemoteBackend(model.to_model_key(), host="https://ndif.example.com", api_key="...")

with model.trace("...", backend=backend):
    out = model.lm_head.output.save()
```

## Gotchas

- **Mutating a variable created outside the block changes only the server's copy.** Create and `.save()` the value inside the block, or call `nnsight.save(outer)` inside it, which rebinds the client's name to the server's version — see [remote-session.md](./remote-session.md#saving-collections-built-across-traces).
- **Tensors from the client arrive as they were locally: CPU, usually float32.** The server's model is half-precision and may be sharded, and an undispatched client model reports `meta` as its device, so nothing can be hard-coded. Derive device and dtype from an activation, per use: `hidden[:, -1] += direction.to(hidden.device, hidden.dtype)`. Tensors built inside the block (`torch.randn(...)`) default to CPU float32 too.
- **A class you ship must call `super(MyClass, self).__init__()`.** Bare `super()` needs a `__class__` cell that only a class body creates; the server rebuilds the class from source outside one and raises `RuntimeError: super(): __class__ cell not found`.
- Local helper modules ship automatically or via `nnsight.register(...)` — see [register-local-modules.md](./register-local-modules.md). Define helpers at module scope; dynamically created functions (lambdas built in loops, `exec`-ed code) can confuse the source-based serializer.
- "Works locally, fails remotely": run `print(nnsight.compare())` — it returns an object, so without `print` you see nothing. See [env-comparison.md](./env-comparison.md).
- The model identifier must match an NDIF deployment; check with `nnsight.is_model_running("...")`.
- A `LOG` line that prints a multi-megabyte tensor floods the websocket. Print summaries.

## Related

- [ndif-overview.md](./ndif-overview.md) — status values, job cost, `remote="local"`.
- [remote-session.md](./remote-session.md) — multiple traces in one queue wait.
- [non-blocking-jobs.md](./non-blocking-jobs.md), [remote-async.md](./remote-async.md) — submit and poll, or await.
- [docs/usage/trace.md](../usage/trace.md) — local tracing.
