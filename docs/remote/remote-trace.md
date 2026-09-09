---
title: Remote Trace
one_liner: Run a single forward-pass trace on NDIF by passing remote=True.
tags: [remote, ndif, trace]
related: [docs/remote/index.md, docs/remote/remote-session.md, docs/remote/non-blocking-jobs.md, docs/usage/trace.md]
sources: [src/nnsight/modeling/mixins/remotable.py:19, src/nnsight/intervention/backends/remote.py:39, src/nnsight/intervention/backends/remote.py:304]
---

# Remote Trace

## What this is for

`model.trace(input, remote=True)` is the simplest way to execute an intervention on NDIF. It's the same `trace` you call locally — you just add `remote=True`, and the backend serializes the traced block, ships it over HTTP, and waits on a websocket for completion.

## When to use / when not to use

- Use for one-off remote runs.
- Use when working with a model too large to load locally.
- For multiple traces in a row, prefer `model.session(remote=True)` to share one queue wait — see [remote-session.md](./remote-session.md).
- For long-running jobs you don't want to block on, use `blocking=False` — see [non-blocking-jobs.md](./non-blocking-jobs.md).
- To await a job from an event loop, use `AsyncRemoteBackend` — see [remote-async.md](./remote-async.md).

## Canonical pattern

```python
from nnsight import TransformersModel, CONFIG

CONFIG.set_default_api_key("YOUR_KEY")

model = TransformersModel("meta-llama/Llama-3.1-70B")   # builds on meta device

with model.trace("The Eiffel Tower is in the city of", remote=True):
    logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

print(model.tokenizer.decode(logit))   # ' Paris'
```

What happens under the hood (`src/nnsight/modeling/mixins/remotable.py:31` and `src/nnsight/intervention/backends/remote.py:304`):

1. `trace(remote=True)` builds `RemoteBackend(self.to_model_key(), env=…, blocking=True)`.
2. On `__exit__`, the traced block is serialized to (optionally compressed) bytes.
3. A websocket connects to `{HOST}/subscribe` and receives a session id.
4. The payload is POSTed to `{HOST}/request`; the initial `RECEIVED` response returns over HTTP.
5. The client reads status updates off the websocket until `COMPLETED` or `ERROR`.
6. On `COMPLETED`, the result is downloaded from a presigned URL, decompressed, `torch.load`ed, and pushed back into your frame.

## Try it offline first with remote="local"

`remote="local"` runs the exact serialize/deserialize/execute path in-process against your local (dispatched) model — no server, no key. It's the fastest way to catch a serialization or missing-module problem before submitting for real:

```python
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("The Eiffel Tower is in the city of", remote="local"):
    logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

print(model.tokenizer.decode(logit))   # ' Paris'
```

## .save() is the transmission mechanism

Only values touched by `.save()` (or `nnsight.save(...)`) are returned. Everything else is local to the server run and discarded after the job finishes.

```python
with model.trace("Hello", remote=True):
    hidden = model.transformer.h[5].output          # not saved -> not returned
    answer = model.lm_head.output[0][-1].argmax(dim=-1).save()   # returned

# 'hidden' is undefined here; 'answer' is a real tensor.
```

Your local variable name is matched to the saved value when the result is pushed back into your frame.

## Move tensors to CPU before saving

Saved values are pickled and shipped over HTTPS. The deserializer always uses `torch.load(..., map_location="cpu")` (`src/nnsight/intervention/backends/remote.py:196`), so tensors land on CPU locally regardless. Calling `.detach().cpu()` inside the block still helps for large activations — it drops the autograd graph and does the conversion server-side, where the tensor already lives, instead of shipping an autograd-attached payload:

```python
with model.trace("Hello", remote=True):
    hidden = model.transformer.h[0].output.detach().cpu().save()
```

## Print statements appear as LOG status

Anything you `print(...)` inside the block runs on the server and is forwarded as a `LOG` response, rendered inline by the status display:

```python
with model.trace("Hello", remote=True):
    h = model.transformer.h[0].output
    print(f"hidden mean: {h.mean()}")     # rendered as: ℹ [job-id] LOG  hidden mean: ...
    out = model.lm_head.output.save()
```

To silence remote logging (and the spinner), set `CONFIG.APP.REMOTE_LOGGING = False`.

## Generation

`model.generate(input, max_new_tokens=N, remote=True)` runs the model and returns **token ids**. Read them off `tracer.result`:

```python
with model.generate("Hello", max_new_tokens=5, do_sample=False, remote=True) as tracer:
    output = tracer.result.save()

print(model.tokenizer.decode(output[0]))
```

`tracer.result` is the generation output (prompt ids plus the new tokens). Use `for _ in tracer.iter[:N]:` for per-step interventions inside generation; see [docs/usage/generate.md](../usage/generate.md). (`model.generate` returns ids; `model.pipe` runs the whole task pipeline and returns its decoded records.)

## Custom host or pre-built backend

```python
# Hit a custom server URL (string form of `remote`)
with model.trace("...", remote="https://self-hosted.example.com"):
    out = model.lm_head.output.save()

# Construct RemoteBackend manually for advanced cases
from nnsight.intervention.backends.remote import RemoteBackend

backend = RemoteBackend(
    model.to_model_key(),
    host="https://api.ndif.us",
    api_key="...",          # overrides CONFIG.API.APIKEY
    blocking=True,
    verbose=True,
)

with model.trace("...", backend=backend):
    out = model.lm_head.output.save()
```

`RemoteBackend.__init__` is at `src/nnsight/intervention/backends/remote.py:47`; its parameters are `model_key, host, api_key, env, blocking, job_id, verbose`. (There is no `callback`/webhook parameter.)

## Gotchas

- Variables created **outside** the trace and mutated inside it won't be transmitted back. Create and `.save()` the value *inside* the block — see [remote-session.md](./remote-session.md).
- The model identifier passed to `TransformersModel` must match an NDIF deployment. Run `nnsight.is_model_running("...")` before submitting.
- A GPT-2 block's `.output` is a plain tensor `(batch, seq, hidden)` in current transformers — `model.transformer.h[i].output` selects the first batch row. Attention submodules still return a tuple; don't assume — check `model.transformer.h[i].source`.
- A `LOG` line that prints a multi-megabyte tensor floods the websocket. Print summaries, not raw tensors.
- Local helper modules (anything not installed on the server) ship automatically via `pull_env`, or register them with `nnsight.register(...)`. See [register-local-modules.md](./register-local-modules.md).
- Define helpers at module scope. Dynamically created functions (lambdas built in loops, `exec`-ed code) can confuse the source-based serializer.

## Related

- [ndif-overview.md](./ndif-overview.md) — full request lifecycle.
- [remote-session.md](./remote-session.md) — multiple traces in one queue wait.
- [non-blocking-jobs.md](./non-blocking-jobs.md) — submit and poll later.
- [remote-async.md](./remote-async.md) — await a job from an event loop.
- [register-local-modules.md](./register-local-modules.md) — when the server can't import your code.
- [docs/usage/trace.md](../usage/trace.md) — local tracing.
