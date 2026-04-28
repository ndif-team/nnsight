---
title: Remote Trace
one_liner: Run a single forward-pass trace on NDIF by passing remote=True.
tags: [remote, ndif, trace]
related: [docs/remote/index.md, docs/remote/remote-session.md, docs/remote/non-blocking-jobs.md, docs/usage/trace.md]
sources: [src/nnsight/modeling/mixins/remoteable.py:31, src/nnsight/intervention/backends/remote.py:421, src/nnsight/intervention/backends/remote.py:869]
---

# Remote Trace

## What this is for

`model.trace(input, remote=True)` is the simplest way to execute an intervention on NDIF. It's the same `trace` you call locally — you just add `remote=True`, and the backend serializes the intervention graph, ships it to NDIF over HTTP, and waits on a WebSocket for completion.

## When to use / when not to use

- Use for one-off remote runs.
- Use when working with a model that's too large to load locally.
- For multiple traces in a row, prefer `model.session(remote=True)` to share one queue wait — see [remote-session.md](./remote-session.md).
- For long-running jobs you don't want to block on, use `blocking=False` — see [non-blocking-jobs.md](./non-blocking-jobs.md).

## Canonical pattern

```python
from nnsight import LanguageModel, CONFIG

CONFIG.set_default_api_key("YOUR_KEY")

model = LanguageModel("meta-llama/Llama-3.1-70B")
print(model.device)   # 'meta' — no GPU memory used locally

with model.trace("The Eiffel Tower is in the city of", remote=True):
    logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

print(model.tokenizer.decode(logit))   # 'Paris'
```

What happens under the hood (`src/nnsight/modeling/mixins/remoteable.py:62` and `src/nnsight/intervention/backends/remote.py:869`):

1. `trace(remote=True)` constructs `RemoteBackend(self.to_model_key(), blocking=True)`.
2. On `__exit__`, the backend serializes `RequestModel(interventions, tracer)` to compressed bytes.
3. `submit_request` POSTs to `{HOST}/request` and gets back a `ResponseModel` with `id` (the job ID).
4. A WebSocket connects to `{HOST}/ws/socket.io` and listens for status updates until `COMPLETED` or `ERROR`.
5. On `COMPLETED`, the result is downloaded (often a presigned URL), decompressed, deserialized via `torch.load`, and `tracer.push(result)` injects the saved values into your local frame.

## .save() is the transmission mechanism

Only values touched by `.save()` (or `nnsight.save(...)`) are returned. Everything else is local to the worker thread on the server and discarded after the job finishes.

```python
with model.trace("Hello", remote=True):
    hidden = model.transformer.h[5].output   # not saved -> not returned
    answer = model.lm_head.output.argmax(dim=-1).save()   # returned

# 'hidden' is undefined here; 'answer' is a real tensor.
```

The value name in the trace becomes the key in the result dict that's pushed back into your frame. `tracer.push(result)` matches your local variable name to the saved value.

## Move tensors to CPU before saving

Saved values are pickled and shipped over HTTPS. For large activations, calling `.detach().cpu()` first reduces the payload (no autograd graph, no GPU pinned memory ceremony) and avoids loading on a default CUDA device when unpickling locally.

```python
with model.trace("Hello", remote=True):
    hidden = model.transformer.h[0].output.detach().cpu().save()
```

The deserializer always uses `torch.load(..., map_location="cpu")` (`src/nnsight/intervention/backends/remote.py:773`), so CPU is the local default regardless — but `.detach().cpu()` runs the conversion server-side, where the tensor is already in GPU memory, instead of bouncing the autograd-attached payload over the network.

## Print statements appear as LOG status

Anything you `print(...)` inside the trace runs on the server and is forwarded as a `LOG` response. The status display renders these inline:

```python
with model.trace("Hello", remote=True):
    h = model.transformer.h[0].output
    print(f"hidden mean: {h.mean()}")    # rendered as: ℹ [job-id] LOG  hidden mean: ...
    out = model.lm_head.output.save()
```

To silence remote logging entirely, set `CONFIG.APP.REMOTE_LOGGING = False` (status spinner disappears too).

## Generation

`model.generate(input, max_new_tokens=N, remote=True)` works the same way. To collect the final tokens:

```python
with model.generate("Hello", max_new_tokens=5, remote=True) as tracer:
    output = tracer.result.save()

print(model.tokenizer.decode(output[0]))
```

`tracer.result` is the generation output. Use `tracer.iter[:]` for per-step interventions inside generation; see [docs/usage/generate.md](../usage/generate.md).

## Custom host or pre-built backend

```python
# Hit a custom server URL
with model.trace("...", backend="https://self-hosted.example.com"):
    out = model.lm_head.output.save()

# Construct RemoteBackend manually for advanced cases
from nnsight.intervention.backends.remote import RemoteBackend

backend = RemoteBackend(
    model_key=model.to_model_key(),
    host="https://api.ndif.us",
    blocking=True,
    api_key="...",       # overrides env / CONFIG
    callback="https://my-webhook/done",
    verbose=True,
)

with model.trace("...", backend=backend):
    out = model.lm_head.output.save()
```

`RemoteBackend.__init__` is at `src/nnsight/intervention/backends/remote.py:470`.

## Gotchas

- Variables created **outside** the trace and mutated inside it won't be transmitted back. Do `my_list = list().save()` *inside* the trace — see [remote-session.md](./remote-session.md) for the canonical pattern.
- The model identifier passed to `LanguageModel` must match an NDIF deployment. Run `nnsight.is_model_running("...")` before submitting.
- A `LOG` line that prints a multi-megabyte tensor will swamp the WebSocket. Print summaries, not raw tensors.
- Local helper modules (anything not pip-installable on the server) need `nnsight.register(...)`. Otherwise the worker raises `ModuleNotFoundError` during deserialization.
- Functions defined dynamically (lambdas inside loops, `exec()`-ed code) sometimes confuse the cloudpickle source extractor. Define helpers at module scope.

## Related

- [ndif-overview.md](./ndif-overview.md) — full request lifecycle.
- [remote-session.md](./remote-session.md) — multiple traces in one queue wait.
- [non-blocking-jobs.md](./non-blocking-jobs.md) — submit and poll later.
- [register-local-modules.md](./register-local-modules.md) — when remote can't import your code.
- [docs/usage/trace.md](../usage/trace.md) — local tracing.
