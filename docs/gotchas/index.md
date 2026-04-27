---
title: Gotchas Index
one_liner: Bucketed catalog of common pitfalls and the fixes for them. Start here when something is "weirdly broken".
tags: [gotcha, index]
related: [docs/usage/index.md, docs/concepts/index.md, docs/remote/index.md]
sources: [src/nnsight/intervention/interleaver.py, src/nnsight/intervention/tracing/tracer.py]
---

# Gotchas Index

This folder is the failure-mode reference. Each doc covers one cluster of related pitfalls — symptom, cause, wrong code, right code. If a trace is misbehaving, find the bucket that matches and read it.

## Buckets

### [save.md](save.md)
- Forgetting `.save()` — values are garbage collected after the trace exits.
- `.save()` is required inside `model.scan()` too, not just `model.trace()`.
- Remote `.save()` is the *only* mechanism that transmits values back from NDIF.
- Local `list.append(x.save())` outside the trace body does not get populated remotely — build the list inside.
- Prefer `nnsight.save(x)` over `x.save()` for non-tensor objects (doesn't depend on the `PYMOUNT` C extension).

### [modification.md](modification.md)
- In-place `[:] = ` vs replacement `=` have different semantics (one mutates, one rebinds the output value).
- Most transformer blocks return tuples — `output[0][:] = 0` mutates the first element; `output = ...` replaces the whole tuple.
- Saving the "before" state of an in-place edit requires `.clone().save()` — otherwise `before` and `after` alias the same tensor.
- Activation patching across invokes that *both* read the same module needs `.clone()` so the slice is materialized as a new tensor before the second invoke overwrites it.

### [order-and-deadlocks.md](order-and-deadlocks.md)
- Module access within an invoke must follow forward-pass order — request layer 5's output then layer 2's output and the worker thread deadlocks (raised as `OutOfOrderError`).
- `model.trace()` with no positional input and no explicit `tracer.invoke(...)` errors — the model never executes.
- You cannot create a `tracer.invoke(...)` *inside* an active forward pass — invokes are top-level only.

### [iteration.md](iteration.md)
- `for step in tracer.iter[:]` and `tracer.all()` are unbounded — pure-Python after the loop runs, but any module `.output`/`.input` access in the trailing code raises `OutOfOrderError` because the model's forward passes are already done.
- Fix with bounded `iter[:N]` or by wrapping the iter in its own `tracer.invoke(...)` and putting trailing module-access logic in a separate empty invoke.
- `.next()` is the manual alternative *inside* an iter loop. Outside an iter loop the iteration tracker is dormant, so `.next()` chains do not work as the older docs suggested — use `tracer.iter[...]` instead.
- The iteration tracker is per-module, only maintained while inside an `iter[...]` loop (lifecycle is `__iter__` setup / `finally` teardown). Outside an iter loop it stays at whatever value it was when the last loop ended.

### [cross-invoke.md](cross-invoke.md)
- If two invokes both access the same module's `.output`/`.input`, sharing a Python variable between them requires `tracer.barrier(n)`.
- `CONFIG.APP.CROSS_INVOKER` (default `True`) controls whether variables flow between invokes at all.
- Empty `tracer.invoke()` (no input) works on bare `NNsight` models — it does not call `_batch()`, so it never raises `NotImplementedError`.
- Decision rule: same module accessed in two invokes? → barrier. Different modules? → no barrier needed (sharing happens automatically via `cross_invoker` push/pull).

### [backward.md](backward.md)
- `with tensor.backward():` opens a *separate* interleaving session — get any `.output`/`.input` you need *before* entering it.
- `.grad` lives on tensors, not modules.
- Gradient access order is the *reverse* of forward access order (gradients flow backward).
- `retain_graph=True` is required if you call `.backward()` more than once on overlapping graphs.
- A standalone `with loss.backward():` outside any `model.trace()` works for simple cases.

### [remote.md](remote.md)
- `.save()` is the only transmission channel — variables not saved are not returned.
- `local_list.append(x.save())` where `local_list` was created *outside* the trace gets the local list nothing — build the list inside the trace.
- `.detach().cpu()` before `.save()` to shrink the download payload.
- Put `remote=True` on `model.session(...)`, not on the inner `model.trace(...)` calls.
- `print(...)` inside a remote trace appears as `LOG` status, not in your local stdout.
- Mismatched local vs server environments raise warnings (see [docs/remote/env-comparison.md]).
- Helper functions/classes defined locally must be registered with `nnsight.register(...)` to be importable on the server (see [docs/remote/register-local-modules.md]).

### [types-and-values.md](types-and-values.md)
- Inside a trace, `.output`/`.input` deliver *real* tensors — `print`, `.shape`, arithmetic all work directly. There are no proxies.
- Use `model.scan(...)` to inspect shapes *without* running the model — values come back as `FakeTensor`s.
- Tensors you create inside a trace must be moved onto the right device (e.g. `torch.randn(...).to(model.transformer.h[0].output[0].device)`).
- Inside `.scan()`, `FakeTensor.__bool__` is patched to always return `True` — Python `if` statements on fake tensors don't reflect runtime truthiness.

### [integrations.md](integrations.md)
- `LanguageModel(hf_model)` on a pre-loaded HF model raises `AttributeError: Tokenizer not found` — pass `tokenizer=`.
- Calling `.source` on a *module* fn from inside another `.source` raises a `ValueError` — access the submodule directly via its envoy.
- Auxiliary modules (SAEs, LoRA adapters) called inside a trace need `module(x, hook=True)` if you want to access `.input`/`.output` on that auxiliary module afterwards.
- vLLM `tracer.invoke(prompt, temperature=..., top_p=..., max_tokens=...)` forwards these to vLLM's `SamplingParams`.
- vLLM pipeline parallelism is not supported (only TP and DP). `pipeline_parallel_size` is forced to 1.

## Where to go next

- Per-feature usage docs are in [docs/usage/](../usage/index.md).
- Architecture / threading mental model is in [docs/concepts/](../concepts/index.md).
- Errors and how to read them: [docs/errors/](../errors/) (parallel folder).
