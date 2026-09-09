---
title: Gotchas Index
one_liner: Bucketed catalog of common pitfalls and the fixes for them. Start here when something is "weirdly broken".
tags: [gotcha, index]
related: [docs/usage/index.md, docs/concepts/index.md, docs/errors/index.md, docs/remote/index.md]
sources: [src/nnsight/intervention/interleaver.py, src/nnsight/tracing/tracer.py]
---

# Gotchas Index

This folder is the failure-mode reference. Each doc covers one cluster of related pitfalls — symptom, cause, wrong code, right code. If a trace is misbehaving, find the bucket that matches and read it. For a single error message and its fix, see the parallel [errors/](../errors/index.md) folder.

## Buckets

### [save.md](save.md)
- **`save()` raises outside a trace** — move every save inside the `with` block.
- Forget `.save()` and the value doesn't cross the trace boundary (`UnboundLocalError` inside a function).
- `.save()` returns the value unchanged; save the value you bind. Nested traces don't need `.save()` between them (only the outermost boundary filters).
- Remote `.save()` is the only transmission channel; collect via a saved container, not an external list.

### [modification.md](modification.md)
- In-place `[:] =` mutates; `=` rebinds and fires a `SWAP` — both take effect, differently.
- A GPT-2 **block** `.output` is a plain tensor `(batch, seq, hidden)` — not a tuple. Attention submodules still return tuples; assigning `attn.output[0] = t` raises `TypeError`.
- `.clone()` before mutating if you want the "before" state, and for slices shared across invokes.

### [order-and-deadlocks.md](order-and-deadlocks.md)
- Out-of-order module access raises `OutOfOrderError` (greenlets — an error at run end, **not** a hang).
- `trace()` with no input and no `tracer.invoke(...)` raises.
- Opening a `tracer.invoke(...)` while the model runs (nested invoke, or inside an iter loop) raises `Cannot invoke while the model is already running.`

### [iteration.md](iteration.md)
- Loop form is `for step in tracer.iter[...]:`; the `with tracer.iter[...]:` form is deprecated and warns.
- Unbounded `tracer.iter[:]` (and `tracer.all()`) drops **all** code after the loop — the worker unwinds at the final dangling step (a warning, not an error). There is no `default_all`. Bound the loop or use a separate empty `tracer.invoke()` for trailing code.
- `.next()` does not exist. `tracer.iter[N]` targets the `(N+1)`-th occurrence of a location.

### [cross-invoke.md](cross-invoke.md)
- A value produced *inside* one invoke needs `tracer.barrier(n)` to reach another (same or different module) — all invoke workers start together, so the consumer runs before the producer binds it (`NameError`).
- A value from the enclosing scope flows into every invoke with no barrier.
- `CONFIG.APP.CROSS_INVOKER` is gone. Empty `tracer.invoke()` works on bare `NNsight`; multiple *input* invokes need a batching model.

### [backward.md](backward.md)
- `with tensor.backward():` interleaves the real backward — capture forward tensors *before* the block (accessing `.output` inside raises `OutOfOrderError`).
- No `requires_grad_(True)` needed. `.grad` is on tensors, not modules. Gradient order is reverse-forward.
- `retain_graph=True` for multiple backwards; standalone backward works if you save the forward tensors.

### [types-and-values.md](types-and-values.md)
- Values inside a trace are **real** tensors, not proxies.
- `model.scan(...)` gives `FakeTensor`s (shapes/dtypes). **Branching on their content raises** (`GuardOnDataDependentSymNode`). Branch only on shapes in scan.
- Tensors you create inside a trace must be moved onto the model's device.

### [remote.md](remote.md)
- `.save()` is the only channel back; build containers inside the trace; `.detach().cpu()` before save.
- `remote=True` on `model.session(...)`, not inner traces. `print` → `LOG` status. Register local helpers with `nnsight.register(...)`.
- Test offline with `remote="local"`.

### [integrations.md](integrations.md)
- `LanguageModel`/`VisionLanguageModel` are deprecated (warn) — use `TransformersModel(repo_id, task=...)`.
- `.source` can't drill into a **submodule** call — go to that submodule's envoy. Op names are the dotted path joined with `_` (`self_c_proj_0`).
- Auxiliary modules (SAE/LoRA) need `module(x, hook=True)` (applied via `edit()`) for their internals to be observable.
- vLLM: sampling kwargs on `trace`/`invoke`; intervene at `model.logits` / `model.samples`; `mode="sync"|"async"`.

## Where to go next

- Per-error pages: [docs/errors/](../errors/index.md).
- Per-feature usage docs: [docs/usage/](../usage/index.md).
- Architecture / greenlet mental model: [docs/concepts/](../concepts/index.md).
