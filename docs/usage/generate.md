---
title: Generate
one_liner: Multi-token autoregressive generation under a tracing context (`model.generate(...)`).
tags: [usage, tracing, generation]
related: [docs/usage/trace.md, docs/usage/iter.md, docs/usage/all-and-next.md, docs/usage/invoke-and-batching.md]
sources: [src/nnsight/modeling/language.py:140, src/nnsight/intervention/envoy.py:963, src/nnsight/intervention/tracing/iterator.py, src/nnsight/intervention/tracing/tracer.py:581]
---

# Generate

## What this is for

`model.generate(input, max_new_tokens=N, ...)` opens a tracing context that drives multi-token generation. It is the same `InterleavingTracer` as `.trace()`, but it dispatches to the model's `generate` method (e.g. HuggingFace `GenerationMixin.generate`) instead of `__call__`. Each token = one forward pass = one "iteration".

For `LanguageModel`, kwargs are forwarded to `transformers.GenerationMixin.generate`. Internally a streamer is wired through `model.generator` so per-step values are observable.

## When to use / when not to use

- Use for autoregressive generation — sampling, decoding, multi-token interventions per step.
- Use `tracer.iter[...]` to scope intervention code to specific generation steps.
- Use `.trace(...)` for a single forward pass.

## Canonical pattern

```python
from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

with model.generate("Hello", max_new_tokens=5) as tracer:
    output = tracer.result.save()         # final generation tensor
    last_hidden = model.transformer.h[-1].output[0].save()

print(model.tokenizer.decode(output[0]))
```

## How it works

`Envoy.__getattr__` wraps callables on the underlying module so `model.generate(...)` becomes a tracing context (`src/nnsight/intervention/envoy.py:963`). For `LanguageModel`, a custom entry point `__nnsight_generate__` runs (`src/nnsight/modeling/language.py:140`):

- Sets `interleaver.default_all = max_new_tokens` so unbounded `tracer.iter[:]` knows when to stop.
- Injects the streamer (`model.generator.streamer`) so each new token flows through `generator(...)`.
- Returns the final generation, mounted onto `model.generator` (so `model.generator.output` is also valid).

## Accessing the result

Two equivalent ways:

```python
# Preferred: tracer.result is an eproperty of InterleavingTracer
with model.generate("Hello", max_new_tokens=5) as tracer:
    output = tracer.result.save()

# Also valid: model.generator.output captures the same value via the
# wrapper module installed by LanguageModel
with model.generate("Hello", max_new_tokens=5):
    output = model.generator.output.save()
```

`tracer.result` is defined as `@eproperty(iterate=False)` on `InterleavingTracer` (`src/nnsight/intervention/tracing/tracer.py:581`) and provided by `Envoy.interleave` after the model returns (`envoy.py:590`).

## Per-step interventions

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    logits_per_step = list().save()
    for step in tracer.iter[:]:
        logits_per_step.append(model.lm_head.output[0, -1].argmax(dim=-1))
```

For details on `tracer.iter[...]`, `tracer.all()`, `module.next()`, and the unbounded-iterator footgun, see `docs/usage/iter.md` and `docs/usage/all-and-next.md`.

## Difference vs `.trace()`

| Aspect | `model.trace(...)` | `model.generate(...)` |
|---|---|---|
| Function dispatched | `model.__call__` | `model.generate` (autoregressive) |
| Iterations | 1 | `max_new_tokens` (one per token) |
| Result accessor | `tracer.result` | `tracer.result` or `model.generator.output` |
| `default_all` set | No | Yes (= `max_new_tokens`) |
| Step iteration | Not meaningful | `tracer.iter[...]`, `module.next()` |

Same Tracer class is used (`InterleavingTracer`); only the wrapped `fn` differs.

## Remote generation

```python
with model.generate("Hello", max_new_tokens=5, remote=True) as tracer:
    output = tracer.result.save()
```

`remote=True` is supported via `RemoteableMixin.trace` — `model.generate` flows through the same `__getattr__` wrap which calls `self.trace(*args, fn=self.generate, ...)`. The remote backend serializes the trace and sends it to NDIF.

## Gotchas

- **Unbounded iter eats trailing code**: `for step in tracer.iter[:]: ...` waits indefinitely for more iterations. Code *after* the for loop in the same invoke never runs. Use a separate empty invoke or a bounded slice. See `docs/gotchas/unbounded-iter.md`.
- Always pass `max_new_tokens` (or the underlying generation config equivalent) — without a stop bound, `tracer.iter[:]` cannot resolve the end.
- Inside a single invoke, modules must still be accessed in forward-pass order *within each step* — the iteration counter handles step-to-step ordering, not within-step ordering.
- `model.generator.streamer.output` is the per-token stream; `model.generator.output` is the final stacked output.

## Related

- `docs/usage/trace.md`
- `docs/usage/iter.md`
- `docs/usage/all-and-next.md`
- `docs/usage/invoke-and-batching.md`
- `docs/usage/save.md`
