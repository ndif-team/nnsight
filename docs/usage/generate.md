---
title: Generate
one_liner: Multi-token generation through the model; returns token ids (`tracer.result`). Greedy by default.
tags: [usage, tracing, generation]
related: [docs/usage/trace.md, docs/usage/pipe.md, docs/usage/iter-all-next.md, docs/usage/invoke-and-batching.md]
sources: [src/nnsight/modeling/transformers.py, src/nnsight/intervention/tracer.py]
---

# Generate

## What this is for

`model.generate(input, max_new_tokens=N, ...)` traces multi-token autoregressive generation. It runs the model's own `generate` (each new token is one forward pass) and returns the **generated token ids** — read them off `tracer.result`. Interventions in the block run against every forward the decode loop makes; use `tracer.iter` to target a particular step.

Generate goes **through the model**, not the task's pipeline. It takes the same inputs a forward does (text, token ids, a tensor, or an encoding) and uses the checkpoint's own generation settings — so it is **greedy by default** (it does not apply the `task_specific_params` sampling a pipeline would). To get the pipeline's decoded records instead, use `model.pipe(...)` — see `docs/usage/pipe.md`.

## When to use / when not to use

- Use for autoregressive generation where you want token ids and/or per-step interventions.
- Use `model.pipe(...)` when you want the task pipeline's decoded output (text, labels). See `docs/usage/pipe.md`.
- Use `model.trace(...)` for a single forward pass.

## Canonical pattern

```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.generate("The Eiffel Tower is in the city of", max_new_tokens=3) as tracer:
    ids = tracer.result.save()

print(ids.shape)                          # torch.Size([1, 13])  (10 prompt + 3 new)
print(model.tokenizer.decode(ids[0]))     # The Eiffel Tower is in the city of Paris, and
```

The result is a `[batch, seq]` tensor of ids — the whole prompt plus completion.

## Greedy by default

Generating through the model uses the checkpoint's settings, not the pipeline's `task_specific_params` (which for gpt2 ask for `do_sample=True`). So two generates match:

```python
with model.generate("The Eiffel Tower is in the city of", max_new_tokens=3) as t:
    a = t.result.save()
with model.generate("The Eiffel Tower is in the city of", max_new_tokens=3) as t:
    b = t.result.save()
# torch.equal(a, b) -> True
```

Ask for sampling explicitly if you want it: `model.generate(..., do_sample=True, top_k=50)`. Kwargs are forwarded to the model's `generate`, so `generation_config=`, `num_return_sequences=`, etc. all work.

## Accessing the result

```python
# Preferred: tracer.result
with model.generate("Hello", max_new_tokens=5) as tracer:
    ids = tracer.result.save()

# Deprecated: reading the finished ids through the generator passthrough
with model.generate("Hello", max_new_tokens=5):
    ids = model.generator.output.save()   # same value, but use tracer.result
```

The generated ids are passed through a `Generator` module so a worker parked on `model.generator.output` receives them. Reading the finished ids there is deprecated (use `tracer.result`); the module remains for per-step streamer access.

## Per-step interventions

```python
import nnsight

with model.generate("Hello", max_new_tokens=5) as tracer:
    per_step = nnsight.save([])
    for step in tracer.iter[:5]:
        per_step.append(model.output.logits[0, -1].argmax(dim=-1))
```

For `tracer.iter[...]`, `tracer.all()`, and the unbounded-iterator footgun, see `docs/usage/iter-all-next.md`.

## Per-step tokens (streamer)

`model.generator.streamer.output` gives the tokens as they are decoded — the prompt arrives as one block, then one new token per step:

```python
import nnsight

with model.generate("The Eiffel Tower is in the city of", max_new_tokens=3) as tracer:
    steps = nnsight.save([])
    for step in tracer.iter[:3]:
        steps.append(model.generator.streamer.output)
# [tuple(s.shape) for s in steps] -> [(1, 10), (1,), (1,)]
```

## Batched generation

Pass a list of prompts, or use `tracer.invoke(...)` per prompt. Shorter prompts are left-padded up to the longest:

```python
with model.generate(
    ["The Eiffel Tower is in", "The Colosseum is in"],
    max_new_tokens=3, do_sample=False,
) as tracer:
    ids = tracer.result.save()
# ids.shape -> torch.Size([2, 10])
# rows decode to "...is in the middle of" each
```

## Difference vs `trace` and `pipe`

| Aspect | `model.trace(...)` | `model.generate(...)` | `model.pipe(...)` |
|---|---|---|---|
| Runs | one forward (`__call__`) | the model's `generate` | the whole task pipeline |
| Iterations | 1 | one per new token | pipeline-defined |
| Result | model output | **token ids** | pipeline **records** (text/labels) |
| Sampling | n/a | greedy unless asked | pipeline's `task_specific_params` apply |

## Remote generation

```python
with model.generate("Hello", max_new_tokens=5, remote=True) as tracer:
    ids = tracer.result.save()
```

`remote=True` serializes the trace and runs it on NDIF; `tracer.result` is shipped back.

## Called directly (no block)

Without a `with`, `generate` just generates and still returns the ids:

```python
ids = model.generate("Hello", max_new_tokens=3)   # torch.Tensor
```

## Gotchas

- **Unbounded iter eats trailing code**: `for step in tracer.iter[:]: ...` runs until the model stops; code after the loop in the same invoke may not run as expected. Use a bounded slice or a separate invoke. See `docs/gotchas/unbounded-iter.md`.
- Always pass a stop bound (`max_new_tokens=` or a `generation_config`).
- Within a step, modules must still be accessed in forward-pass order.
- Reading `model.generator.output` for the finished ids is deprecated — use `tracer.result`.

## Related

- `docs/usage/trace.md`
- `docs/usage/pipe.md`
- `docs/usage/iter-all-next.md`
- `docs/usage/invoke-and-batching.md`
- `docs/usage/save.md`
