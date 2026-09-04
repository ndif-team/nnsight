---
title: Generate
one_liner: Multi-token generation through the model; returns token ids (`tracer.result`). Greedy by default.
tags: [usage, tracing, generation]
related: [docs/usage/trace.md, docs/usage/pipe.md, docs/usage/iter-all-next.md, docs/gotchas/iteration.md, docs/usage/invoke-and-batching.md]
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

Ask for sampling explicitly if you want it: `model.generate(..., do_sample=True, top_k=50)`. Sampled generation is reproducible from a plain `torch.manual_seed(n)` immediately before the call — inside a trace or outside one:

```python
import torch

torch.manual_seed(0)
with model.generate("The Eiffel Tower is in the city of", max_new_tokens=5, do_sample=True, top_k=50) as tracer:
    a = tracer.result.save()
torch.manual_seed(0)
with model.generate("The Eiffel Tower is in the city of", max_new_tokens=5, do_sample=True, top_k=50) as tracer:
    b = tracer.result.save()
# torch.equal(a, b) -> True
```

## Accessing the result

```python
# Preferred: tracer.result
with model.generate("Hello", max_new_tokens=5) as tracer:
    ids = tracer.result.save()

# Deprecated: reading the finished ids through the generator passthrough
with model.generate("Hello", max_new_tokens=5):
    ids = model.generator.output.save()   # NNsightDeprecationWarning
```

The generated ids are passed through a `Generator` module so a worker parked on `model.generator.output` receives them. It is the same tensor `tracer.result` gives, prompt ids included, and reading it warns:

```
NNsightDeprecationWarning: model.generator.output is deprecated; use
tracer.result instead (model.generator.streamer.output still gives per-step tokens).
```

`tracer.result` is served during the run, so it has to be read inside the block — after the `with` block it raises ``ValueError: Cannot access `result` outside of interleaving``.

## Per-step interventions

```python
import nnsight

with model.generate("Hello", max_new_tokens=5) as tracer:
    per_step = nnsight.save([])
    for step in tracer.iter[:5]:
        per_step.append(model.output.logits[0, -1].argmax(dim=-1))
```

A loop must not ask for a step the run does not make: `iter[:5]` is right here because `max_new_tokens=5` and nothing ends the generation sooner. See `docs/usage/iter-all-next.md` for the forms and `docs/gotchas/iteration.md` for what happens when a loop outruns the run.

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
    ["The Eiffel Tower is in", "Paris"],
    max_new_tokens=3, do_sample=False,
) as tracer:
    ids = tracer.result.save()
# ids.shape -> torch.Size([2, 10])
# row 0: 'The Eiffel Tower is in the middle of'
# row 1: '<|endoftext|>' * 6 + 'Paris, France,'
```

The padding is still there in the result, so decode with
`skip_special_tokens=True` (or slice the row) unless you want it.

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

## Extra kwargs go straight to the model's `generate`

Anything the tracer does not consume is forwarded to `model.generate` —
`do_sample`, `temperature`, `top_p`, `num_return_sequences`,
`generation_config=`, stopping criteria — and whatever it returns is what lands
on `tracer.result`. So the return type is yours to choose:

```python
# default: token ids
with model.generate(prompt, max_new_tokens=3) as tracer:
    ids = tracer.result.save()

# ask HF for the full output object and you get it, scores included
with model.generate(
    prompt, max_new_tokens=3,
    return_dict_in_generate=True, output_scores=True,
) as tracer:
    out = tracer.result.save()

out.sequences        # [batch, prompt + new]
out.scores           # one [batch, vocab] tensor per generated step
```

That is the way to get per-step logits out of a traced generation without
reading `lm_head` under `tracer.iter`.

Two of those kwargs decide how many steps an iteration loop may ask for:
`max_new_tokens=N` is an upper bound, and `min_new_tokens=N` holds the
generation to N steps by suppressing EOS until then.

## Gotchas

- **A `tracer.iter` loop must not ask for a step the run does not make.** A bound the run meets is fine and the code after the loop runs; a loop that outruns the run — bounded or open — warns, keeps what it saved, and drops the statements after the loop, so the result looks complete while being shorter than the bound. `max_new_tokens` is an upper bound, so pass `min_new_tokens=` when the loop's bound has to hold, and check the `len()` of what you collected. See `docs/gotchas/iteration.md`.
- Always pass a stop bound (`max_new_tokens=` or a `generation_config`).
- Within a step, modules must still be accessed in forward-pass order — inside an iteration loop an out-of-order write parks on the *next* step instead.
- Reading `model.generator.output` for the finished ids is deprecated — use `tracer.result`.

## Related

- `docs/usage/trace.md`
- `docs/usage/pipe.md`
- `docs/usage/iter-all-next.md`
- `docs/gotchas/iteration.md`
- `docs/usage/invoke-and-batching.md`
- `docs/usage/save.md`
