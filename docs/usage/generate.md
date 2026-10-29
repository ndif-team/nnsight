---
title: Generate
one_liner: Multi-token generation through the model; returns token ids (`tracer.result`). Decoding follows the checkpoint's `generation_config`.
tags: [usage, tracing, generation]
related: [docs/usage/trace.md, docs/usage/pipe.md, docs/usage/iteration.md, docs/usage/invoke-and-batching.md, docs/usage/save.md]
sources: [src/nnsight/modeling/transformers.py, src/nnsight/intervention/tracer.py]
---

# Generate

`model.generate(input, max_new_tokens=N, ...)` traces autoregressive generation. It runs the model's own `generate` (one forward per new token) and returns the **generated token ids** on `tracer.result`. Interventions in the block run against the decode loop's forwards; target particular steps with [`tracer.iter`](iteration.md). It takes the same inputs a forward does (text, token ids, a tensor, an encoding). For the task pipeline's decoded records use [pipe](pipe.md); for one forward, [trace](trace.md).

## Canonical pattern

```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.generate("The Eiffel Tower is in the city of", max_new_tokens=3) as tracer:
    ids = tracer.result.save()

print(ids.shape)                          # torch.Size([1, 13])  (10 prompt + 3 new)
print(model.tokenizer.decode(ids[0]))     # The Eiffel Tower is in the city of Paris, and
```

The result is a `[batch, seq]` tensor of ids — prompt plus completion. `tracer.result` is served during the run, so read it inside the block; afterwards it raises ``ValueError: Cannot access `result` outside of interleaving``. Without a `with`, `ids = model.generate("Hello", max_new_tokens=3)` just returns the ids.

## Decoding follows the checkpoint's `generation_config`

Generate decodes with the checkpoint's own `generation_config`, not the `task_specific_params` a pipeline would apply. gpt2's is greedy; many instruct checkpoints (`gemma-3-1b-it`, `Llama-3.1-8B-Instruct`, `Qwen3-4B-Instruct`) set `do_sample=True` with a `temperature` and `top_p`, so **generate samples on those unless you say otherwise** — two identical generates differ, and a logit lens or probe reading the run disagrees with the tokens it produced. Check `model.generation_config.do_sample` (`None`, as on gpt2, means unset, i.e. greedy), or simply pass `do_sample=False` for determinism:

```python
with model.generate("The Eiffel Tower is in the city of", max_new_tokens=12, do_sample=False) as tracer:
    ids = tracer.result.save()
```

Sampled generation (`do_sample=True, top_k=50`) is reproducible from a plain `torch.manual_seed(n)` immediately before the call.

## Extra kwargs go straight to the model's `generate`

Anything the tracer does not consume is forwarded — `do_sample`, `temperature`, `top_p`, `num_return_sequences`, `generation_config=`, stopping criteria — and whatever HF returns lands on `tracer.result`:

```python
with model.generate(
    prompt, max_new_tokens=3,
    return_dict_in_generate=True, output_scores=True,
) as tracer:
    out = tracer.result.save()

out.sequences        # [batch, prompt + new]
out.scores           # one [batch, vocab] tensor per generated step
```

That is the way to get per-step logits without reading `lm_head` under `tracer.iter`.

**Bound the run, and the loops over it.** Always pass `max_new_tokens=` (or a `generation_config`). It is an upper bound: a `tracer.iter` loop that asks for more steps than the run makes is cut short with a warning and the code after it is dropped. `min_new_tokens=N` holds the run to N steps by suppressing EOS. See [iteration](iteration.md#the-one-rule).

## Per-step tokens (streamer)

`model.generator.streamer.output` gives tokens as they are decoded — the prompt as one block, then one new token per step:

```python
import nnsight

with model.generate("The Eiffel Tower is in the city of", max_new_tokens=3) as tracer:
    steps = nnsight.save([])
    for step in tracer.iter[:3]:
        steps.append(model.generator.streamer.output)
# [tuple(s.shape) for s in steps] -> [(1, 10), (1,), (1,)]
```

Reading the finished ids through `model.generator.output` is deprecated (`NNsightDeprecationWarning`); use `tracer.result`.

## Batched generation

Pass a list of prompts, or one `tracer.invoke(...)` per prompt. Shorter prompts are left-padded, and the padding stays in the result:

```python
with model.generate(
    ["The Eiffel Tower is in", "Paris"],
    max_new_tokens=3, do_sample=False,
) as tracer:
    ids = tracer.result.save()
# ids.shape -> torch.Size([2, 10])
# row 1: '<|endoftext|>' * 6 + 'Paris, France,'
```

Decode with `skip_special_tokens=True`, or slice each row by its own token count.

## `trace` vs `generate` vs `pipe`

| Aspect | `model.trace(...)` | `model.generate(...)` | `model.pipe(...)` |
|---|---|---|---|
| Runs | one forward (`__call__`) | the model's `generate` | the whole task pipeline |
| Iterations | 1 | one per new token | pipeline-defined |
| Result | model output | **token ids** | pipeline **records** (text/labels) |
| Sampling | n/a | the checkpoint's `generation_config` (may sample) | that plus the pipeline's `task_specific_params` (may sample) |
| Input | anything the forward takes | text, ids, tensor, encoding | what the task pipeline accepts (text, chat, images) |

`remote=True` runs a generate on NDIF like a trace; see [remote-trace](../remote/remote-trace.md).

## Gotchas

- Within a step, modules must still be read in forward-pass order. Inside an iteration loop an out-of-order read or write can bind to the *next* step instead of raising; see [iteration pitfalls](iteration.md#pitfalls).
