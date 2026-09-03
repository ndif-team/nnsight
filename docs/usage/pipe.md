---
title: Pipe
one_liner: Run the whole task pipeline under a trace; returns its records (decoded text, labels, ...).
tags: [usage, tracing, pipeline]
related: [docs/usage/generate.md, docs/usage/trace.md, docs/usage/iter-all-next.md]
sources: [src/nnsight/modeling/transformers.py]
---

# Pipe

## What this is for

`model.pipe(input, ...)` runs the model's whole `transformers.pipeline` end to end and returns what the pipeline **postprocesses to** — decoded-text records for text-generation, labels for a classifier, and so on. It is traced like `trace`/`generate`: the block sees every forward the pipeline makes.

The two are split:

- `model.generate(...)` runs the model and returns **token ids** (`tracer.result`). See `docs/usage/generate.md`.
- `model.pipe(...)` runs the pipeline and returns its **records**.

The pipeline tokenizes and collates its own input, and applies the checkpoint's `task_specific_params` (for gpt2 text-generation, that includes `do_sample=True`) — so pipe output is **sampled by default**. Pass `do_sample=False` for greedy/deterministic output.

## Canonical pattern

```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.pipe("The Eiffel Tower is in the city of", max_new_tokens=5, do_sample=False) as tracer:
    records = tracer.result.save()

print(type(records))                       # <class 'list'>
print(records[0]["generated_text"])
# The Eiffel Tower is in the city of Paris, and the E
```

For text-generation the records are a `list` of `{"generated_text": ...}` dicts. Other tasks return their own record shapes (a classifier returns label/score dicts, etc.).

## Interventions still apply

The pipeline runs the real model, so interventions in the block fire on its forward passes:

```python
with model.pipe("The Eiffel Tower is in the city of", max_new_tokens=5, do_sample=False) as tracer:
    model.transformer.h[0].output[:] = 0    # affects generation
    records = tracer.result.save()
```

## generate vs pipe

```python
with model.generate("The Eiffel Tower is in the city of", max_new_tokens=3, do_sample=False) as tracer:
    ids = tracer.result.save()              # torch.Tensor of token ids
with model.pipe("The Eiffel Tower is in the city of", max_new_tokens=3, do_sample=False) as tracer:
    records = tracer.result.save()          # list of decoded-text records
```

| | `model.generate(...)` | `model.pipe(...)` |
|---|---|---|
| Runs | the model's `generate` | the whole task pipeline |
| Returns | token ids (tensor) | pipeline records (text/labels/...) |
| Default sampling | greedy | pipeline's `task_specific_params` (may sample) |
| Input | text, ids, tensor, encoding | what the task pipeline accepts (text, chat, images) |

## Per-step interventions and early exit

A pipeline that generates runs one forward per new token, so `tracer.iter[...]`,
`tracer.all()` and `tracer.stop()` work exactly as they do under `generate`:

```python
import nnsight

with model.pipe("The Eiffel Tower is in the city of", max_new_tokens=5, do_sample=False) as tracer:
    per_step = nnsight.save([])
    for step in tracer.iter[:5]:
        per_step.append(model.lm_head.output[0, -1].argmax(dim=-1))
    records = tracer.result.save()
# len(per_step) == 5
```

The same rule applies: the loop must not ask for a step the pipeline does not
run — see [iter-all-next.md](iter-all-next.md).

```python
with model.pipe("The Eiffel Tower is in the city of", max_new_tokens=5, do_sample=False) as tracer:
    h = model.transformer.h[2].output.save()
    tracer.stop()
# h.shape -> torch.Size([1, 10, 768]); the pipeline's records are not produced
```

## Chat messages

A chat pipeline takes a message list and applies the checkpoint's template
itself, so there is no `apply_chat_template` call to write:

```python
chat = TransformersModel("HuggingFaceTB/SmolLM2-135M-Instruct", dispatch=True)

records = chat.pipe(
    [{"role": "user", "content": "Name one city in France."}],
    max_new_tokens=10, do_sample=False,
)
# [{'generated_text': [{'role': 'user', ...},
#                      {'role': 'assistant', 'content': 'One of the most beautiful cities in France is Paris'}]}]
```

The record's `generated_text` is the whole conversation, the assistant turn
appended. To format a chat prompt yourself and generate token ids instead, apply
the template with the tokenizer and use `model.generate(...)`.

## Batching

Pass a list of prompts; pipe hands them to the pipeline batched with `batch_size` and the pipeline collates them itself:

```python
with model.pipe(["The Eiffel Tower is in", "The Colosseum is in"],
                max_new_tokens=3, do_sample=False) as tracer:
    records = tracer.result.save()          # one record per prompt
```

A single multimodal payload (a VLM's `text=`/`images=` keywords) is passed through as-is; batching several such payloads is refused.

## Called directly (no block)

Without a `with`, `pipe` just runs the pipeline and returns its records:

```python
records = model.pipe("Hello", max_new_tokens=3, do_sample=False)
```

## Gotchas

- Pipe is **sampled by default** for gpt2 (the pipeline's `task_specific_params`). Pass `do_sample=False` for reproducible output.
- Kwargs go to the pipeline (`max_new_tokens`, `do_sample`, task-specific params), not to `model.generate` directly.
- For token ids or the model's own (non-pipeline) settings, use `model.generate(...)`.
- `tracer.result` is the pipeline's records, not ids — decode nothing, index the record.

## Related

- `docs/usage/generate.md`
- `docs/usage/trace.md`
- `docs/usage/iter-all-next.md`
- `docs/usage/save.md`
