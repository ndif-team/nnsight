---
title: Pipe
one_liner: Run the whole task pipeline under a trace; returns its records (decoded text, labels, ...).
tags: [usage, tracing, pipeline]
related: [docs/usage/generate.md, docs/usage/trace.md, docs/usage/iteration.md, docs/usage/save.md]
sources: [src/nnsight/modeling/transformers.py]
---

# Pipe

`model.pipe(input, ...)` runs the model's whole `transformers.pipeline` end to end and returns what the pipeline **postprocesses to** — decoded-text records for text-generation, labels for a classifier, and so on. It is traced like `trace`/`generate`: the block sees every forward the pipeline makes, and interventions in it fire. For token ids or the model's own (non-pipeline) settings, use [generate](generate.md); the full comparison is the table in [generate.md](generate.md#trace-vs-generate-vs-pipe).

The pipeline tokenizes and collates its own input, takes the kwargs (`max_new_tokens`, `do_sample`, task-specific params), and applies the checkpoint's `task_specific_params` — for gpt2 text-generation that includes `do_sample=True`, so pipe output is **sampled by default**. Pass `do_sample=False` for deterministic output.

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

## Per-step interventions and early exit

Per-step `tracer.iter[...]`, `tracer.all()` and `tracer.stop()` work exactly as under `generate`, one step per new token ([iteration](iteration.md), [stop-and-early-exit](stop-and-early-exit.md)). A `stop()` means the pipeline's records are never produced.

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

Without a `with`, `pipe` just runs the pipeline and returns its records: `records = model.pipe("Hello", max_new_tokens=3, do_sample=False)`.
