---
title: Invoke and Batching
one_liner: Multiple inputs in one trace via `tracer.invoke(...)`, including empty invokes that operate on the full batch.
tags: [usage, batching, invoker]
related: [docs/usage/trace.md, docs/usage/barrier.md, docs/usage/access-and-modify.md]
sources: [src/nnsight/intervention/tracer.py, src/nnsight/intervention/batching.py, src/nnsight/intervention/envoy.py]
---

# Invoke and Batching

## What this is for

`with model.trace() as tracer:` may contain several `with tracer.invoke(x):`
blocks. Their inputs are combined into a **single batched forward**, and each
block's interventions see only *its* rows of every activation.

Each invoke's block runs as its own **worker** (a greenlet `Mediator`). Workers
resume in the order the model reaches what each asked for, not the order they were
written — that is what makes them a batch rather than a sequence.

Two kinds of invoke:

- **Input invoke**: `tracer.invoke(prompt)` — contributes rows to the batch. Its
  reads/edits are scoped (narrowed) to those rows.
- **Empty invoke**: `tracer.invoke()` — no input, no row scoping. Sees the
  **entire** combined batch.

## When to use / when not to use

- Use multiple input invokes to run several prompts in one forward pass. Requires
  a model that implements `_batch_size` / `_batch` (`TransformersModel` does;
  base `NNsight` does not — see below).
- Use an empty invoke to run logic over the whole combined batch.
- Use a single positional arg on `.trace(x)` when you only have one input — it is
  an implicit single invoke over the whole batch.

## Canonical pattern

```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        out_paris = model.lm_head.output[:, -1].save()

    with tracer.invoke("The Colosseum is in"):
        out_rome = model.lm_head.output[:, -1].save()
```

Each invoke's `.output` carries only that invoke's row(s). A batched last-token
logit from an invoke matches the same prompt run on its own **up to floating-point
kernel selection** — not bit-for-bit. Measured drift on fp32 CUDA is ~2e-4 (GPT-2)
and ~2e-5 (Llama-3.2-1B); it is a property of torch, not nnsight (nnsight's logits
are `torch.equal` to raw HuggingFace at every batch size), it does not grow with
batch size, and reruns are bit-identical. It is driven by the *shape* the kernel
sees, so lengthening a sequence perturbs earlier positions by the same magnitude.
In fp32 this is far below any real effect size; in **bf16** it is comparable to the
metric quantum and can reorder a head ranking, so read a ranking metric through an
fp32 head.

## Batched input (single invoke, list of strings)

```python
with model.trace(["Hello", "World"]):
    logits = model.lm_head.output.save()   # shape: [2, seq, vocab]
```

`TransformersModel._batch_size` tokenizes the list with left-padding and
reports one row per prompt.

## Empty invokes — operate on the full batch

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        a = model.lm_head.output[:, -1].save()      # 1 row
    with tracer.invoke(["Rome", "Berlin"]):
        b = model.lm_head.output[:, -1].save()      # 2 rows

    with tracer.invoke():                            # whole batch: 3 rows
        whole = model.lm_head.output[:, -1].save()
# whole.shape[0] == 3, and torch.cat([a, b]) == whole
```

An empty invoke does **not** call `_batch`, so it works even on a base `NNsight`.

## Mixed input formats

Every input format a forward accepts is batchable, and formats can be mixed
across invokes — string, list of strings, token-id list, 1-D tensor, pre-tokenized
encoding:

```python
ids = model.tokenizer("Madison Square Garden is in").input_ids
with model.trace() as tracer:
    with tracer.invoke(ids):                 # token-id list -> 1 row
        pass
    with tracer.invoke(["a b c", "d e"]):    # 2 rows
        pass
    with tracer.invoke():
        whole = model.lm_head.output[:, -1].save()   # 3 rows total
```

Tokenizer kwargs on an invoke apply to that invoke's tokenization (not the
model): `tracer.invoke("word " * 50, truncation=True, max_length=4)`.

## Cross-invoke value sharing

Invokes of one trace share the scope they were written in, so a name one invoke
binds is readable by a later invoke — no config flag required (on the local
runtimes; on `VLLM` each invoke is its own request and its own scope, see
[Passing values between invokes](../models/vllm.md#passing-values-between-invokes)):

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        clean = model.transformer.h[5].output

    with tracer.invoke("The Colosseum is in"):
        # Park past where `clean` was bound (h[5]) before reading it.
        model.transformer.h[6].output   # advances this worker past h[5]
        patched = (clean.sum()).save()
```

The reader must reach the name **after** the binder produced it. Every worker
runs up to its first park before the model runs, so a read is only safe once the
reader has parked on a location the model reaches *after* the binder's. Reading a
cross-invoke name too early raises `NameError` (see Gotchas). When both invokes
need to hand a value across the *same* module, use `tracer.barrier(n)` — see
[barrier.md](barrier.md).

## Per-invoke iteration and cache

Each invoke keeps its own iteration counter and its own cache scope:

```python
with model.generate(max_new_tokens=5, do_sample=False) as tracer:
    with tracer.invoke("Madison Square Garden is in"):
        first = nnsight.save([])
        for _ in tracer.iter[1:3]:
            first.append(model.lm_head.output[0][-1].argmax(dim=-1))
    with tracer.invoke("Madison Square Garden is in"):
        second = nnsight.save([])
        for _ in tracer.iter[:3]:
            second.append(model.lm_head.output[0][-1].argmax(dim=-1))
# len(first) == 2, len(second) == 3
```

A `tracer.cache(...)` opened inside an invoke records that invoke's rows only.

## Implementing batching for a custom model

Base `NNsight` runs a single invoke fine, but batching two or more raises
`NotImplementedError`. To support it, override `_batch_size` (row count) and
`_batch` (combine the invokes):

```python
import torch
from nnsight import Envoy

class BatchEnvoy(Envoy):
    def _batch_size(self, *inputs, **kwargs):
        # rows this invoke contributes (0 if it has no data)
        return inputs[0].shape[0] if inputs else 0

    def _batch(self, invokes, fn):
        # invokes: list of (inputs, kwargs); return (args, kwargs) for fn
        return (torch.cat([inputs[0] for inputs, _ in invokes]),), {}
```

## Gotchas

- **Base `NNsight` cannot batch multiple input invokes** — two input invokes raise
  `NotImplementedError: ... does not support batching multiple invokes`. Empty
  invokes always work.
- **A trace with no direct input needs at least one invoke.** `with model.trace():`
  with an empty body raises `ValueError` ("trace() needs an input, or at least one
  `with tracer.invoke(...)` block").
- **Invokes cannot nest.** Opening an invoke while the model is running raises
  `ValueError: Cannot invoke while the model is already running.`
- **Reading a cross-invoke name before the binder ran raises `NameError`.** Park
  the reader past the module the value came from first.
- **Batching only narrows with two or more non-empty invokes.** A lone invoke *is*
  the whole batch, so it sees every row untouched.
- **A batched `.skip()` must cover every row** — skip the module in every invoke or
  none; a shared forward can't run for only the unskipped rows (`ValueError: ...
  cover every row`). See [skip.md](skip.md).

## Related

- [trace.md](trace.md)
- [barrier.md](barrier.md) — cross-invoke synchronization.
- [access-and-modify.md](access-and-modify.md)
- [skip.md](skip.md)
