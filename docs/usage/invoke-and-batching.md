---
title: Invoke and Batching
one_liner: Multiple inputs in one trace via `tracer.invoke(...)`, including empty invokes that operate on the full batch.
tags: [usage, batching, invoker]
related: [docs/usage/trace.md, docs/usage/barrier.md, docs/usage/access-and-modify.md, docs/usage/session.md, docs/usage/skip.md]
sources: [src/nnsight/intervention/tracer.py, src/nnsight/intervention/batching.py, src/nnsight/intervention/envoy.py]
---

# Invoke and Batching

`with model.trace() as tracer:` may contain several `with tracer.invoke(x):`
blocks. Their inputs are combined into a **single batched forward**, and each
block's interventions see only *its* rows of every activation. Each block runs as
its own worker, resumed in the order the model reaches what it asked for, not the
order the blocks were written.

- **Input invoke**: `tracer.invoke(prompt)` contributes rows to the batch; its
  reads and edits are narrowed to those rows.
- **Empty invoke**: `tracer.invoke()` contributes no input and sees the **entire**
  combined batch.

With a single input, pass it to `.trace(x)` directly; that is an implicit single
invoke.

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

Each invoke's `.output` carries only that invoke's rows. A batched result matches
the same prompt run alone up to floating-point kernel selection, not bit-for-bit;
negligible in fp32, but in **bf16** it can reorder a head ranking, so read a
ranking metric through an fp32 head.

## Comparing prompts: one trace, not a loop of traces

Clean vs corrupt, baseline vs ablated, a sweep over a prompt set: write it as N
invokes in one trace rather than `for p in prompts: with model.trace(p): ...`.
Setup is paid once, an empty invoke applies one intervention to every prompt,
invokes can share values, and with `remote=True` it is one job instead of N.

```python
with model.trace() as tracer:
    with tracer.invoke(prompt):
        baseline = model.lm_head.output[:, -1, :].save()
    with tracer.invoke(prompt):
        model.transformer.h[9].mlp.output[:] = 0
        ablated = model.lm_head.output[:, -1, :].save()
# baseline -> ' Paris', ablated -> ' London' for "The Eiffel Tower is in the city of"
```

Key a sweep's results by what the loop varies (a dict), not by list position —
see [Save by identity](#save-by-identity-not-by-position).

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

The batch is assembled from every invoke before any worker starts, so where the
empty invoke is written makes no difference: put it first and it still sees the
rows the invokes below it contribute.

An empty invoke works even on a base `NNsight`, which cannot batch input invokes.
It is a fresh worker, which lets it reach a module an earlier invoke already
passed — but within its own block it still reads in forward order, and going back
raises `OutOfOrderError` like anywhere else.

## Input formats

Every input format a forward accepts is batchable and can be mixed across invokes
— string, list of strings (one row each), token-id list, 1-D tensor,
pre-tokenized encoding:

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
binds is readable in another (on the local runtimes; on `VLLM` each invoke is its
own scope, see
[Passing values between invokes](../models/vllm.md#passing-values-between-invokes)).
Whether the name is bound *yet* depends on where each worker has parked: a worker
runs until it asks for a value the model has not produced, parks at that
**location**, and resumes when the model reaches it.

> A name bound in invoke A is readable in invoke B once B has parked at a
> location the model reaches **after** A's binding location — or at the same
> location, if A's block is written above B's.

The source order of the blocks is otherwise irrelevant. A worker that has not
parked yet is before every binding, and **an assignment evaluates its right-hand
side before it parks**: `module.output[...] = donor` reads `donor` first, so a
block whose first statement is a write has never parked when it reads `donor`.

### Reading a value: park past the binder

For a read-only consumer, one extra `.output` access is the whole mechanism:

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        clean = model.transformer.h[5].output

    with tracer.invoke("The Colosseum is in"):
        model.transformer.h[6].output   # park past h[5], where `clean` is bound
        total = clean.sum().save()
```

Drop the `h[6].output` line and the read raises
`NameError: name 'clean' is not defined`. Park on a location the model reaches
*before* `h[5]` and it raises the same thing.

### Writing a value: use a barrier

Park-past does not help a consumer that writes, or one that must act at or before
the producer's location. Order those with [`tracer.barrier(n)`](barrier.md), and
make a barrier the default for anything that writes: park-past depends on the
relative position of two lines, and inserting one above the read turns it into a
`NameError`.

**Use a name that exists nowhere outside the invokes.** Each block starts from a
copy of the surrounding scope and checks that copy first; a sibling's bind reaches
it only when the name is not already in the copy. So a name left over from an
earlier notebook cell or the enclosing function shadows the one the producer binds,
and nothing raises — the write lands the wrong tensor:

```python
donor = torch.zeros(768)                 # left over from an earlier cell

with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("The Eiffel Tower is in"):
        donor = model.transformer.h[5].output[:, -1].clone()
        barrier()
    with tracer.invoke("The Colosseum is in"):
        barrier()
        model.transformer.h[5].output[:, -1] = donor    # the zeros, not the clone
```

Rename it (`eiffel_h5`) and the patch takes. A cross-invoke handoff with no error
*and* no effect is this, not a barrier problem — a missing barrier raises.

## Rows, padding, and getting values back

Rows sit in the batch in invoke **declaration** order: concatenating each
invoke's saved activation in the order the invokes were written reproduces what an
empty invoke sees.

### Every invoke is padded to the whole batch's length

`TransformersModel._batch` tokenizes with left padding and pads every invoke to
the longest input in the batch. A one-token prompt batched against a fourteen-token
one has activations of shape `[1, 14, 768]`, thirteen of which are pad:

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        short = model.transformer.h[6].output.save()
    with tracer.invoke("The Eiffel Tower is located in the beautiful and historic city of"):
        long = model.transformer.h[6].output.save()
# short.shape == long.shape == (1, 14, 768)
```

The same prompt alone gives `(1, 1, 768)`. So **an absolute position index means a
different token depending on what else is in the batch**: `output[:, 0]` is the
first real token in a solo trace and a pad token in this one, and
`output[:, SUBJECT]` lands on the subject only when every prompt in the batch has
the same length.

Indexing from the right (`[:, -1]`) is stable, because the padding is on the left.
For any other position, work out the offset from that invoke's own token count, or
from the attention mask.

Pad positions are not blank. The attention mask keeps them from affecting the real
positions, but the model still computes values there, often with a *larger* norm
than the real tokens. A max, mean or top-k over the sequence axis that does not
mask them out will find them.

A batched `generate` returns the padding too. With `max_new_tokens=5` and inputs
of 5 and 18 tokens, `tracer.result` is `[2, 23]`, and decoding the short row prints
thirteen `<|endoftext|>` ahead of its prompt. Slice a row by its own token count
rather than by one offset shared across the batch.

### Save by identity, not by position

Values appended to one list from several invokes arrive in the order the model
reaches them, which is not the order the invokes were written:

```python
outs = nnsight.save([])
for i, layer in enumerate([9, 2, 6, 0]):
    with tracer.invoke(f"prompt {i}"):
        outs.append((i, model.transformer.h[layer].output.norm()))
# outs comes back keyed [3, 1, 2, 0] — layers 0, 2, 6, 9, in model order
```

Pairing that list with the prompt list by position silently misattributes every
result. (Appending at the *same* module in every invoke happens to preserve order,
which is why the pattern looks safe until someone varies the module.) Key the
results by whatever the loop varies:

```python
with model.trace() as tracer:
    norms = nnsight.save({})
    for layer in [9, 2, 6, 0]:
        with tracer.invoke("The Eiffel Tower is in"):
            norms[layer] = model.transformer.h[layer].output.norm()
```

For the same reason, give each invoke its own saved name. Three invokes that each
bind `out = ....save()` leave one value behind, from whichever ran last.

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

Everything an invoke reads is narrowed the same way: `.input`, `.output`,
`tracer.result`, and a `tracer.cache(...)` opened inside the block all carry that
invoke's rows alone. In a 1 + 2 row batch the two input invokes see
`[1, seq, 768]` and `[2, seq, 768]`, and an empty invoke sees `[3, seq, 768]`.

## Batching a custom model

Base `NNsight` runs a single invoke, but two or more input invokes raise
`NotImplementedError: ... does not support batching multiple invokes`. Implement
`_batch_size` / `_batch` as in
[developing/extending.md](../developing/extending.md#1-batching-_batch_size--_batch).
Empty invokes always work.

## Gotchas

- **A trace whose only invoke is empty has nothing to run**: it contributes no
  rows, and on `TransformersModel` surfaces as `IndexError: list index out of
  range` from the tokenizer.
- **Direct input and invokes are alternatives, not a combination.** Opening
  `tracer.invoke(y)` inside `with model.trace(x)` raises `ValueError: Cannot invoke
  while the model is already running.`, because the direct input already started
  the run. The same error covers nesting two invokes.
- **A batched write has to keep its rows.** A whole-tensor write is spliced back
  into the invoke's rows as given, and nothing checks the height: a block owning
  one row of two that assigns a two-row tensor builds a 3-row batch, rows shift
  under the other invokes, and the mismatch surfaces later or not at all.
- **Batching only narrows with two or more input invokes.** A lone invoke *is* the
  whole batch, so it sees every row untouched — and, since nothing is spliced back,
  a write from a lone invoke (or from an empty invoke) may change the leading dim
  and widen the run.
- **`tracer.stop()` in one invoke ends the whole batch** — see
  [stop-and-early-exit](stop-and-early-exit.md). **A batched `.skip()` must cover
  every row** — see [skip](skip.md).
