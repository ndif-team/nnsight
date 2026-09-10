---
title: Invoke and Batching
one_liner: Multiple inputs in one trace via `tracer.invoke(...)`, including empty invokes that operate on the full batch.
tags: [usage, batching, invoker]
related: [docs/usage/trace.md, docs/usage/barrier.md, docs/usage/access-and-modify.md, docs/usage/session.md]
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

The batch is assembled from every invoke before any worker starts, so where the
empty invoke is written makes no difference: put it first and it still sees the
rows the invokes below it contribute.

An empty invoke does **not** call `_batch`, so it works even on a base `NNsight`.
It is a fresh worker, which lets it reach a module an earlier invoke already
passed — but within its own block it still reads in forward order, and going back
raises `OutOfOrderError` like anywhere else.

## Statements outside the invoke blocks run first

In invoke mode — `model.trace()` with no input — the trace body is run once
immediately, before any forward, to collect the `tracer.invoke(...)` blocks in it.
Only the invoke bodies then run interleaved with the model. So a statement written
outside an invoke has already run by the time the model starts, whatever it is
written *after*:

```python
with model.trace() as tracer:
    losses = nnsight.save([])
    for t in texts:
        with tracer.invoke(t):
            losses.append(model.transformer.h[5].output[:, -1].sum())
    loss = torch.stack(losses).mean()   # runs first, on the still-empty list
# RuntimeError: stack expects a non-empty TensorList
```

Do the reduction **after** the trace instead, where the list is full:

```python
with model.trace() as tracer:
    parts = nnsight.save([])
    for t in texts:
        with tracer.invoke(t):
            parts.append(model.transformer.h[5].output[:, -1])

torch.stack(parts)          # len(parts) == 2, shape (2, 1, 768)
```

If the reduced value has to feed another trace, put both traces in a
[`session`](session.md).

A `save()` written after the last invoke fails differently and further from its
cause: saved names are pushed back out of the invokes' scopes, and the collection
pass isn't one of them, so `nnsight.save(h)` on the last line of the body leaves
nothing behind and reading `h` afterwards raises `NameError: name 'h' is not
defined`. A save written *before* the invokes does come back, because each invoke's
scope is copied from the surrounding one — which is why the rule looks arbitrary
until you know that the body ran once on its own. Save inside the invoke that
produces the value.

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

### Forward keywords are the batch's, not an invoke's

Everything that reaches the model does so in **one** forward call, so it takes one
set of keywords. Keywords from every invoke are merged into that call and the last
invoke to pass one decides it for every row:

```python
with model.trace() as tracer:
    with tracer.invoke(P1, output_hidden_states=True):
        ...
    with tracer.invoke(P2, output_hidden_states=False):   # wins, for both rows
        ...
# UserWarning: Invokes disagree on the forward keyword 'output_hidden_states' ...
```

Disagreeing invokes warn; agreeing ones (and a keyword only one invoke passes) are
quiet. Anything that has to differ per row belongs in that invoke's *input* rather
than its keywords — passing `{"input_ids": ..., "attention_mask": ...}` as the
invoke's positional input collates per row correctly, while
`attention_mask=` as a keyword replaces the collated mask for the whole batch (on
CUDA, usually as a device-side assert from somewhere inside the model). The same
applies to generation parameters: `max_new_tokens` on an invoke is the batch's.

## Cross-invoke value sharing

Invokes of one trace share the scope they were written in, so a name one invoke
binds is readable in another (on the local runtimes; on `VLLM` each invoke is its
own request and its own scope, see
[Passing values between invokes](../models/vllm.md#passing-values-between-invokes)).
What decides whether the name is bound *yet* is not which module each invoke
touches. It is where each worker has parked.

A worker starts at the top of its block and runs until it asks for a value the
model has not produced yet; that request parks it at a **location**. The model
then runs, and a parked worker resumes when the model reaches its location. So:

> A name bound in invoke A is readable in invoke B once B has parked at a
> location the model reaches **after** A's binding location — or at the same
> location, if A's block is written above B's, since workers parked on one
> location resume in block order.

A worker that has not parked at all is still where it was before the model
started, which is before every binding any other worker makes. Two consequences
account for most cross-invoke `NameError`s:

- **An assignment evaluates its right-hand side before it parks.**
  `module.output[...] = donor` reads `donor` first, then performs the attribute
  access that parks the worker. A block whose first statement is a write has
  parked zero times at the moment it reads `donor`, whichever module it writes
  to.
- **Source order of the invoke blocks is irrelevant.** A binder written below the
  reader works fine, as long as the model reaches the binder's location first.

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

Park-past does not help a consumer that writes, because the write reads the
donor before it parks. Order those with [`tracer.barrier(n)`](barrier.md), which
is also the only option when the consumer has to act at or before the producer's
location — the embedding-transfer case, where there is no earlier module to park
on:

```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)

    with tracer.invoke("The Eiffel Tower is in"):
        clean = model.transformer.h[5].output.clone()
        barrier()                       # clean is bound

    with tracer.invoke("The Colosseum is in"):
        barrier()                       # wait for it
        model.transformer.h[5].output[:] = clean
        logits = model.lm_head.output.save()
```

Make a barrier the default for anything that writes. Park-past is real and it is
cheaper, but it depends on the relative position of two locations in a file that
someone will edit: insert one line above the read and it turns into a `NameError`
with no other warning.

**Use a name that exists nowhere outside the invokes.** Each block starts from a
copy of the surrounding scope and checks that copy first, so if `clean` is already
bound out there, the consumer reads the *old* value. Nothing raises.

## Rows, padding, and getting values back

Rows sit in the batch in invoke **declaration** order: concatenating each
invoke's saved activation in the order the invokes were written reproduces what an
empty invoke sees.

### Which values are scoped to your rows

Scoping is decided by a value's **leading dimension**, not by where the value came
from. A tensor is narrowed to the invoke's rows when its leading dim is the
combined batch size, or a whole multiple of it. The multiple is what covers a model
that folds tokens into the batch axis before a module — every transformers MoE
block does, ahead of its router, so `mlp.gate.output` and `mlp.experts.input` are
`(batch*seq, ...)` and an invoke owns `seq` of those rows for each row of batch it
contributed. Values reached through `.source` below such a reshape are scoped the
same way. A leading dim that is a multiple by *coincidence* — four experts against
two invokes — is sliced too; the shape is all there is to go on.

Anything else is handed to every invoke **whole**. That is correct for a value that
is not batched at all (a causal mask, rotary `cos`/`sin`, a constant built inside
the forward), and wrong for one that is batched along an axis nnsight cannot
recognize: a time-major `(seq, batch, hidden)` activation, a vision tower's
`(patches, hidden)`, a list of per-sequence tensors (containers are not split;
only the tensors inside them are). For those, three things follow:

- a **read** sees the whole batch rather than this invoke's part of it;
- an **in-place edit** applies to every invoke;
- a **replacement** is dropped — there is nowhere to splice it.

Both writes warn, naming the location, its leading dim and the batch size:

```
UserWarning: An in-place edit to `model.visual.merger.output` applies to every
invoke: its leading dimension (256) is neither the batch size (2) nor a multiple
of it, so nnsight served the whole batch rather than this invoke's rows.
```

A read does **not** warn: from a shape alone a value that isn't batched is
indistinguishable from one that is batched on an axis nnsight can't read, and most
values in that state are the former. When the warning does fire, trace that input
on its own — or, if the model's batch layout is genuinely a different shape, give
it a `_batcher_class` that knows it (see
[extending.md](extending.md#4-custom-batcher-for-non-standard-batch-layouts)),
which is how diffusion's guidance doubling and vLLM's flat token axis are handled.

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
positions — the last-token activation of a batched invoke matches the same prompt
run alone — but the model still computes something at each one, and at GPT-2's
layer 6 those values carry a *larger* norm than the real token beside them (3118.5
against 3029.8 for the batch above). A max, mean, or top-k over the sequence axis
that does not mask them out will find them.

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

Pairing that list back up with the prompt list by position misattributes every
result, silently. A sweep where every invoke appends at the *same* module happens
to preserve order — the workers all park on one location and resume in block
order — which is why the pattern looks safe until someone varies the module.

Carry the identity in the value instead. A dict keyed by whatever the loop varies
is order-proof:

```python
with model.trace() as tracer:
    norms = nnsight.save({})
    for layer in [9, 2, 6, 0]:
        with tracer.invoke("The Eiffel Tower is in"):
            norms[layer] = model.transformer.h[layer].output.norm()
# {0: 189.7, 2: 2568.2, 6: 3054.4, 9: 3135.5}
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
invoke's rows alone, for every value the [row rules](#which-values-are-scoped-to-your-rows)
can scope. In a 1 + 2 row batch the two input invokes see `[1, seq, 768]` and
`[2, seq, 768]`, and an empty invoke sees `[3, seq, 768]`.

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
  `with tracer.invoke(...)` block"). A trace whose *only* invoke is an empty one
  contributes no rows and there is nothing to run: on `TransformersModel` that
  surfaces as `IndexError: list index out of range` from the tokenizer.
- **Direct input and invokes are alternatives, not a combination.** Opening
  `tracer.invoke(y)` inside `with model.trace(x)` raises `ValueError: Cannot invoke
  while the model is already running.`, because the direct input already started
  the run. The same error covers nesting two invokes.
- **Reading a cross-invoke name before the binder ran raises `NameError`.** Park
  the reader past the binder's location, or use a barrier — see above.
- **Statements outside the invoke blocks run before the model does.** The body of
  an input-less `trace()` runs once to collect the invokes, so a reduction written
  under the loop sees empty lists, and a `save()` after the last invoke leaves
  nothing behind. Reduce after the trace — see above.
- **Only values whose leading dim matches the batch are scoped to an invoke.**
  Anything else is served whole; a write to one warns and applies batch-wide (or,
  for a replacement, is dropped). See
  [Which values are scoped to your rows](#which-values-are-scoped-to-your-rows).
- **Forward keywords on an invoke are batch-wide.** One forward call takes one set
  of them, so invokes that disagree warn and the last one written wins.
- **A batched write has to keep its rows.** A block owns its invoke's rows of the
  combined batch, and a whole-tensor write is spliced back in as given — nothing
  checks the height. A block that owns rows `0:1` of 2 and assigns a two-row
  tensor builds a 3-row batch: downstream modules see a batch that is no longer
  the model's, rows shift under the other invokes, and the mismatch surfaces in
  some later module or not at all.
- **Batching only narrows with two or more input invokes.** A lone invoke *is* the
  whole batch, so it sees every row untouched — and, since nothing is spliced back,
  a write from a lone invoke (or from an empty invoke) may change the leading dim
  and widen the run.
- **`tracer.stop()` halts the shared forward, not one invoke.** A sibling parked on
  a later location dies with `OutOfOrderError: 'model.lm_head.output.i0' was
  requested but the model already ran past it`. Stop from a batch only when every
  invoke is finished by that point.
- **A batched `.skip()` must cover every row** — skip the module in every invoke or
  none; a shared forward can't run for only the unskipped rows (`ValueError: ...
  cover every row`). See [skip.md](skip.md).

## Related

- [trace.md](trace.md)
- [barrier.md](barrier.md) — cross-invoke synchronization.
- [access-and-modify.md](access-and-modify.md)
- [skip.md](skip.md)
- [session.md](session.md) — sharing values across whole traces.
- [gotchas/cross-invoke.md](../gotchas/cross-invoke.md)
