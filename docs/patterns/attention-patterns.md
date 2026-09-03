---
title: Attention Patterns
one_liner: Extract attention probability matrices from transformer blocks via `.source` to visualize what each head attends to.
tags: [pattern, interpretability, attention, source-tracing]
related: [docs/usage/source.md, docs/patterns/per-head-attention.md, docs/patterns/logit-lens.md]
sources: [src/nnsight/intervention/source.py, src/nnsight/intervention/envoy.py, tests/test_source.py]
---

# Attention Patterns

## What this is for

The attention pattern (the softmax-normalized matrix `A` in
`softmax(QK^T / sqrt(d)) V`) is the most direct read on what a head is "looking
at". Visualizing per-head attention probabilities reveals induction heads, copy
heads, name-mover heads, and so on.

Under `attn_implementation="eager"` the attention module returns the matrix
alongside its value-weighted output, so `attn.output[1]` is the pattern and
`model.output.attentions` is every layer's at once. Reach for `.source` when you
want the raw `softmax(QK^T/√d)` itself — before the dtype cast and the dropout —
or when a module does not hand the weights back.

For GPT-2, the source operation that returns `(attn_output, attn_weights)` is
`attention_interface_1` (`attention_interface_0` is the assignment that picks the
implementation), and the softmax inside it is `nn_functional_softmax_0`. Both
names are `transformers` 5.15 on nnsight 0.8; discover yours with
`print(...source)`.

## When to use

- Visualizing what each head attends to on a prompt.
- Identifying induction heads, copy heads, etc.
- Confirming an attention-pattern hypothesis from another method.
- Per-head metrics (entropy, max attention, attention to a specific position).

## Canonical pattern

You need `attn_implementation="eager"` to get attention weights. GPT-2 defaults to
`sdpa`, which returns `None` for the weights (SDPA/FlashAttention never materialize
them).

```python
import torch
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel(
    "openai-community/gpt2",
    dispatch=True,
    attn_implementation="eager",     # required to expose attention weights
)

prompt = "The cat sat on the"

with model.trace(prompt):
    weights = model.transformer.h[0].attn.output[1].save()

print(weights.shape)                      # [batch, n_heads, q_seq, k_seq]
print(weights[0, 0].sum(-1))              # rows sum to 1

assert weights.shape == (1, 12, 5, 5)
assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)))
assert (torch.triu(weights, diagonal=1) == 0).all()      # causal mask, exactly zero
```

```
torch.Size([1, 12, 5, 5])
tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000], grad_fn=<SumBackward1>)
```

Every layer at once, from the model's own return value:

```python
with model.trace(prompt, output_attentions=True):
    attentions = model.output.attentions.save()

assert len(attentions) == 12
assert torch.equal(attentions[0], weights)
```

Both need `eager`. Under `sdpa` or `flash_attention_2` the weights element is
`None` and `model.output.attentions` is the empty tuple — no warning, and the
first index raises `IndexError: tuple index out of range` somewhere downstream.

Discover the operation name by printing `.source` (works outside a trace):

```python
print(model.transformer.h[0].attn.source)
# ...
#  attention_interface_1  -> 71     attn_output, attn_weights = attention_interface(
# ...
```

The trailing number is the occurrence index inside the forward, not a layer
index; assignments and calls share the sequence.

## Variations

### Selected layers in one trace

`model.output.attentions` gives every layer. To pick layers, or to skip
`output_attentions=True`, accumulate them yourself:

```python
import nnsight

with model.trace(prompt):
    patterns = nnsight.save([])
    for block in model.transformer.h[4:8]:
        patterns.append(block.attn.output[1])

assert len(patterns) == 4 and patterns[0].shape == (1, 12, 5, 5)
```

Unpack the source operation's tuple the same way — `_, w = op.output` — but never
across a `.save()`: `a, b = op.output.save()` leaves both names unbound after the
trace. Save the tuple to one name and index it afterwards.

### Raw softmax weights (recursive source)

`attention_interface_1` resolves at run time to a plain Python function
(`eager_attention_forward`), so you can chain `.source` again to reach the raw
softmax — before the dtype cast and dropout:

```python
with model.trace(prompt):
    softmax_w = (
        model.transformer.h[0].attn.source
        .attention_interface_1
        .source.nn_functional_softmax_0
        .output.save()
    )
# softmax_w.shape == [1, 12, 5, 5]
```

Recursive `.source` only works **inside a trace** (the called function is resolved
from the live value). Print the inner ops with
`print(model.transformer.h[0].attn.source.attention_interface_1.source)` inside a
trace. See `docs/usage/source.md`.

### Average attention across a batch

```python
import nnsight

prompts = ["The cat sat on the", "A dog ran under the", "The bird flew over the"]

with model.trace() as tracer:
    pieces = nnsight.save([])
    for p in prompts:
        with tracer.invoke(p):
            _, w = model.transformer.h[5].attn.source.attention_interface_1.output
            pieces.append(w)

# each pieces[i] is [1, 12, seq, seq]; pad/clip if seq lengths differ.
```

### Patching the attention output (not the weights)

To *modify* attention behavior, replace the operation's output. `.output` is a
tuple `(attn_output, attn_weights)`; rebuild it and assign the whole tuple:

```python
import torch

with model.trace(prompt):
    out = model.transformer.h[0].attn.source.attention_interface_1.output
    new = (torch.zeros_like(out[0]),) + tuple(out[1:])   # zero the attn output
    model.transformer.h[0].attn.source.attention_interface_1.output = new
    logits = model.lm_head.output.save()
```

See `tests/test_source.py` for tested source-patching examples.

## Interpretation tips

- **Shape**: weights are `[batch, n_heads, q_seq, k_seq]`. `weights[b, h, i, j]` is
  "how much position `i` attends to `j` in head `h`". Rows sum to 1 (causal mask
  zeros the upper triangle).
- **Watch the BOS / position-0 attention.** Many heads dump mass on position 0
  ("attention sink"); that often means "this head isn't engaged here".
- **Diagonal** = self / positional behavior. **Off-diagonal** = information movement.
- **Induction heads** attend from position `i` to the token after the previous
  occurrence of the token at `i-1`.
- **Compare across prompts**, not just within one.
- **Different implementations expose different things.** With `sdpa` /
  `flash_attention_2` the weights element is `None`. Use `eager`.

## Gotchas

- **`print(...source)` lists operations the forward never runs.** GPT-2's
  attention prints 50 operations, 22 of which are on a cross-attention branch a
  decoder-only model never takes. Asking for one raises
  `OutOfOrderError: '...transpose_0.output.i0' was requested but the model already
  ran past it`, which reads like an ordering bug you do not have. The real key
  transpose is `transpose_2`, one character away from the dead `transpose_0`.
- **Drill before you read.** Reaching an inner operation *after* reading the outer
  one is out of order — the call has already returned:

  ```python
  with model.trace(prompt):
      w = model.transformer.h[0].attn.source.attention_interface_1.output[1].save()
      sm = (model.transformer.h[0].attn.source.attention_interface_1
            .source.nn_functional_softmax_0.output.save())
  # OutOfOrderError: '...attention_interface_1.fn.i0' was requested but the model
  # already ran past it
  ```

  Reversing the two statements works.
- **The raw softmax is float32; the returned weights are the model's dtype.** On a
  bf16 checkpoint `nn_functional_softmax_0.output` and `attn.output[1]` fail
  `torch.equal` and agree to `2e-3` after a cast. On float32 models they are the
  same tensor.
- Request `attention_interface_1` **before** `attn.output` in the same forward —
  the source op runs first, so reading `attn.output` and then the source op is out
  of order (`OutOfOrderError`).
- Operation names vary between `transformers` versions and model classes. For
  Llama / Mistral / Qwen the path is `model.model.layers[i].self_attn.source....`;
  the inner names carry over, because those families and GPT-2 both go through
  `eager_attention_forward`. **There is no universal op-name table — print
  `.source` to find yours.**
- `[batch, n_heads, q_seq, k_seq]` grows as `seq^2` per head per layer. For long
  contexts, save only the heads / layers you need, and wrap read-only capture in
  `torch.no_grad()` — a trace runs with autograd on, and a saved pattern otherwise
  pins the whole forward graph.

## Related

- `docs/usage/source.md` — how `.source` works (forward rewriting, op names,
  recursive access).
- [per-head-attention](per-head-attention.md) — operating on individual heads.
- [logit-lens](logit-lens.md).
- `nnterp`'s `StandardizedTransformer(..., enable_attention_probs=True)` gives
  `model.attention_probabilities[i]` with no per-architecture names at all.
- `tests/test_source.py` — source-tracing tests.
