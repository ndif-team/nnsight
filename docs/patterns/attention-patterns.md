---
title: Attention Patterns
one_liner: Read attention probability matrices from `attn.output[1]` (or `.source` for the raw softmax) to see what each head attends to.
tags: [pattern, interpretability, attention, source-tracing]
related: [docs/usage/source.md, docs/patterns/per-head-attention.md, docs/patterns/logit-lens.md]
sources: [src/nnsight/intervention/source.py, src/nnsight/intervention/envoy.py, tests/test_source.py]
---

# Attention Patterns

## What this is for

The attention pattern (the softmax-normalized matrix `A` in
`softmax(QK^T / sqrt(d)) V`) is the most direct read on what a head is looking at —
use it to find induction, copy, previous-token or name-mover heads, to compute
per-head metrics (entropy, attention to a position), or to check a hypothesis from
another method.

Under `attn_implementation="eager"` the attention module returns the matrix
alongside its output, so `attn.output[1]` is the pattern and
`model.output.attentions` is every layer's at once. Reach for `.source` when you
want the raw `softmax(QK^T/√d)` itself — before the dtype cast and dropout — or when a
module does not hand the weights back.

**`eager` is required.** GPT-2 defaults to `sdpa`; under `sdpa` or
`flash_attention_2` the weights element is `None` and `model.output.attentions` is
the empty tuple — no warning, and the first index raises
`IndexError: tuple index out of range` somewhere downstream.

## Canonical pattern

```python
import torch
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel(
    "openai-community/gpt2",
    dispatch=True,
    attn_implementation="eager",
)

prompt = "The cat sat on the"

with model.trace(prompt):
    weights = model.transformer.h[0].attn.output[1].save()

print(weights.shape)                      # [batch, n_heads, q_seq, k_seq]

assert weights.shape == (1, 12, 5, 5)
assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)))   # rows sum to 1
assert (torch.triu(weights, diagonal=1) == 0).all()      # causal mask, exactly zero
```

Every layer at once, from the model's own return value:

```python
with model.trace(prompt, output_attentions=True):
    attentions = model.output.attentions.save()

assert len(attentions) == 12
assert torch.equal(attentions[0], weights)
```

To pick layers without `output_attentions=True`, append `block.attn.output[1]` for
each block you want to an `nnsight.save([])` list inside the trace.

## Raw softmax weights via `.source`

For GPT-2, the operation inside the attention forward that returns
`(attn_output, attn_weights)` is `attention_interface_1` (`attention_interface_0` is
the assignment that picks the implementation). It resolves at run time to
`eager_attention_forward`, so you can chain `.source` again to reach the softmax:

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

Op names vary between `transformers` versions and model classes — print `.source`
to discover the names for your version:

```python
print(model.transformer.h[0].attn.source)          # works outside a trace
# ...
#  attention_interface_1  -> 66     attn_output, attn_weights = attention_interface(
# ...
```

The trailing number is the occurrence index inside the forward, not a layer index.
Recursive `.source` (the inner function's ops) only resolves **inside a trace**. For
Llama / Mistral / Qwen the path is `model.model.layers[i].self_attn.source...`; the
inner names carry over, because those families also go through
`eager_attention_forward`. See [source](../usage/source.md).

To edit attention *output* per head rather than read weights, see
[per-head-attention](per-head-attention.md). To compare patterns across prompts, run
them as invokes in one trace — see [invoke-and-batching](../usage/invoke-and-batching.md).

## Interpretation tips

- **Shape**: `weights[b, h, i, j]` is how much position `i` attends to `j` in head
  `h`.
- **Watch the BOS / position-0 attention.** Many heads dump mass on position 0
  ("attention sink"); that often means "this head isn't engaged here".
- **Diagonal** = self / positional behavior. **Off-diagonal** = information movement.
- **Induction heads** attend from position `i` to the token after the previous
  occurrence of the token at `i-1`.

## Gotchas

- **`print(...source)` lists operations the forward never runs.** GPT-2's
  attention prints 50 operations, 22 of which are on a cross-attention branch a
  decoder-only model never takes. Asking for one raises
  `OutOfOrderError: '...transpose_0.output.i0' was requested but the model already
  ran past it`, which reads like an ordering bug you do not have. The real key
  transpose is `transpose_2`, one character away from the dead `transpose_0`.
- **Inner before outer.** Source ops run before the module returns, and an inner op
  runs before the call that contains it. Read `...attention_interface_1.source.<op>`
  before `attention_interface_1.output`, and both before `attn.output`; the reverse
  raises `OutOfOrderError`.
- **The raw softmax is float32; the returned weights are the model's dtype.** On a
  bf16 checkpoint `nn_functional_softmax_0.output` and `attn.output[1]` fail
  `torch.equal` and agree to `2e-3` after a cast. On float32 models they are the
  same tensor.
- **Memory.** Patterns grow as `seq^2` per head per layer. For long contexts save
  only the heads / layers you need, and wrap read-only capture in `torch.no_grad()`
  — a saved pattern otherwise pins the whole forward graph.
- Unpacking a tuple across `.save()` (`a, b = op.output.save()`) leaves both names
  unbound. Save the tuple to one name and index it afterwards.

## Related

- [source](../usage/source.md) — how `.source` works (forward rewriting, op names,
  recursive access).
- [per-head-attention](per-head-attention.md) — operating on individual heads.
- `nnterp`'s `StandardizedTransformer(..., enable_attention_probs=True)` gives
  `model.attention_probabilities[i]` with no per-architecture names at all.
- `tests/test_source.py` — source-tracing tests.
