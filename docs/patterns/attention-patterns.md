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

The attention *block* doesn't expose the probabilities via `.output` — it returns
the value-weighted result. To get the probabilities you reach into the attention
computation with `.source`, which hooks intermediate operations inside the module's
forward (see `docs/usage/source.md`).

For GPT-2, the relevant operation is `attention_interface_1` — the function call
that returns `(attn_output, attn_weights)` (`attention_interface_0` is the
assignment that picks the implementation). Discover it with `print(...source)`.

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
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel(
    "openai-community/gpt2",
    dispatch=True,
    attn_implementation="eager",     # required to expose attention weights
)

prompt = "The cat sat on the"

with model.trace(prompt):
    # attention_interface_1 returns (attn_output, attn_weights).
    attn_out, attn_weights = (
        model.transformer.h[0].attn.source.attention_interface_1.output.save()
    )

print(attn_weights.shape)                 # [batch, n_heads, q_seq, k_seq]
print(attn_weights[0, 0].sum(-1))         # rows sum to 1
```

```
torch.Size([1, 12, 5, 5])
tensor([1., 1., 1., 1., 1.])
```

Discover the operation name by printing `.source` (works outside a trace):

```python
print(model.transformer.h[0].attn.source)
# ...
#  attention_interface_1  -> 71     attn_output, attn_weights = attention_interface(
# ...
```

The trailing `_0` is the occurrence index inside the forward, not a layer index.

## Variations

### All layers in one trace

```python
import nnsight

with model.trace(prompt):
    patterns = nnsight.save([])
    for block in model.transformer.h:
        _, weights = block.attn.source.attention_interface_1.output
        patterns.append(weights)

# len(patterns) == 12; patterns[L].shape == [1, 12, 5, 5]
```

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

- The operation name can vary between transformer versions. `print(...attn.source)`
  first to confirm what's available.
- Request `attention_interface_1` **before** `attn.output` in the same forward —
  the source op runs first, so reading `attn.output` and then the source op is out
  of order (`OutOfOrderError`).
- For Llama / Mistral / Qwen the path is typically
  `model.model.layers[i].self_attn.source....`. **There is no universal op-name
  table — read the model's `forward` (or `print(...source)`) to find it.**
- `[batch, n_heads, q_seq, k_seq]` grows as `seq^2` per head per layer. For long
  contexts, save only the heads / layers you need.

## Related

- `docs/usage/source.md` — how `.source` works (forward rewriting, op names,
  recursive access).
- [per-head-attention](per-head-attention.md) — operating on individual heads.
- [logit-lens](logit-lens.md).
- `tests/test_source.py` — source-tracing tests.
