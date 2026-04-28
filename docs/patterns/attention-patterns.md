---
title: Attention Patterns
one_liner: Extract attention probability matrices from transformer blocks via `.source` to visualize what each head attends to.
tags: [pattern, interpretability, attention, source-tracing]
related: [docs/usage/source.md, docs/patterns/per-head-attention.md, docs/patterns/logit-lens.md]
sources: [src/nnsight/intervention/source.py:610, src/nnsight/intervention/envoy.py:223, tests/test_lm.py:691]
---

# Attention Patterns

## What this is for

The attention pattern (the softmax-normalized matrix `A` in `softmax(QK^T / sqrt(d)) V`) is the most direct read on what an attention head is "looking at". Visualizing per-layer per-head attention probabilities reveals induction heads, copy heads, syntactic heads, name-mover heads, and so on.

In nnsight, attention probabilities are not directly exposed by `.output` of an attention block - the block returns the *value-weighted* output. To get the probabilities themselves you reach into the attention computation using `.source`, which hooks intermediate operations inside the module's forward method (see `docs/usage/source.md`).

For HuggingFace transformer attention, the relevant operation is typically named `attention_interface_0` (the function call that computes the QKV-weighted result) and its inner `torch_nn_functional_scaled_dot_product_attention_0` (the actual SDPA call).

## When to use

- Visualizing what each head attends to on a given prompt.
- Identifying induction heads, copy heads, etc.
- Confirming an attention pattern hypothesis discovered by another method.
- Computing per-head metrics (entropy, max attention, attention to a specific position).

## Canonical pattern

To get attention weights you usually need `attn_implementation="eager"`. Modern HF defaults to `sdpa` or `flash_attention_2`, which **do not return attention weights** as part of their output (FlashAttention computes them implicitly and never materializes them; SDPA returns `None` for weights unless `output_attentions=True` is requested through the eager path).

```python
from nnsight import LanguageModel

# eager attention so attention weights are exposed.
model = LanguageModel(
    "openai-community/gpt2",
    device_map="auto",
    dispatch=True,
    attn_implementation="eager",
)

prompt = "The cat sat on the"

with model.trace(prompt):
    # `attention_interface_0` returns (attn_output, attn_weights).
    attn_out, attn_weights = (
        model.transformer.h[0].attn.source.attention_interface_0.output.save()
    )

# attn_weights shape with eager attention: [batch, n_heads, seq, seq]
print(attn_weights.shape)
```

To discover the operation name on your model, print `.source` outside a trace:

```python
print(model.transformer.h[0].attn.source)
# ...
# attention_interface_0  -> 66    attn_output, attn_weights = attention_interface(...)
# ...
```

The exact name (`attention_interface_0`, `attention_interface_1`, ...) reflects the operation's index inside the forward method - the leading `_0` is iteration, not a layer index.

## Variations

### All layers in one trace

```python
with model.trace(prompt):
    patterns = []
    for block in model.transformer.h:
        _, weights = block.attn.source.attention_interface_0.output
        patterns.append(weights.save())

# patterns[L].shape == [batch, n_heads, seq, seq]
```

### Inside the SDPA call (recursive source)

If you want the *raw* QK softmax (e.g. before any masking quirks), recurse into the inner SDPA function:

```python
with model.trace(prompt):
    sdpa_out = (
        model.transformer.h[0].attn.source
        .attention_interface_0
        .source
        .torch_nn_functional_scaled_dot_product_attention_0
        .output.save()
    )
```

The output shape and what is returned depends on the SDPA backend. With `attn_implementation="eager"`, the wrapping `attention_interface_0` is the simpler and more reliable target.

### Average attention across a batch

```python
prompts = ["The cat sat on the", "A dog ran under the", "The bird flew over the"]

with model.trace() as tracer:
    pieces = []
    for p in prompts:
        with tracer.invoke(p):
            _, w = model.transformer.h[5].attn.source.attention_interface_0.output
            pieces.append(w.save())

# Each pieces[i] is [1, n_heads, seq, seq]; pad/clip if seq lengths differ.
```

### Patching the attention output (not the weights)

If you want to *modify* attention behavior, patch the operation's output rather than its weights:

```python
with model.trace(prompt):
    out = model.transformer.h[0].attn.source.attention_interface_0.output
    new = (torch.zeros_like(out[0]),) + out[1:]    # zero the attn output, keep weights
    model.transformer.h[0].attn.source.attention_interface_0.output = new
    logits = model.lm_head.output.save()
```

See `tests/test_lm.py:691` and `tests/test_source.py` for tested examples of source patching.

## Interpretation tips

- **Shape**: with eager attention, weights are `[batch, n_heads, q_seq, k_seq]`. `weights[b, h, i, j]` is "how much position `i` attends to position `j` in head `h`". Rows sum to 1 (with causal masking, the upper triangle is 0).
- **Look at the BOS token attention.** Many heads dump probability mass on position 0 ("attention sink"). Strong attention to BOS often means "this head is not engaged on this prompt."
- **Diagonal patterns** = self-attention / position-encoding behavior. **Off-diagonal** = real information movement.
- **Induction heads** show a characteristic pattern on repeated tokens - position `i` attends to the position one after the previous occurrence of token at `i-1`.
- **Compare across prompts**, not just within one. A head's behavior is more robustly characterized by its *consistent* attention across many prompts.
- **Different attention implementations expose different things.** With `sdpa` or `flash_attention_2`, the second tuple element of `attention_interface_0` may be `None`. Use `eager`.

## Gotchas

- The operation name can vary between transformer versions. Always `print(model.transformer.h[0].attn.source)` first to confirm what is available.
- `.source` requires accessing it on the module that *calls* the operation. Do not chain `.source.foo.source` on a *submodule* - access that submodule directly. See `docs/usage/source.md`.
- For Llama / Mistral / Qwen the path is typically `model.model.layers[i].self_attn.source....` instead of `model.transformer.h[i].attn.source...`. The exact operation name (`attention_interface_0`, `eager_attention_forward_0`, etc.) varies by family. **There is no universal table — read the model's `forward` source code (or `print(model.<path>.source)`) to find the operation name.** This is the canonical way to discover op names per architecture.
- `attn_implementation="flash_attention_2"` is the fastest at runtime but does not let you read attention weights. Pay the cost and use eager when you need patterns.
- `[batch, n_heads, q_seq, k_seq]` can be very large for long contexts (`seq^2` per head per layer per batch element). For long-context analysis, save only the heads / layers you need.

## Related

- `docs/usage/source.md` - How `.source` works in general (forward rewriting, operation names, recursive access).
- [per-head-attention](per-head-attention.md) - Operating on individual heads in attention output.
- [logit-lens](logit-lens.md) - Pair attention patterns with logit lens to ask "what was this head reading, and what did the model predict next?"
- `tests/test_lm.py:691` - Source-tracing test that exercises attention output access.
