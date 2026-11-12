---
title: Attention Patterns
one_liner: Read attention probability matrices from the fused kernel with `attn.kernel.probs(...)`, or from `attn.output[1]` under eager execution, to see what each head attends to.
tags: [pattern, interpretability, attention, kernel, source-tracing]
related: [docs/usage/kernel.md, docs/usage/source.md, docs/patterns/per-head-attention.md, docs/patterns/logit-lens.md]
sources: [src/nnsight/intervention/kernel.py, src/nnsight/intervention/sites.py, src/nnsight/intervention/attention.py, src/nnsight/modeling/kernels/eager.py, src/nnsight/intervention/source.py, tests/test_kernel.py, tests/test_source.py]
---

# Attention Patterns

## What this is for

The attention pattern (the softmax-normalized matrix `A` in
`softmax(QK^T / sqrt(d)) V`) is the most direct read on what a head is looking at —
use it to find induction, copy, previous-token or name-mover heads, to compute
per-head metrics (entropy, attention to a position), or to check a hypothesis from
another method.

Under the default attention implementation (`sdpa`, which runs FlashAttention,
cuDNN, or memory-efficient attention on GPU), the kernel never materializes the
full probability matrix, so `attn.output[1]` is `None` and
`model.output.attentions` is an empty tuple. However, the kernel retains its
per-row logsumexp, and `attn.kernel.probs(...)` reconstructs the probabilities
used by the kernel in fp32 while leaving the model's output unchanged
([kernel](../usage/kernel.md)).

## Canonical pattern

```python
import torch
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

prompt = "The cat sat on the"

with model.trace(prompt):
    weights = model.transformer.h[0].attn.kernel.probs().save()

print(weights.shape)                      # [batch, n_heads, q_seq, k_seq]
print(weights.provenance.kind)            # 'exact reconstruction'

assert weights.shape == (1, 12, 5, 5)
assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)))   # rows sum to 1
assert (torch.triu(weights, diagonal=1) == 0).all()      # causal mask, exactly zero
```

Request only the slice you need: `probs(heads=4, query=-1)` returns the last
query row of head 4 with shape `[batch, k_seq]`, whereas requesting the full
matrix at long context materializes `batch x heads x L x L` floats.
Grouped-query models (such as Llama and Qwen) work the same way, with `heads=`
indexing query heads.

## Head scores from the kernel

The examples below compute standard head metrics from targeted slices of the
attention matrix. Each layer is requested in forward order inside a single
trace (matching `tests/test_kernel.py::TestWorkloads`).

**Induction score.** Repeat a random token sequence twice. Query `L + i` sits
at the second occurrence of token `i`, and an induction head places its
attention mass on key `i + 1` (the token that followed the first occurrence).
The induction score is the mean of that diagonal across the second half.

```python
import nnsight

torch.manual_seed(0)
half = torch.randint(1000, 20000, (1, 24))
with model.trace(torch.cat([half, half], 1)):
    rows = nnsight.save([])
    for block in model.transformer.h:
        rows.append(block.attn.kernel.probs(query=slice(24, None)))   # [1, heads, 24, 48]

queries, keys = torch.arange(24), torch.arange(24) + 1
scores = torch.stack([row[0][:, queries, keys].mean(-1) for row in rows])   # [layers, heads]
# scores[5, 5] = 0.93, scores[7, 10] = 0.87, scores[6, 9] = 0.82; layers 0 and 1 stay under 0.04
```

**Attention sink.** The attention mass placed on the first key per head and
layer is a single column of the probability matrix: `probs(key=0)` returns
`[batch, heads, queries]` directly without materializing the other key columns.

```python
with model.trace(prompt):
    sinks = nnsight.save([])
    for block in model.transformer.h:
        sinks.append(block.attn.kernel.probs(key=0))

mass = torch.stack([sink[0, :, 1:].mean(-1) for sink in sinks])   # [layers, heads], queries after the first
# on a 20-token prompt: layer 0 averages 0.19, layers 5 to 10 put between 0.69 and 0.77 of a row on key 0
```

**Retrieval head during generation.** Feed the start of a repeated random
sequence and let greedy decoding continue it. At each decode step, the kernel
runs a single query row against the KV cache, so `probs(heads=h, query=-1)` has
shape `[batch, prompt + step]`. The retrieval head's highest-probability key
points to the token following the previous occurrence, which matches the token
generated at that step.

```python
half = torch.randint(1000, 20000, (1, 12))
ids = torch.cat([half, half[:, :3]], 1)
attn = model.transformer.h[5].attn
with model.generate(ids, max_new_tokens=6, do_sample=False) as tracer:
    rows = nnsight.save([])
    for _ in tracer.iter[0:6]:
        rows.append(attn.kernel.probs(heads=5, query=-1))
    out = tracer.result.save()

for step, row in enumerate(rows):
    assert out[0, row[0].argmax()] == out[0, 15 + step]   # holds at all six steps
```

**Knockout.** Kernel sites are read-only; to block a specific `(query, key)`
pair, modify the attention mask passed to the kernel. For an unpadded prompt,
GPT-2 passes `None` as the mask and relies on causal masking, so you can
construct an explicit causal bias with `-inf` at the blocked `(query, key)`
entry and assign it to the attention interface's inputs. Reading `probs` then
confirms that the blocked key receives `0` probability with the remaining keys
renormalized.

```python
attn = model.transformer.h[5].attn
with model.trace(prompt):
    args, kwargs = attn.source.attention_interface_1.inputs       # (module, q, k, v, mask)
    q = args[1]
    n = q.shape[2]
    bias = torch.zeros(n, n, device=q.device).masked_fill(torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    bias[-1, 1] = float("-inf")                                    # last query may not see key 1
    attn.source.attention_interface_1.inputs = ((*args[:4], bias), kwargs)
    knocked = attn.kernel.probs(heads=2, query=-1).save()
    logits = model.output.logits.save()
# knocked[0] = [0.418, 0.0, 0.349, 0.218, 0.015]; before the edit the row was [0.272, 0.349, 0.227, 0.142, 0.01]
```

Reductions across the full key axis (such as entropy or row maximum) require
all keys and materialize `heads x L x L` floats per layer; at long context,
slice `heads=` to the subset of heads you need.

## Under `attn_implementation="eager"`

Under eager execution, the attention module computes the softmax in Python and
returns the probability matrix alongside its output, so `attn.output[1]` gives
that layer's pattern and `model.output.attentions` collects all layers at once.
Because eager attention uses different floating-point arithmetic from fused
kernels (on Qwen3-8B in bf16 across a 56-token prompt and layers 0 to 9, eager
probabilities differ from the kernel's by a median of 0.6% and up to 20% at the
top key), prefer `attn.kernel.probs(...)` when analyzing the default model. Use
eager execution when the model uses a non-SDPA kernel or custom Python
attention logic.

```python
model = TransformersModel("openai-community/gpt2", dispatch=True, attn_implementation="eager")

with model.trace(prompt):
    weights = model.transformer.h[0].attn.output[1].save()
```

`attn.kernel.probs(...)` is also served under `attn_implementation="eager"` and
returns that same weights tensor (`provenance.kind == "exact export"`,
`provenance.kernel == "modeling_gpt2.eager_attention_forward"`), sliced by
`heads=`, `query=`, and `key=`, so code written against `.kernel` runs under
either implementation ([kernel](../usage/kernel.md#under-eager-attention)).

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
the assignment that picks the implementation). Under `attn_implementation="eager"`
it resolves at run time to `eager_attention_forward`, so you can chain `.source`
again to reach the softmax (under `sdpa`, no softmax runs in Python, and drilling
raises an error that `nn_functional_softmax_0` is not an operation):

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
`eager_attention_forward` under eager. See [source](../usage/source.md).

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
- Don't unpack across `.save()` (`a, b = op.output.save()`); see
  [save](../usage/save.md#gotchas).

## Related

- [source](../usage/source.md) — how `.source` works (forward rewriting, op names,
  recursive access).
- [per-head-attention](per-head-attention.md) — operating on individual heads.
- `nnterp`'s `StandardizedTransformer(..., enable_attention_probs=True)` gives
  `model.attention_probabilities[i]` with no per-architecture names at all.
- `tests/test_source.py` — source-tracing tests.
