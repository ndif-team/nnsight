---
title: Per-Head Attention
one_liner: Read, ablate, or patch individual attention heads by slicing the output projection's input, by reading the already-per-head tensor from `.source`, or by exposing a `.heads` accessor with an `eproperty`.
tags: [pattern, interpretability, attention, heads]
related: [docs/patterns/attention-patterns.md, docs/usage/source.md, docs/patterns/ablation.md, docs/developing/extending-envoy.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/eproperty.py, src/nnsight/intervention/source.py, tests/test_source.py, tests/test_language.py]
---

# Per-Head Attention

## What this is for

Reading one head's output, ablating a head, or patching one head from another run.
Heads exist only **before** the attention output projection (`c_proj` on GPT-2,
`o_proj` on Llama-style models): its input is `[batch, seq, n_heads * head_dim]` with
head-major columns. After the projection, the hidden dimension no longer decomposes
per head.

## Pattern A: slice the output projection's input

The simplest correct route. Columns `[h*head_dim : (h+1)*head_dim]` of
`c_proj.input` are head `h`:

```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

n_heads  = model.config.n_head
# read it from the module — equals n_embd // n_head on GPT-2, but not on GQA/Gemma-style models
head_dim = model.transformer.h[0].attn._module.head_dim
LAYER, HEAD = 5, 4
prompt = "The cat sat on the"
lo, hi = HEAD * head_dim, (HEAD + 1) * head_dim

# Read every head's output.
with model.trace(prompt):
    x = model.transformer.h[LAYER].attn.c_proj.input       # [B, S, hidden]
    per_head = x.view(*x.shape[:2], n_heads, head_dim).save()

assert per_head.shape == (1, 5, n_heads, head_dim)

# Ablate one head.
with model.trace(prompt):
    model.transformer.h[LAYER].attn.c_proj.input[:, :, lo:hi] = 0
    logits = model.lm_head.output[:, -1, :].save()
```

Writing in place into a module's `.input`, as the ablation does, is supported: the
edit lands in the tensor `c_proj` receives. On this prompt, ablating layer 5 head 4
moves the baseline's top-token (`' floor'`) logit by `+0.307`. To sweep every head in
one trace, see [ablation](ablation.md#sweep-every-head).

!!! warning "Do not reshape `attn.output[0]`"

    `attn.output[0]` is the output *of* `c_proj`. `view(B, S, n_heads, head_dim)`
    succeeds on it but the slices are not heads: zeroing "head 4" there moves the
    same logit by `-0.178` — the wrong sign. Across all 144 GPT-2 heads the two
    deltas disagree in sign for about half (71). No error, just a plausible number
    unrelated to the head.

## Pattern B: per-head straight from `.source`

Inside GPT-2's attention forward, the attention-interface call returns a tensor that
is already `[B, S, n_heads, head_dim]`, before it is flattened and projected:

```python
with model.trace(prompt):
    out = model.transformer.h[LAYER].attn.source.attention_interface_1.output
    per = out[0].clone()                # [B, S, n_heads, head_dim]
    per[:, :, HEAD, :] = 0
    model.transformer.h[LAYER].attn.source.attention_interface_1.output = (per,) + tuple(out[1:])
    logits = model.lm_head.output[:, -1, :].save()
```

The saved logits are bitwise equal to Pattern A's. The op name
(`attention_interface_1`; `_0` is the assignment choosing the implementation) is
GPT-2-specific — discover yours with `print(model.transformer.h[0].attn.source)`. See
[source](../usage/source.md). Prefer Pattern A for plain ablation; use this form when
you also want the per-head tensor without a reshape.

## Pattern C: a first-class `.heads` accessor via `eproperty`

For repeated use, serve the per-head view of the projection's input as its own value:
an `eproperty` on an `Envoy` subclass, wired to `attn.c_proj` with `envoys=`.

```python
from nnsight import Envoy
from nnsight.intervention.eproperty import eproperty

class ProjHeads(Envoy):
    n_heads = 12                                   # model.config.n_head

    @eproperty(key="input")
    def heads(self, value):
        (x,), _ = value                            # key="input" serves (args, kwargs)
        b, s, d = x.shape
        return x.view(b, s, self.n_heads, d // self.n_heads)

    @heads.transform
    def heads(self, value, raw):                   # repack into (args, kwargs)
        b, s, n, head_dim = value.shape
        return ((value.reshape(b, s, n * head_dim),), {})

model = TransformersModel(
    "openai-community/gpt2", task="text-generation",
    envoys={"attn.c_proj": ProjHeads}, dispatch=True,
)

with model.trace(prompt):
    model.transformer.h[LAYER].attn.c_proj.heads[:, :, HEAD, :] = 0
    logits = model.lm_head.output[:, -1, :].save()
```

This reproduces Pattern A's logits bitwise. Key on the path suffix `"attn.c_proj"`,
not the `Conv1D` type — GPT-2's MLP has a `c_proj` too. Authoring details are in
[extending-envoy](../developing/extending-envoy.md); `tests/test_language.py`
(`TestCustomEnvoys`) is the tested example.

## Patching and attribution per head

To patch one head across invokes, combine Pattern A with `tracer.barrier(n)` — see
[activation-patching](activation-patching.md). For a `[layer, head]` attribution
map, see [attribution-patching](attribution-patching.md#sub-block-and-per-head-attribution).

## Interpretation tips

- **`head_dim` is not `hidden_size // n_heads`.** Read it off the attention
  module: `attn._module.head_dim` is there on GPT-2, Qwen and Llama alike. The config
  is less reliable — `Gemma2Config` carries `head_dim`, `Qwen2Config` raises
  `AttributeError` for it. On `gemma-2-2b`, `hidden_size` is 2304 and
  `num_attention_heads` is 8, but `head_dim` is 256 and `o_proj.in_features` is 2048,
  so `attn.output[0].view(B, S, 8, 288)` succeeds and is not heads.
- **Under grouped-query attention the head convention still holds.** The pattern
  and the projection's input are indexed by *query* head:
  `o_proj.input.view(B, S, num_attention_heads, head_dim)` is `torch.equal` to
  `.source.attention_interface_1.output[0]` on Qwen2.5-0.5B (14 query / 2 KV
  heads) and gemma-2-2b (8 / 4). Only the K and V projections are sliced by
  `num_key_value_heads`.
- **Aliasing matters.** `.view()` shares storage; `.reshape()` / `.contiguous()` may
  copy. If edits don't show up downstream, you mutated a copy — rebuild and assign.

## Gotchas

- Request source ops before the module's own `.output` in a forward, or hit
  `OutOfOrderError`.
- A wrong `head_dim` does not always raise. When `n_heads * head_dim` still equals
  the flat dimension, `.view()` succeeds and returns something that is not heads.
  When it does not, the error surfaces deep in the forward — `model.scan(prompt)`
  catches that case cheaply.

## Related

- [attention-patterns](attention-patterns.md) — reading attention probabilities.
- [activation-patching](activation-patching.md), [ablation](ablation.md),
  [attribution-patching](attribution-patching.md).
- [source](../usage/source.md) — how `.source` exposes intermediate ops.
- [extending-envoy](../developing/extending-envoy.md) — custom served values and the `envoys=` wiring.
