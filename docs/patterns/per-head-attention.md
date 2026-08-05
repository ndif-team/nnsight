---
title: Per-Head Attention
one_liner: Read and modify individual attention heads by reshaping the attention output, by reading the already-per-head tensor from `.source`, or by exposing a first-class `.heads` accessor with an `eproperty`.
tags: [pattern, interpretability, attention, heads]
related: [docs/patterns/attention-patterns.md, docs/usage/source.md, docs/patterns/ablation.md, docs/concepts/envoy.md, docs/usage/extending.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/eproperty.py, src/nnsight/intervention/source.py, tests/test_source.py, tests/test_language.py]
---

# Per-Head Attention

## What this is for

The value-weighted attention output (after `c_proj` / `o_proj`) is laid out as
`[batch, seq, n_heads * head_dim]`. To operate on individual heads — read one
head's output, ablate a head, replace one head with another's — view that flat
dimension as `[batch, seq, n_heads, head_dim]`.

Three ways:

1. **Inline reshape** of `attn.output[0]` inside a trace. Works on the
   post-projection output.
2. **Read `.source.attention_interface_0.output[0]`**, which is *already* shaped
   `[batch, seq, n_heads, head_dim]` (before the reshape + `c_proj`). No manual
   reshape needed.
3. **Expose a first-class `.heads` accessor** with a custom `eproperty` on an
   `Envoy` subclass, wired to the attention module via `envoys=`. Then
   `attn.heads` is a hookable per-head view you read and write like any other
   activation — no reshape at the call site.

## When to use

- Per-head attention-output reads.
- Per-head ablation studies.
- Per-head patching (with `tracer.barrier(n)` to bring values across invokes).
- Building a per-head metric pipeline.

## Pattern A: inline reshape of the attention output

```python
import torch
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

n_heads  = model.config.n_head
head_dim = model.config.n_embd // n_heads
LAYER, HEAD = 5, 4
prompt = "The cat sat on the"

# Read one head's output.
with model.trace(prompt):
    attn_out = model.transformer.h[LAYER].attn.output[0]   # [B, S, hidden]
    B, S, H = attn_out.shape
    per_head = attn_out.view(B, S, n_heads, head_dim).save()

print(per_head.shape)              # torch.Size([1, 5, 12, 64])
print(per_head[:, :, HEAD].shape)  # torch.Size([1, 5, 64])  -- head 4's output
```

`.attn.output` is a tuple `(attn_out, weights)`, so index `[0]` for the tensor.

### Ablate one head

Clone, zero one head, reshape back to flat, and assign the whole tuple:

```python
with model.trace(prompt):
    out = model.transformer.h[LAYER].attn.output
    attn = out[0]
    B, S, _ = attn.shape
    edited = attn.view(B, S, n_heads, head_dim).clone()
    edited[:, :, HEAD, :] = 0
    new_attn = edited.view(B, S, n_heads * head_dim)
    model.transformer.h[LAYER].attn.output = (new_attn,) + tuple(out[1:])
    logits = model.lm_head.output[:, -1, :].save()
```

Here the whole tuple is rebuilt because `.view()` produces a *different* tensor
that has to take the element's place, and a tuple's elements cannot be
reassigned (`attn.output[0] = new_attn` raises). Editing the existing tensor in
place needs no rebuild — `attn.output[0][:, :, HEAD, :] = 0` writes straight
through, since `.output` hands back the live tensor.

## Pattern B: per-head straight from `.source`

Inside GPT-2's attention forward, the attention output is per-head *before* it gets
flattened and projected. Read it directly from the source op — no reshape:

```python
with model.trace(prompt):
    ph = (
        model.transformer.h[LAYER].attn
        .source.attention_interface_0.output[0]     # already [B, S, n_heads, head_dim]
        .save()
    )
print(ph.shape)   # torch.Size([1, 5, 12, 64])
```

Ablate a head at this stage (before `c_proj`) by rebuilding the op's output tuple:

```python
with model.trace(prompt):
    out = model.transformer.h[LAYER].attn.source.attention_interface_0.output
    per = out[0].clone()                # [B, S, n_heads, head_dim]
    per[:, :, HEAD, :] = 0
    model.transformer.h[LAYER].attn.source.attention_interface_0.output = (per,) + tuple(out[1:])
    logits = model.lm_head.output[:, -1, :].save()
```

The op name (`attention_interface_0`) is GPT-2-specific — discover yours with
`print(model.transformer.h[0].attn.source)`. See `docs/usage/source.md` and
`docs/patterns/attention-patterns.md`.

## Pattern C: a first-class `.heads` accessor via `eproperty`

For repeated use, expose the per-head view as its own hookable value. An
`eproperty` is the descriptor behind `.input` / `.output`; you can define your own.
The decorated stub is the **preprocess** — it takes the raw value served at the
module's location and returns what you read. Give it `@eproperty(key="output")` to
hook the module's output. Put it on an `Envoy` subclass, then wire that subclass to
the attention module with the `envoys=` argument, which maps a module **type** (or a
dotted **path suffix**) to a custom `Envoy` class.

GPT-2's attention `.output` is a `(attn_out, weights)` tuple, so the preprocess
indexes `value[0]`:

```python
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.eproperty import eproperty
from nnsight.modeling.transformers import TransformersModel

class AttnHeads(Envoy):
    @eproperty(key="output")
    def heads(self, value):                 # value = attn output tuple; [0] is [B, S, H]
        h = value[0]
        b, s, d = h.shape
        n = self._module.num_heads
        return h.view(b, s, n, d // n).transpose(1, 2)   # aliasing view -> edits propagate

model = TransformersModel(
    "openai-community/gpt2", task="text-generation",
    envoys={GPT2Attention: AttnHeads}, dispatch=True,
)

with model.trace(prompt):
    model.transformer.h[LAYER].attn.heads[:, 5] = 0      # zero head 5, in place
    logits = model.lm_head.output[:, -1, :].save()
```

`envoys={GPT2Attention: AttnHeads}` makes every `GPT2Attention` module an
`AttnHeads` envoy; modules not named by the map stay the base `Envoy`. A string key
matches by dotted path suffix instead of type: `envoys={"attn": AttnHeads}`.
`self._module` is the wrapped `torch.nn.Module`, so `self._module.num_heads` reads
the head count straight off GPT-2's attention.

### Aliasing view vs `.transform`

Whether you need a write-back callback depends on what the preprocess returns:

- **Aliasing view — no `.transform` needed.** `value.view(...).transpose(1, 2)`
  shares storage with the served tensor, so an in-place edit
  (`attn.heads[:, 5] = 0`) writes through to the model for free. The example above
  relies on exactly this.
- **Computed / non-aliasing value — add a `.transform`.** If the preprocess
  returns a copy (a `.reshape()` that can't view, a stack, an arithmetic result),
  in-place edits to it never reach the model. Register a `@heads.transform` that
  maps the edited view back to the module's real layout; it fires once, after the
  read, and is spliced in like a swap.

A module whose `.output` is a bare `[B, S, H]` tensor (an MLP, a block) uses the
same shape as the `Heads` example in `tests/test_language.py`, which pairs a
reshaping preprocess with a `.transform`:

```python
class Heads(Envoy):
    n_heads = 12

    @eproperty(key="output")
    def heads(self, value):                     # [B, S, H] -> [B, n_heads, S, head_dim]
        b, s, h = value.shape
        return value.view(b, s, self.n_heads, h // self.n_heads).transpose(1, 2)

    @heads.transform
    def heads(self, value):                     # write the edited view back
        b, nh, s, hd = value.shape
        return value.transpose(1, 2).reshape(b, s, nh * hd)
```

See `docs/concepts/envoy.md` and `docs/usage/extending.md` for the full `eproperty`
surface (`preprocess` / `postprocess` / `transform` / `provide`) and the `envoys=`
wiring. `tests/test_language.py` (`TestCustomEnvoys`) is the worked, tested example.

## Variations

### A reusable helper

```python
def heads(attn_output_tensor, n_heads):
    B, S, H = attn_output_tensor.shape
    return attn_output_tensor.view(B, S, n_heads, H // n_heads)

with model.trace(prompt):
    per_head = heads(model.transformer.h[LAYER].attn.output[0], n_heads).save()
```

### Per-head patching across invokes

Combine Pattern A or B with `tracer.barrier(n)` to move one head's activation from a
clean run into a corrupt run. See `docs/patterns/activation-patching.md`.

### Per-head attribution

Multiply a per-head reshape of `(act_clean - act_corrupt)` against the corresponding
gradient and sum over `head_dim` for a `[layer, head]` map. See
`docs/patterns/attribution-patching.md`.

## Interpretation tips

- **`n_heads` and `head_dim` are model-specific.** Read from `model.config`
  (`n_head` / `n_embd` for GPT-2; `num_attention_heads` for Llama-family).
- **`attn.output[0]` is post-projection**; `.source.attention_interface_0.output[0]`
  is pre-projection and already per-head.
- **Aliasing matters.** `.view()` shares storage; `.reshape()` / `.contiguous()` may
  copy. If edits don't show up downstream, you mutated a copy — rebuild and assign.
- **Position dimension.** `[B, S, n_heads, head_dim]` slices both head and position:
  `per_head[:, -1, HEAD, :]` is "head HEAD at the last position".

## Gotchas

- `.attn.output` is a tuple — index `[0]` for the value-weighted output.
- When replacing, rebuild the whole tuple `(new,) + tuple(out[1:])` and assign;
  don't `__setitem__` a tuple (`attn.output[0] = x` fails).
- Request source ops before the module's own `.output` in a forward, or hit
  `OutOfOrderError`.
- Mismatched `n_heads` / `head_dim` produces shape errors deep in the forward —
  check with `model.scan(prompt)` first.

## Related

- [attention-patterns](attention-patterns.md) — reading attention probabilities.
- [activation-patching](activation-patching.md), [ablation](ablation.md).
- [attribution-patching](attribution-patching.md) — per-head attribution maps.
- `docs/usage/source.md` — how `.source` exposes intermediate ops.
- `docs/concepts/envoy.md` — the extension surface (`eproperty`, subclassing `Envoy`).
- `docs/usage/extending.md` — custom hookable values and the `envoys=` wiring.
