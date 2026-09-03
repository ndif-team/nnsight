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
2. **Read `.source.attention_interface_1.output[0]`**, which is *already* shaped
   `[batch, seq, n_heads, head_dim]` (before the reshape + `c_proj`). No manual
   reshape needed.
3. **Expose a first-class `.heads` accessor** with a custom `eproperty` on an
   `Envoy` subclass, wired to the output projection via `envoys=`. Then
   `attn.c_proj.heads` is a per-head view you read and write like any other
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
head_dim = model.transformer.h[0].attn._module.head_dim   # not n_embd // n_heads
LAYER, HEAD = 5, 4
prompt = "The cat sat on the"

# Read one head's output.
with model.trace(prompt):
    attn_out = model.transformer.h[LAYER].attn.output[0]   # [B, S, hidden]
    B, S, H = attn_out.shape
    per_head = attn_out.view(B, S, n_heads, head_dim).save()

print(per_head.shape)              # torch.Size([1, 5, 12, 64])
print(per_head[:, :, HEAD].shape)  # torch.Size([1, 5, 64])  -- head 4's output

assert per_head.shape == (1, 5, n_heads, head_dim)
```

`.attn.output` is a tuple `(attn_out, weights)`, so index `[0]` for the tensor.

### Reshaping the output tuple

Clone, zero one head's *slice of the projected output*, reshape back to flat, and
assign the whole tuple:

!!! warning "This is not head ablation"

    `attn.output[0]` is the output of `c_proj`. After that projection the hidden
    dimension no longer decomposes per head, so columns
    `[h*head_dim : (h+1)*head_dim]` are **not** head `h` — zeroing them removes
    something, but not that head's contribution. Measure it: take the prompt
    above, and score each run by how much the final-position logit of the
    baseline's top token moves —

    ```python
    with model.trace(prompt):
        base = model.lm_head.output[0, -1, :].save()
    tok = base.argmax().item()          # ' floor' for "The cat sat on the"
    delta = float(logits[0, tok] - base[tok])   # logits: the edited run's save
    ```

    — running the column-zeroing edit below and the real ablation of
    [Pattern B](#pattern-b-per-head-straight-from-source) for the same head.
    On GPT-2 layer 5 head 8, zeroing the columns moves the logit by `+0.016` —
    a rounding error — while really ablating the head moves it by `-1.47`: a
    ~90x understatement with the sign flipped. Layer 11 head 7 fails the other
    way: `+8.25` for the columns, `-0.06` for the head. Across all 144 heads
    the two deltas disagree in sign for 71 — no error, and a plausible-looking
    number unrelated to the head.

    To actually ablate a head, cut it *before* the projection — see
    [Pattern B](#pattern-b-per-head-straight-from-source) below, or slice
    `attn.c_proj.input`, which is equivalent. Use the pattern here only when you
    genuinely want to edit the projected output.

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
reassigned (`attn.output[0] = new_attn` raises). An in-place write needs no
rebuild, but it has to index the layout that is actually there:
`attn.output[0]` is `[B, S, hidden]`, so `attn.output[0][:, :, HEAD, :] = 0`
raises `IndexError: too many indices for tensor of dimension 3`. Slice the flat
columns instead — and see the warning above for what that does and does not
mean.

## Pattern B: per-head straight from `.source`

Inside GPT-2's attention forward, the attention output is per-head *before* it gets
flattened and projected. Read it directly from the source op — no reshape:

```python
with model.trace(prompt):
    ph = (
        model.transformer.h[LAYER].attn
        .source.attention_interface_1.output[0]     # already [B, S, n_heads, head_dim]
        .save()
    )
print(ph.shape)   # torch.Size([1, 5, 12, 64])

assert ph.shape == (1, 5, n_heads, head_dim)
```

Ablate a head at this stage (before `c_proj`) by rebuilding the op's output tuple:

```python
with model.trace(prompt):
    out = model.transformer.h[LAYER].attn.source.attention_interface_1.output
    per = out[0].clone()                # [B, S, n_heads, head_dim]
    per[:, :, HEAD, :] = 0
    model.transformer.h[LAYER].attn.source.attention_interface_1.output = (per,) + tuple(out[1:])
    logits = model.lm_head.output[:, -1, :].save()
```

The op name (`attention_interface_1`; `_0` is the assignment choosing the
implementation) is GPT-2-specific — discover yours with
`print(model.transformer.h[0].attn.source)`. See `docs/usage/source.md` and
`docs/patterns/attention-patterns.md`.

Equivalently, zero the head's slice of the projection's *input*, which needs no
tuple rebuild and no source-op name:

```python
with model.trace(prompt):
    lo, hi = HEAD * head_dim, (HEAD + 1) * head_dim
    model.transformer.h[LAYER].attn.c_proj.input[:, :, lo:hi] = 0
    logits = model.lm_head.output[:, -1, :].save()
```

Both routes agree exactly — the saved logits are bitwise equal. With the metric
from Pattern A's warning (GPT-2, layer 5 head 4, "The cat sat on the"), the
`' floor'` logit moves `-80.654` → `-80.347`, a delta of `+0.307`; the Pattern A
reshape gives `-0.178` for the same head — a different magnitude and the wrong
sign, because it is not a head ablation. Prefer this
one when you just want the ablation; prefer the source-op form when you also want to
read or edit the per-head tensor.

## Pattern C: a first-class `.heads` accessor via `eproperty`

For repeated use, expose the per-head view as its own served value. An `eproperty`
is the descriptor behind `.input` / `.output`; you can define your own. The
decorated stub is the **preprocess** — it takes the raw value served at the
module's location and returns what you read. Put it on an `Envoy` subclass, then
wire that subclass to a module with the `envoys=` argument, which maps a module
**type** (or a dotted **path suffix**) to a custom `Envoy` class.

Serve the **projection's input**, for the reason Pattern A's warning gives: that is
the last point at which the hidden dimension still decomposes per head.

```python
import torch
from nnsight import Envoy
from nnsight.intervention.eproperty import eproperty
from nnsight.modeling.transformers import TransformersModel

class ProjHeads(Envoy):
    n_heads = 12                                   # model.config.n_head

    @eproperty(key="input")
    def heads(self, value):
        (x,), _ = value                            # key="input" serves (args, kwargs)
        b, s, d = x.shape
        return x.view(b, s, self.n_heads, d // self.n_heads)

    @heads.transform
    def heads(self, value):                        # repack into (args, kwargs)
        b, s, n, head_dim = value.shape
        return ((value.reshape(b, s, n * head_dim),), {})

model = TransformersModel(
    "openai-community/gpt2", task="text-generation",
    envoys={"attn.c_proj": ProjHeads}, dispatch=True,
)

with model.trace(prompt):
    per_head = model.transformer.h[LAYER].attn.c_proj.heads.save()

assert per_head.shape == (1, 5, 12, 64)            # [B, S, n_heads, head_dim]

with model.trace(prompt):
    model.transformer.h[LAYER].attn.c_proj.heads[:, :, HEAD, :] = 0
    logits = model.lm_head.output[:, -1, :].save()
```

On the layer 5 head 4 setup above this reproduces the true ablation exactly —
the saved logits equal the source-op route's and the `c_proj.input` slice's
bitwise: the same `+0.307` move of the `' floor'` logit, against `-0.178` for
the post-projection reshape.

Three things this example turns on, none of them guessable from the signature:

- **A string key matches by dotted path suffix; a type key matches by class.**
  `"attn.c_proj"` is the right key here because GPT-2's MLP has a `c_proj` too and
  both are `Conv1D` — a type key would wrap the MLP projection as well.
- **`key="input"` serves the raw `(args, kwargs)` pair**, not a bare tensor. The
  preprocess destructures `(x,), _ = value`, and the transform has to hand back
  the same shape: `((tensor,), {})`.
- **`self._module` is the wrapped `torch.nn.Module`** — a `Conv1D` here, which
  knows nothing about heads, so the count is a class attribute. Where the module
  does carry it (an attention module has `num_heads`), read it from there.

### Aliasing view vs `.transform`

Whether you need a write-back callback depends on what the preprocess returns:

- **Aliasing view.** `value.view(...)` shares storage with the served tensor, so an
  in-place edit reaches the model without a `.transform`.
- **Computed / non-aliasing value.** If the preprocess returns a copy (a
  `.reshape()` that cannot view, a stack, an arithmetic result), in-place edits to
  it never reach the model. Register a `@heads.transform` that maps the edited view
  back to the module's real layout; it fires once, after the read, and is spliced
  in like a swap.

A `key="input"` preprocess is always in the second case, whatever it returns: the
served value is a container, so something has to rebuild it. A module whose
`.output` is a bare `[B, S, H]` tensor takes the same shape as the `Heads` example
in `tests/test_language.py`:

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

- **`head_dim` is not `hidden_size // n_heads`.** Read it off the attention
  module: `attn._module.head_dim` is there on GPT-2, Qwen and Llama alike
  (`transformers` 5.15). The config is less reliable — `Gemma2Config` carries
  `head_dim`, `Qwen2Config` raises `AttributeError` for it. On `gemma-2-2b`,
  `hidden_size` is 2304 and `num_attention_heads` is 8, but `head_dim` is 256 and
  `o_proj.in_features` is 2048, so `attn.output[0].view(B, S, 8, 288)` succeeds
  and is not heads.
- **Under grouped-query attention the head convention still holds.** The pattern
  and the projection's input are indexed by *query* head:
  `o_proj.input.view(B, S, num_attention_heads, head_dim)` is `torch.equal` to
  `.source.attention_interface_1.output[0]` on Qwen2.5-0.5B (14 query / 2 KV
  heads) and gemma-2-2b (8 / 4). Only the K and V projections are sliced by
  `num_key_value_heads`.
- **`attn.output[0]` is post-projection**; `.source.attention_interface_1.output[0]`
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
- A wrong `head_dim` does not always raise. When `n_heads * head_dim` still equals
  the flat dimension, `.view()` succeeds and returns something that is not heads.
  When it does not, the error surfaces deep in the forward — `model.scan(prompt)`
  catches that case cheaply.

## Related

- [attention-patterns](attention-patterns.md) — reading attention probabilities.
- [activation-patching](activation-patching.md), [ablation](ablation.md).
- [attribution-patching](attribution-patching.md) — per-head attribution maps.
- `docs/usage/source.md` — how `.source` exposes intermediate ops.
- `docs/concepts/envoy.md` — the extension surface (`eproperty`, subclassing `Envoy`).
- `docs/usage/extending.md` — custom served values and the `envoys=` wiring.
