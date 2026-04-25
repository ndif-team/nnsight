---
title: Per-Head Attention
one_liner: Two ways to read and modify individual attention heads - inline reshape inside a trace, or a custom Envoy with `eproperty.transform`.
tags: [pattern, interpretability, attention, heads, extending]
related: [docs/patterns/attention-patterns.md, docs/usage/extending.md, docs/patterns/ablation.md]
sources: [src/nnsight/intervention/envoy.py:168, src/nnsight/intervention/interleaver.py:60, tests/test_transform.py:317, tests/test_transform.py:294]
---

# Per-Head Attention

## What this is for

Attention output (the result of the `o_proj` / `c_proj` step) is naturally laid out as `[batch, seq, n_heads * head_dim]`. To operate on individual heads - read one head's output, ablate one head, replace one head with another's - you need to view that flat dimension as `[batch, seq, n_heads, head_dim]`.

Two equally valid ways:

1. **Inline reshape inside the trace.** Quick, no boilerplate. Best for one-off scripts.
2. **A custom `Envoy` with an `eproperty`** that exposes `.heads` as a per-head view. Cleaner for repeated use across a model and across many traces - you write the reshape once, then `model.attn.heads[3]` everywhere.

For modifying a head, the second pattern needs `eproperty.transform` to swap the user-edited reshape back into the flat tensor that the model continues to use. `transform` runs on the mediator side after the worker yields control. See `src/nnsight/intervention/interleaver.py:198` (the `transform` decorator) and the worked example at `tests/test_transform.py:317`.

## When to use

- Per-head attention pattern reads.
- Per-head ablation studies.
- Per-head patching (using `tracer.barrier(n)` to bring values across invokes).
- Building a per-head metric pipeline you reuse across many models.

## Pattern A: inline reshape

The simplest approach - reshape inside the trace, modify a slice, reshape back.

```python
import torch
from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

n_heads  = model.config.n_head
head_dim = model.config.n_embd // n_heads
LAYER    = 5
HEAD     = 4
prompt   = "The cat sat on the"

# Read one head's output.
with model.trace(prompt):
    attn_out = model.transformer.h[LAYER].attn.output[0]    # [B, S, hidden]
    B, S, H = attn_out.shape
    per_head = attn_out.view(B, S, n_heads, head_dim).save()

print(per_head.shape)             # [B, S, n_heads, head_dim]
print(per_head[:, :, HEAD].shape) # [B, S, head_dim]  -- head 4's output
```

To ablate one head:

```python
with model.trace(prompt):
    attn = model.transformer.h[LAYER].attn.output[0]
    B, S, _ = attn.shape

    # In-place: zero head HEAD across all positions.
    reshaped = attn.view(B, S, n_heads, head_dim)
    reshaped[:, :, HEAD, :] = 0
    # No swap needed - the .view shares storage with the underlying tensor,
    # so the in-place write propagates.

    logits = model.lm_head.output[:, -1, :].save()
```

This works because `view()` returns a tensor that shares storage with `attn`. In-place writes go through to the original.

If you instead clone or otherwise break aliasing (e.g. `.reshape` on a non-contiguous tensor returns a copy in some cases), you must write the modified tensor back:

```python
with model.trace(prompt):
    attn = model.transformer.h[LAYER].attn.output[0]
    B, S, _ = attn.shape

    edited = attn.view(B, S, n_heads, head_dim).clone()
    edited[:, :, HEAD, :] = 0
    new_attn = edited.view(B, S, n_heads * head_dim)

    # Replace - .attn.output is a tuple; preserve the rest.
    out = model.transformer.h[LAYER].attn.output
    model.transformer.h[LAYER].attn.output = (new_attn,) + out[1:]
```

## Pattern B: custom Envoy with `eproperty`

For a model where you do this often, define a custom `Envoy` subclass that exposes `.heads` directly. The cleanest version uses `preprocess` only - it returns a list of *views* into the model's tensor, and in-place writes propagate naturally without needing `transform`. This is exactly the pattern in `tests/test_transform.py:294`:

```python
import torch
from nnsight import NNsight
from nnsight.intervention.envoy import Envoy, eproperty
from nnsight.intervention.hooks import requires_output


class AttnHeadsEnvoy(Envoy):
    """Exposes `.heads` as a list of [B, S, head_dim] views into attn output.

    No clone, no transform: each list entry is a view of the underlying
    [B, S, hidden] tensor, so in-place edits propagate directly.
    """

    @eproperty(key="output")
    @requires_output
    def heads(self): ...

    @heads.preprocess
    def heads(self, value):
        n_heads = self._module.n_heads          # set on the underlying module
        B, S, H = value.shape
        return list(value.view(B, S, n_heads, H // n_heads).unbind(dim=2))
```

Wire it onto the model with the `envoys=` mapping:

```python
# Suppose YourAttnClass is the torch class for attention output that has .n_heads
model = NNsight(your_model, envoys={YourAttnClass: AttnHeadsEnvoy})

with model.trace(x):
    heads = model.attn.heads.save()    # list of n_heads tensors, each [B, S, head_dim]
    heads[1][:] = 0                    # zero head 1 in place; propagates to model
    out = model.output.save()
```

For HuggingFace GPT-2, the attention output module is the parent attention block (`GPT2Attention`); you can mount the `AttnHeadsEnvoy` on it via `envoys={GPT2Attention: AttnHeadsEnvoy}` and access `model.transformer.h[L].attn.heads`.

### When you do need `transform`

If you need to expose a *clone* (so users get a safe edit surface without aliasing surprises), or you need to change the *shape* of the value seen by users vs the model, use `eproperty.preprocess` to return the user-facing form and `eproperty.transform` to reshape it back before the model sees it:

```python
class ReshapeHeadsEnvoy(Envoy):
    n_heads = 2  # also discoverable from the wrapped module

    @eproperty(key="output")
    @requires_output
    def heads(self): ...

    @heads.preprocess
    def heads(self, value):
        B, H = value.shape
        return value.clone().view(B, self.n_heads, H // self.n_heads)

    @heads.transform
    @staticmethod
    def heads(value):
        # value is the (possibly mutated) clone from preprocess.
        # Reshape back to what the model expects.
        B, n_heads, head_dim = value.shape
        return value.reshape(B, n_heads * head_dim)
```

The `transform` callback runs on the mediator after the worker yields control; whatever it returns is `batcher.swap`'d back into the model. See the docstring on `eproperty.transform` (`src/nnsight/intervention/interleaver.py:198`) and the worked example `tests/test_transform.py:177` (`test_reshape_transform_per_head_edit`).

## Variations

### Per-head patching across invokes

Combine pattern A or B with `tracer.barrier(n)` to move one head's activation from a clean run into a corrupt run. See `docs/patterns/activation-patching.md`.

### Per-head attribution

Multiply a per-head reshape of `(act_clean - act_corrupt)` against the corresponding gradient and sum over `head_dim` to get a `[layer, head]` attribution map. See `docs/patterns/attribution-patching.md`.

## Interpretation tips

- **`n_heads` and `head_dim` are model-specific.** Read from `model.config` (HF) or store on the underlying module.
- **`attn.output[0]` is post-projection.** This is the value-weighted output after `o_proj`. To operate before `o_proj`, you need `.source` to reach the SDPA call - see `docs/patterns/attention-patterns.md`.
- **Aliasing matters.** A `.view()` shares storage; a `.reshape()` or `.contiguous()` may not. If your edits do not show up downstream, check whether you mutated a copy.
- **Position dimension.** `[B, S, n_heads, head_dim]` lets you slice both head and position: `reshaped[:, -1, HEAD, :]` is "head HEAD at the last position".

## Gotchas

- `.attn.output` is a tuple. Index `[0]` for the value-weighted output.
- For `eproperty.transform`, the function is decorated with `@staticmethod` because the preprocessed value is bound by closure - see the docstring at `src/nnsight/intervention/interleaver.py:198`.
- `eproperty` requires `IEnvoy`-like state (a `_module` and a `path`); subclass `Envoy` and use `@requires_output` / `@requires_input` from `nnsight.intervention.hooks`.
- Mismatched `n_heads` / `head_dim` between your reshape and `model.config` produces shape errors deep in the forward; double-check using `model.scan(prompt)` first.

## Related

- [attention-patterns](attention-patterns.md) - Reading attention probabilities (vs operating on output).
- [activation-patching](activation-patching.md), [ablation](ablation.md) - Things to do with one head once you have access.
- [attribution-patching](attribution-patching.md) - Per-head attribution maps.
- `docs/usage/extending.md` - Full reference for `envoys=` and `eproperty`.
- `tests/test_transform.py:294` - End-to-end test of `_AttnHeadsEnvoy` on a tiny attention block.
- `tests/test_transform.py:177` - `test_reshape_transform_per_head_edit` shows the `preprocess + transform` round-trip.
