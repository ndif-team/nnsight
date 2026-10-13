---
title: SAEs and Auxiliary Modules
one_liner: Wire a sparse autoencoder (or any auxiliary `nn.Module`) into a model and trace through it as a first-class submodule.
tags: [pattern, interpretability, sae, dictionaries, extending]
related: [docs/usage/edit.md, docs/usage/access-and-modify.md, docs/usage/skip.md, docs/usage/invoke-and-batching.md, docs/developing/extending-envoy.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/eproperty.py, tests/test_editing.py, tests/test_language.py]
---

# SAEs and Auxiliary Modules

## What this is for

Sparse autoencoders (SAEs), transcoders, dictionaries, probes and adapters are
*added* modules: not part of the original model, but run on its intermediate
activations. Use these patterns to read SAE feature activations, to replace a layer's
residual with the SAE's reconstruction, to steer or ablate individual features, or to
run a transcoder in place of an MLP. Four patterns, in order of setup cost:

1. **Apply inline** in a trace as a one-shot intervention.
2. **Attach as a submodule** (`model.transformer.h[6].sae = sae`) and route the layer
   through it in an `edit()`, so it applies on every trace and its internals
   (`sae.encoder.output`) become observable.
3. **Replace a block entirely** with `skip`.
4. **Expose a derived view** — an `eproperty` on an `Envoy` subclass wired with
   `envoys=`, so the features read like a built-in `.output`.

## Pattern A: inline application

Define your SAE (here a placeholder — in practice load weights from a checkpoint),
then apply it directly inside the trace. A GPT-2 block's `.output` is a plain
`(batch, seq, hidden)` tensor, so you overwrite it with a plain assignment.

```python
import torch
from nnsight.modeling.transformers import TransformersModel

torch.manual_seed(0)                        # the SAE below is randomly initialised
model = TransformersModel("openai-community/gpt2", dispatch=True)

# Stand-in for a real SAE.
class SAE(torch.nn.Module):
    def __init__(self, d_model, d_dict):
        super().__init__()
        self.encoder = torch.nn.Linear(d_model, d_dict)
        self.decoder = torch.nn.Linear(d_dict, d_model)
    def forward(self, x):
        return self.decoder(torch.relu(self.encoder(x)))

d_model = model.config.n_embd
sae = SAE(d_model, 4 * d_model).to(model.device)

LAYER = 6
prompt = "The Eiffel Tower is in the city of"

with model.trace(prompt):
    hs = model.transformer.h[LAYER].output          # plain tensor (B, S, H)
    feats = torch.relu(sae.encoder(hs)).save()      # SAE features, computed directly
    model.transformer.h[LAYER].output = sae(hs)     # replace with reconstruction
    logits = model.lm_head.output[:, -1, :].save()

print(feats.shape)                                  # torch.Size([1, 10, 3072])
print(repr(model.tokenizer.decode(logits.argmax(-1))))   # ' to'
```

Calling `sae(hs)` is plain Python on the real tensor; nnsight does not wrap it.
Compute features directly (`sae.encoder(hs)`) as above — to read
`sae.encoder.output` instead, use Pattern B.

## Pattern B: attach as a submodule, then observe its internals

Assigning an `nn.Module` to an envoy attribute registers it on the underlying module
and mirrors it into the envoy tree — no `add_module` needed. Route the layer through
it in an `edit()`, and every later trace runs the SAE and can read
`sae.encoder.output`.

```python
sae = SAE(d_model, 4 * d_model).to(model.device)
model.transformer.h[LAYER].sae = sae            # attach -> mirrored as an Envoy

with model.edit(inplace=True):
    acts = model.transformer.h[LAYER].output
    model.transformer.h[LAYER].output = model.transformer.h[LAYER].sae(acts, hook=True)

with model.trace(prompt):
    feats = model.transformer.h[LAYER].sae.encoder.output.save()   # served by the call
    recon = model.transformer.h[LAYER].output.save()

print(feats.shape)   # torch.Size([1, 10, 3072])
```

`hook=True` runs the attachment's full `module(...)` call (not just `forward`), so
its submodules hand off their values; a plain `aux(x)` call bypasses them. **An
attachment's internals must be read in a different worker than the one that called
it**: the edit runs the SAE call, and the trace body parks on `sae.encoder.output`.
Calling `sae(acts, hook=True)` inline and then reading `sae.encoder.output` in the
same trace body raises `OutOfOrderError` — the call already ran the encoder. See
[test_editing.py](../../tests/test_editing.py) `TestEditingWithAttachment` for the
full matrix (batching, per-invoke narrowing, per-step).

`edit(inplace=True)` mutates this model and accumulates across calls; `edit()` (the
default) edits a copy. Clear edits with `model.clear_edits()`. See
[edit](../usage/edit.md).

### Applying it on every generation step

A plain edit applies at the layer's *first* occurrence. To reapply on every step of
a generation loop, put the passthrough under the edit tracer's `iter`:

```python
import nnsight

model.clear_edits()                         # or the edit above runs a second time
model.transformer.h[LAYER].sae = SAE(d_model, 4 * d_model).to(model.device)

with model.edit(inplace=True) as tracer:
    for _ in tracer.iter[:]:
        acts = model.transformer.h[LAYER].output
        model.transformer.h[LAYER].output = model.transformer.h[LAYER].sae(acts, hook=True)

with model.generate("The Eiffel Tower is in", max_new_tokens=3, do_sample=False) as tracer:
    feats = nnsight.save([])
    for _ in tracer.iter[:3]:
        feats.append(model.transformer.h[LAYER].sae.encoder.output)

print([tuple(f.shape) for f in feats])   # [(1, 7, 3072), (1, 1, 3072), (1, 1, 3072)]
model.clear_edits()
```

Step 0 processes the whole prompt; later (KV-cached) steps process one token.

### Making a parameter-based SAE observable

The SAE above is built from `torch.nn.Linear` submodules, so `sae.encoder.output`
is a location for free. Real pretrained dictionaries ship as an array file, and a
faithful loader holds bare parameters and does the work in plain methods.
Only `nn.Module` children are locations: with `encode` as a plain method,
`sae.encode.output` raises `AttributeError: 'function' object has no attribute
'output'`. Make every value you want to read pass through a module. `WrapperModule`
is an identity module for exactly this — routing a value through it costs nothing
and gives that value a location.

```python
from nnsight.modeling.transformers import WrapperModule

n_features = 4 * d_model

class ModuleSAE(torch.nn.Module):
    def __init__(self, d_model, n_features):
        super().__init__()
        self.encode = WrapperModule()              # now a location
        self.decode = WrapperModule()
        self.W_enc = torch.nn.Parameter(torch.randn(d_model, n_features) / d_model**0.5)
        self.W_dec = torch.nn.Parameter(torch.randn(n_features, d_model) / n_features**0.5)
        self.threshold = torch.nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        pre = x @ self.W_enc
        acts = self.encode(torch.relu(pre) * (pre > self.threshold))
        return self.decode(acts @ self.W_dec)
```

Load your checkpoint's arrays (Gemma Scope ships a `.npz` of
`W_enc`/`W_dec`/`threshold`) in place of the random initialisation. Then attach it,
splice it in with an `edit`, and read its internals from any later trace:

```python
model.clear_edits()
model.transformer.h[LAYER].sae = ModuleSAE(d_model, n_features).to(model.device)

with model.edit(inplace=True):
    acts = model.transformer.h[LAYER].output
    model.transformer.h[LAYER].output = model.transformer.h[LAYER].sae(acts, hook=True)

with model.trace(prompt):
    feats = model.transformer.h[LAYER].sae.encode.output.save()
    recon = model.transformer.h[LAYER].sae.decode.output.save()

print(tuple(feats.shape), tuple(recon.shape))   # (1, 10, 3072) (1, 10, 768)
model.clear_edits()
```

`evaluating_saes.ipynb` in the tutorials builds this class against a real Gemma
Scope `.npz` and checks the result against the dictionary's published L0.

The same trick applies to anything you want to observe that is not already a
module — an intermediate in a custom loss, a probe's pre-activation, a step inside
an adapter.

## Pattern C: replace a block entirely with `skip`

If your transcoder is meant to *replace* the MLP, skip the MLP and feed the
transcoder's output as its result:

```python
transcoder = SAE(d_model, 4 * d_model).to(model.device)

with model.trace(prompt):
    mlp_in = model.transformer.h[LAYER].mlp.input
    model.transformer.h[LAYER].mlp.skip(transcoder(mlp_in))
    out = model.transformer.h[LAYER].mlp.output.save()

print(out.shape)   # torch.Size([1, 10, 768])
```

`skip(x)` bypasses the module's forward and substitutes `x` as its output. Read the
module's own `.input` first (it is offered before the skip gate). See
`docs/usage/skip.md`.

## Pattern D: a first-class derived view via `eproperty`

Patterns A–C compute or attach at the call site every trace. To make the SAE's
features a permanent, served quantity — read like any built-in `.output` — give the
site's `Envoy` a custom `eproperty` and map the site to it with `envoys=`:

```python
from nnsight import Envoy
from nnsight.intervention.eproperty import eproperty

class SAEView(Envoy):
    @eproperty(key="output")
    def features(self, value):                       # value = layer output [B, S, H]
        return torch.relu(self.sae.encoder(value))   # computed view of the served tensor

SAEView.sae = SAE(d_model, 4 * d_model).to(model.device)   # class-level, like a config

model = TransformersModel(
    "openai-community/gpt2", task="text-generation",
    envoys={"transformer.h.6": SAEView}, dispatch=True,   # a type key would wrap every block
)
with model.trace(prompt):
    feats = model.transformer.h[6].features.save()   # torch.Size([1, 10, 3072])
```

Reading `.features` leaves the layer's output untouched. SAE features are a
computed tensor, so edits to them reach the model only through a
`@features.transform` (e.g. `return self.sae.decoder(value)`) — and that splices the
reconstruction in on *every* read, since `decoder(encoder(x)) != x`. Add one only
when you want the reconstruction to replace the layer; otherwise keep `.features`
read-only and reconstruct with a plain `.output =` (Pattern A). The full surface is
in [extending-envoy.md](../developing/extending-envoy.md).

## Interpretation tips

- **Check reconstruction faithfulness first.** Replacing a layer's residual with
  `sae(hs)` should keep task performance within a few percent of clean.
- **Zeroing one feature out of thousands often does nothing measurable**; aggregate
  across prompts, or steer with the feature's decoder direction.
- **SAE quality varies by layer**; choose the site from the SAE's own metrics.

## Gotchas

- **Device placement** of the SAE must match the activation site
  (`sae.to(model.device)`).
- **A batched trace pads, and the SAE fires on the padding.** Everything in one
  trace is left-padded to the batch's longest input, so
  `sae.encode(block.output).reshape(-1, d_features)` over prompts of different
  lengths mixes real activations with pad ones. At GPT-2's layer 6 a pad
  position's residual carries about eight times the norm of a real one, so a
  max-activating-example search reports a pad position as a feature's best example.
  Index the activation with the batch's `attention_mask` before ranking or fitting
  (see [the masked
  reduction](../usage/invoke-and-batching.md#reducing-over-real-tokens-the-attention-mask)).
  The ratio is architecture-specific (pythia-70m's pad residuals are *smaller*). See
  [invoke-and-batching.md](../usage/invoke-and-batching.md#rows-padding-and-getting-values-back).

## Related

- [edit](../usage/edit.md) — the `edit()` / `clear_edits()` reference.
- [access-and-modify](../usage/access-and-modify.md) — `hook=True` and the `__call__` vs `forward` dispatch.
- [skip](../usage/skip.md) — replacing a module's output.
- [extending-envoy](../developing/extending-envoy.md) — custom served values and the `envoys=` wiring.
- [per-head-attention](per-head-attention.md) — another `eproperty` derived view.
- `tests/test_editing.py` — `TestEditingWithAttachment` worked adapter/SAE examples.
- `tests/test_language.py` — `TestCustomEnvoys`, the `envoys=` + `eproperty` matrix.
