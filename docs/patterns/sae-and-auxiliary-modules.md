---
title: SAEs and Auxiliary Modules
one_liner: Wire a sparse autoencoder (or any auxiliary `nn.Module`) into a model and trace through it as a first-class submodule.
tags: [pattern, interpretability, sae, dictionaries, extending]
related: [docs/usage/edit.md, docs/usage/access-and-modify.md, docs/usage/skip.md, docs/concepts/envoy.md, docs/usage/extending.md]
sources: [src/nnsight/intervention/envoy.py, src/nnsight/intervention/eproperty.py, tests/test_editing.py, tests/test_language.py]
---

# SAEs and Auxiliary Modules

## What this is for

Sparse autoencoders (SAEs), transcoders, dictionaries, probes, and LoRA adapters
are *added* modules — they were not part of the original model, but you want to run
them on intermediate activations and observe / modify their outputs. The classic
interpretability move: replace a layer's output with `sae(hs)` to check the SAE
reconstructs the model's behavior, then patch / ablate / save SAE features.

Four patterns, in order of setup cost:

1. **Apply inline** in a trace as a one-shot intervention. Quick, no setup.
2. **Attach as a submodule** (`model.transformer.h[6].sae = sae`) then route the
   layer through it in an `edit()`, so it applies on *every* trace and its internals
   (e.g. `sae.encoder.output`) become observable.
3. **Replace a block entirely** with `skip`.
4. **Expose a first-class hookable derived view** — a custom `eproperty` on an
   `Envoy` subclass wired to a site via `envoys=`, so the SAE's features (or any
   derived quantity) read/write like a built-in `.output`.

The same three patterns apply to transcoders, adapters, probes, and any other
"extra" module.

## When to use

- Reading SAE feature activations on a given prompt.
- Replacing a layer's residual with the SAE's reconstruction (drop-in test).
- Steering / ablating individual SAE features and measuring downstream effect.
- Attaching small auxiliary models (probes, classifiers) at fixed sites.
- Running a transcoder in place of an MLP block.

## Pattern A: inline application

Define your SAE (here a placeholder — in practice load weights from a checkpoint),
then apply it directly inside the trace. A GPT-2 block's `.output` is a plain
`(batch, seq, hidden)` tensor, so you overwrite it with a plain assignment.

```python
import torch
from nnsight.modeling.transformers import TransformersModel

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

Calling `sae(hs)` is plain Python — it runs immediately on the real tensor your
worker greenlet received from the hook. There is no nnsight wrapping unless you ask
for it. If you want the SAE's *features*, compute them directly (`sae.encoder(hs)`)
as above — reading `sae.encoder.output` inline does **not** work, because the worker
that called `sae(...)` has already run past that location by the time it asks for it
(see the gotcha below). To observe an attachment's internals, use Pattern B.

## Pattern B: attach as a submodule, then observe its internals

Attaching the SAE to the model gives it a permanent path
(`model.transformer.h[6].sae`) — just assign an `nn.Module` to an envoy attribute
and it is mirrored into the envoy tree. Route the layer through it with an
`edit()`, calling the attachment with `hook=True` so its own submodules' hooks fire.
Now every future trace runs the SAE, and you can read `sae.encoder.output`
directly.

```python
sae = SAE(d_model, 4 * d_model).to(model.device)
model.transformer.h[LAYER].sae = sae            # attach -> mirrored as an Envoy

with model.edit(inplace=True):
    acts = model.transformer.h[LAYER].output
    model.transformer.h[LAYER].output = model.transformer.h[LAYER].sae(acts, hook=True)

with model.trace(prompt):
    feats = model.transformer.h[LAYER].sae.encoder.output.save()   # observed via hook
    recon = model.transformer.h[LAYER].output.save()

print(feats.shape)   # torch.Size([1, 10, 3072])
```

Why this works and inline doesn't: `hook=True` runs the attachment's full
`module(...)` call (not just `forward`), so its submodules' hooks fire. The edit
runs the SAE call in the *edit worker*, while the trace body — reading
`sae.encoder.output` — runs in a *separate worker* parked on that location. One
worker produces the value, the other consumes it. See
[test_editing.py](../../tests/test_editing.py) `TestEditingWithAttachment` for the
full matrix (batching, per-invoke narrowing, per-step).

`edit(inplace=True)` mutates this model; `edit()` (the default) edits a copy and
hands it back as `with model.edit() as (tracer, edited):` — use the copy in later
traces. Clear edits with `model.clear_edits()`. See `docs/usage/edit.md`.

### Applying it on every generation step

A plain edit applies at the layer's *first* occurrence. To reapply on every step of
a generation loop, put the passthrough under the edit tracer's `iter`:

```python
import nnsight

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
```

Step 0 processes the whole prompt; later (KV-cached) steps process one token.

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

## Pattern D: a first-class hookable derived view via `eproperty`

Patterns A–C compute or attach at the call site every trace. To make the SAE's
features a *permanent, hookable* quantity — read like any built-in `.output` — give
the site's `Envoy` a custom `eproperty`.

An `eproperty` is the descriptor behind `.input` / `.output`; you can define your
own. The decorated stub is the **preprocess**: it receives the raw value served at
the module's location and returns what you read. Tagging it `@eproperty(key="output")`
hooks the module's output, so the preprocess can compute a derived view from the
layer's activation. Put it on an `Envoy` subclass, then wire that subclass to the
site with the `envoys=` argument, which maps a module **type** or a dotted **path
suffix** to a custom `Envoy` class.

```python
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.eproperty import eproperty

class SAEView(Envoy):
    """Exposes a layer's SAE feature activations as a first-class `.features` view."""

    @eproperty(key="output")
    def features(self, value):                       # value = layer output [B, S, H]
        return torch.relu(self.sae.encoder(value))   # computed view of the served tensor

SAEView.sae = SAE(d_model, 4 * d_model).to(model.device)   # class-level, like a config

model = TransformersModel(
    "openai-community/gpt2", task="text-generation",
    envoys={"transformer.h.6": SAEView}, dispatch=True,
)

with model.trace(prompt):
    feats = model.transformer.h[6].features.save()   # SAE features, read like any activation

print(feats.shape)                                   # torch.Size([1, 10, 3072])
```

`envoys={"transformer.h.6": SAEView}` matches the block at layer 6 by dotted path
suffix (component-wise, not substring); a type key like `envoys={GPT2Block: SAEView}`
would wrap *every* block instead. Modules not named by the map stay the base
`Envoy`. `self._module` is the wrapped `torch.nn.Module` if you need to read config
off it; here the SAE is a class attribute (shared by the subclass, like `n_heads`
in the per-head example), so `self.sae` reaches it.

### Aliasing view vs `.transform`

Reading `.features` is safe on its own: the preprocess computes a fresh tensor and
the layer's real output is untouched. To make edits to a view flow *back* into the
model, whether you need a write-back callback depends on what the preprocess returns:

- **Aliasing view — no `.transform`.** If the preprocess returns a view that shares
  storage with the served tensor (a `.view()` / `.transpose()` reshape, as in the
  per-head accessor), an in-place edit writes through for free.
- **Computed value — add a `.transform`.** SAE features are a *computed*,
  non-aliasing tensor, so edits to them never reach the model. Register a
  `@features.transform` to map the edited view back to the layer's `[B, S, H]`
  layout (e.g. `return self.sae.decoder(value)`); it fires once, after the read, and
  is spliced in like a swap. Note that with a decode-based transform, a decode is
  spliced in on *every* read (`decoder(encoder(x)) != x`), so add the transform only
  when you actually want the SAE reconstruction to replace the layer — otherwise
  keep `.features` read-only and do reconstruction with a plain `.output =`
  assignment (Pattern A).

The canonical, tested `eproperty` derived-view example is the per-head attention
accessor in `tests/test_language.py` (`Heads` / `TestCustomEnvoys`) and
`docs/patterns/per-head-attention.md`, where the reshape round-trips so its
`.transform` cleanly writes head edits back. See `docs/concepts/envoy.md` and
`docs/usage/extending.md` for the full `eproperty` surface (`preprocess` /
`postprocess` / `transform` / `provide`) and the `envoys=` wiring.

## Interpretation tips

- **Reconstruction faithfulness first.** Before drawing conclusions from SAE
  features, check that replacing a layer's residual with `sae(hs)` does not destroy
  task performance. Drop-in replacements should match clean accuracy within a few
  percent.
- **Feature ablation needs careful normalization.** Zeroing one feature out of
  thousands often does nothing measurable; aggregate across prompts, or steer with
  the feature's decoder direction.
- **Encoder pre-activations vs post-ReLU activations** answer different questions.
  Save both in research code.
- **Not every layer is a good site.** SAE quality varies by layer; choose your
  intervention site from the SAE's own metrics.

## Gotchas

- **Reading an attachment's internals must happen in a different worker than the
  one that called it.** Inline `sae(hs, hook=True)` followed by
  `sae.encoder.output` in the same trace body raises
  `OutOfOrderError` — the call already ran the encoder. Apply the attachment in an
  `edit()` (or a separate invoke) and read its internals in the trace body.
- **Attach on the envoy, not the raw module.** Assigning an `nn.Module` to an envoy
  attribute (`model.transformer.h[6].sae = sae`) registers it on the underlying
  module *and* mirrors it as a child envoy (see `Envoy.__setattr__`). You do not
  call `add_module` yourself.
- **`hook=True` is required to observe internals.** A plain `aux(x)` call inside a
  trace runs `forward` and skips hooks — the right default for `model.lm_head(...)`
  in a logit-lens recipe, but it means `.output`/`.input` on the aux module's
  submodules are never populated. Pass `hook=True` to opt into the hook path.
- **Device placement** of the SAE must match the activation site
  (`sae.to(model.device)`).

## Related

- `docs/usage/edit.md` — the `edit()` / `clear_edits()` reference.
- `docs/usage/access-and-modify.md` — `hook=True` and the `__call__` vs `forward` dispatch.
- `docs/usage/skip.md` — replacing a module's output.
- `docs/concepts/envoy.md` — the extension surface (`eproperty`, subclassing, attaching modules).
- `docs/usage/extending.md` — custom hookable values and the `envoys=` wiring.
- `docs/patterns/per-head-attention.md` — the tested `eproperty` derived-view example.
- `tests/test_editing.py` — `TestEditingWithAttachment` worked adapter/SAE examples.
- `tests/test_language.py` — `Heads` / `TestCustomEnvoys`, the `envoys=` + `eproperty` matrix.
