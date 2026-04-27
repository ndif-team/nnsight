---
title: SAEs and Auxiliary Modules
one_liner: Wire a sparse autoencoder (or any auxiliary `nn.Module`) into a model and trace through it as a first-class submodule.
tags: [pattern, interpretability, sae, dictionaries, extending]
related: [docs/usage/extending.md, docs/usage/access-and-modify.md, docs/concepts/interleaver-and-hooks.md]
sources: [src/nnsight/intervention/envoy.py:239, src/nnsight/intervention/envoy.py:168, tests/test_transform.py]
---

# SAEs and Auxiliary Modules

## What this is for

Sparse autoencoders (SAEs), transcoders, dictionaries, and probes are *added* modules - they were not part of the original model but you want to run them on intermediate activations and observe / modify their outputs. The interpretability use case: replace a layer's output with `sae.decode(sae.encode(hs))` to see whether the SAE faithfully reconstructs the model's behavior, then patch / ablate / save SAE features.

Two patterns work well in nnsight:

1. **Apply the SAE inline** inside a trace as a one-shot intervention. Quick, no setup. Use `hook=True` if you want `.input` / `.output` access on the SAE itself.
2. **Attach the SAE as a submodule** of the wrapped model so it has a permanent path (`model.layers[5].sae`) and behaves like any other module - including in nested trace patterns and edits. This is also the pattern for first-class per-head access via `eproperty.transform`.

The same patterns apply to LoRA adapters, hooks, classifier heads, and any other "extra" module you want to interleave.

## When to use

- Reading SAE feature activations on a given prompt.
- Replacing a layer's residual with the SAE's reconstruction (drop-in test).
- Steering / ablating individual SAE features and measuring downstream effect.
- Attaching small auxiliary models (probes, classifiers) at fixed sites.
- Running a transcoder in place of an MLP block.

## Pattern A: inline application

Define your SAE somewhere (here a placeholder), then apply it directly inside the trace. Use `hook=True` when calling the auxiliary module if you want to read its `.input` / `.output` from elsewhere.

```python
import torch
from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

# Stand-in for a real SAE; in practice load weights from a checkpoint.
class SAE(torch.nn.Module):
    def __init__(self, d_model, d_dict):
        super().__init__()
        self.encoder = torch.nn.Linear(d_model, d_dict)
        self.decoder = torch.nn.Linear(d_dict, d_model)
    def forward(self, x):
        return self.decoder(torch.relu(self.encoder(x)))

sae = SAE(model.config.n_embd, 4 * model.config.n_embd).to(model.device)

LAYER = 6
prompt = "The Eiffel Tower is in the city of"

with model.trace(prompt):
    hs = model.transformer.h[LAYER].output
    reconstructed = sae(hs)                                  # plain Python call
    model.transformer.h[LAYER].output[:] = reconstructed
    logits = model.lm_head.output[:, -1, :].save()
```

Calling `sae(hs)` is just Python - it runs immediately on the real tensor your worker thread received from the hook. There is no nnsight wrapping unless you want it.

### When you want `.input` / `.output` on the SAE

Wrap the SAE with `NNsight` and reference it by attribute. To make a *call* to a wrapped Envoy fire its hooks (so `.input` / `.output` are populated), pass `hook=True`:

```python
from nnsight import NNsight

wrapped_sae = NNsight(sae)

with model.trace(prompt):
    hs = model.transformer.h[LAYER].output
    reconstructed = wrapped_sae(hs, hook=True)               # fire hooks
    sae_acts = wrapped_sae.encoder.output.save()             # works because hook=True
    model.transformer.h[LAYER].output[:] = reconstructed
```

The `hook` flag is part of `Envoy.__call__` (`src/nnsight/intervention/envoy.py:239`). By default, calling a wrapped module inside a trace dispatches to `forward()` and bypasses hooks - this is the right default for things like `model.lm_head(...)` in a logit-lens recipe. Setting `hook=True` opts back into the hook path so you can observe the call.

## Pattern B: attach the SAE as a submodule

Mounting the SAE on the underlying model gives it a permanent path inside the wrapped envoy tree. You can save its activations from a separate invoke without re-passing references:

```python
import torch
from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

LAYER = 6
sae = SAE(model.config.n_embd, 4 * model.config.n_embd).to(model.device)

# Mount the SAE inside the *underlying* PyTorch model.
# The Envoy tree will pick this up as model.transformer.h[LAYER].sae.
model.transformer.h[LAYER]._module.add_module("sae", sae)

# NOTE: the exact mechanism for refreshing the Envoy tree after add_module
# is currently under-documented — confirm with the maintainers if you hit
# "module not in envoy tree" errors. As a workaround you can wrap the model
# fresh (NNsight(...)) after mounting all SAEs, before any trace.

prompt = "The Eiffel Tower is in the city of"

with model.trace() as tracer:
    with tracer.invoke(prompt):
        # Use the SAE inline; hook=True so its .output is observable in another invoke.
        hs = model.transformer.h[LAYER].output
        recon = model.transformer.h[LAYER].sae(hs, hook=True)
        model.transformer.h[LAYER].output[:] = recon

    with tracer.invoke():
        # Empty invoke runs on the same batch - read SAE activations here.
        sae_out = model.transformer.h[LAYER].sae.output.save()
```

This pattern is cleaner if you have many SAEs at many layers - they all live at predictable paths and you do not have to thread Python references through your interventions.

## Pattern C: replace a block entirely

If your transcoder is meant to *replace* the MLP, use module skipping plus the transcoder's output:

```python
with model.trace(prompt):
    hs = model.transformer.h[LAYER].input              # block input
    transcoder_out = transcoder(hs, hook=True)
    # Skip the MLP entirely, providing transcoder output as its result.
    model.transformer.h[LAYER].mlp.skip(transcoder_out)
    logits = model.lm_head.output[:, -1, :].save()
```

See `docs/usage/skip.md`.

## Pattern D: first-class hookable values via `eproperty`

If you want a derived view of an auxiliary module's value to behave like `.output` — including in-place edits propagating back into the running forward pass — define an `eproperty` on a custom Envoy subclass with `preprocess`/`postprocess`/`transform`. This is the same machinery that backs the per-head attention pattern (`docs/patterns/per-head-attention.md`).

Worked example: split a module's output into a list (one tensor per head), let the user edit any head in-place, then recombine and feed the recombined value back. The `transform` callback is what makes the in-place edits propagate downstream.

```python
import torch
from nnsight import NNsight
from nnsight.intervention.envoy import eproperty
from nnsight.intervention.hooks import requires_output


class HeadedAttn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = 8
        self.head_dim = 6
        self.attn = torch.nn.Linear(self.n_heads * self.head_dim,
                                    self.n_heads * self.head_dim)

    def forward(self, x):
        return self.attn(x)


class HeadsEnvoy(NNsight):
    """Custom Envoy that exposes attention output split per head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached = None

    @eproperty(key="output")
    @requires_output
    def heads(self):
        # Stub — the decorator stack does the work; this body is never run.
        ...

    @heads.preprocess
    def heads(self, value):
        # Returned to the user as a list of [batch, head_dim] tensors.
        # The user can mutate any element in place.
        self.cached = list(value.view(self.n_heads, value.shape[0], self.head_dim))
        return self.cached

    @heads.transform
    def heads(self, value):
        # Fired after the user is done editing. Recombine the (possibly mutated)
        # heads and return the value that should be swapped into the model.
        return torch.cat(value, dim=1).view(value[0].shape[0], -1)


model = HeadsEnvoy(HeadedAttn())
example = torch.randn(2, model.n_heads * model.head_dim)

with model.trace(example):
    heads = model.heads.save()       # list of per-head tensors
    heads[0][:] = 0                  # ablate head 0 — propagates via transform
    output = model.output.save()
```

For the full mechanism (`preprocess` / `postprocess` / `transform` semantics), see `docs/usage/extending.md`. A second worked example with `tests/test_transform.py` shows the same pattern applied directly to a transformer attention output.

> **Reference SAE checkpoints.** A worked example loading a public SAE checkpoint (e.g., Goodfire / EleutherAI / Anthropic releases) into nnsight is on the wishlist but not yet in the docs. PRs welcome.

## Interpretation tips

- **Reconstruction faithfulness first.** Before drawing conclusions from SAE features, check that replacing the layer's residual with `sae(hs)` does not destroy task performance. Drop-in replacements should match clean accuracy within a few percent.
- **Feature ablation needs careful normalization.** Zeroing one feature out of thousands often does nothing measurable; aggregate across many prompts or use steering with the feature's decoder direction.
- **Encoder pre-activations vs post-ReLU activations** answer different questions. Save both in research code.
- **Not every layer is a good site.** SAE quality (reconstruction loss, feature interpretability) varies by layer; choose your intervention site based on the SAE's own metrics.

## Gotchas

- `.output` / `.input` on an auxiliary module only works if a hook fired on it. Plain `aux(x)` inside a trace does *not* fire hooks - use `aux(x, hook=True)`. See `docs/usage/access-and-modify.md` and `Envoy.__call__` (`src/nnsight/intervention/envoy.py:239`).
- Putting submodules into the wrapped tree must happen on the **underlying module** (`envoy._module.add_module(...)`), not on the Envoy. Re-tracing usually picks up the new submodule.
- Device placement of the SAE must match the activation site. Use `sae.to(model.device)` (or the device of the layer's output if device-mapped).
- For an SAE attached as a submodule, accessing `.output` requires the SAE to actually have been called. If you mount it but never call it inside the trace, you cannot save its `.output`.

## Related

- `docs/usage/extending.md` - Full reference for `envoys=`, `eproperty`, and `eproperty.transform`.
- `docs/usage/access-and-modify.md` - `hook=True` flag and the `__call__` vs `forward` dispatch.
- [per-head-attention](per-head-attention.md) - Same `eproperty.transform` machinery applied to attention heads.
- `tests/test_transform.py` - Worked transform / preprocess examples on a tiny model.
