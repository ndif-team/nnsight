# Patterns - Open Questions

Questions that came up while writing `docs/patterns/`. Resolve before publishing externally.

## docs/patterns/logit-lens.md

1. The example uses `model.transformer.h[i].output[0]` as the residual. For HF GPT-2 with `output_attentions=False`, is the tuple still `(hidden, ...)` such that `[0]` is correct, or does it depend on `use_cache` / `output_hidden_states`? Worth verifying with `model.scan(prompt)` and stating the expected shape definitively.
2. Should we recommend a specific Llama-style example alongside GPT-2 (paths differ: `model.model.layers[L].output[0]`, `model.model.norm`)? The current doc punts to "use `print(model)`".

## docs/patterns/activation-patching.md

1. Is `tracer.barrier(2)` always required when both invokes only *read* `.output[0]` (clean) and one writes (corrupt)? The code reads `[:, -1, :]` from clean as a slice, then writes the same slice in corrupt. Confirm whether the "no barrier" path actually deadlocks vs raises a NameError vs just produces wrong results.
2. The "Patch attention output / MLP output" variation says "shape and tuple-vs-tensor varies by submodule type". Worth a one-line table per common arch (GPT-2 vs Llama vs Mistral)?
3. Should we provide a notebook-runnable end-to-end sweep with a plot? The current doc has the code skeleton but no visualization step.

## docs/patterns/ablation.md

1. The "zero one attention head" variation reshapes via `attn_out.view(B, S, n_heads, head_dim).clone()` then writes back via tuple replacement. With GPT-2's `attn.output` tuple, are there ever extra elements beyond `[0]` whose preservation is required (presents, attentions)? `out[1:]` should be safe but worth confirming.
2. Mean ablation example uses `acts = []` then `torch.stack([a for a in acts])`. The list itself is local Python; do all elements need to be `.save()`'d individually, or does `nnsight.save(acts)` cover it? The doc currently saves each one individually.

## docs/patterns/attention-patterns.md

1. We tell users to use `attn_implementation="eager"` for attention weights. Confirm this is still the right kwarg path for GPT-2 in current transformers (sometimes it's `attn_implementation` on the config, sometimes a kwarg to `from_pretrained`). Does `LanguageModel("openai-community/gpt2", attn_implementation="eager")` actually forward this through to `AutoModelForCausalLM.from_pretrained`?
2. The doc says "the second tuple element of `attention_interface_0` may be `None`" with sdpa. Worth verifying on a current transformers release - has this been changed to raise instead?
3. For Llama / Mistral, what is the analogous `attention_interface_0` path? Worth a one-line note per family.

## docs/patterns/steering.md

1. Refusal-direction doc references Arditi et al. (2024) - do they project out the direction or subtract it? My text says "subtracts (or zero-projects)" which covers both, but the canonical recipe is projection. Should I show projection explicitly?
2. The "Steering during generation" example puts the addition outside any `iter[...]`. Confirm that this fires on every generation step (not just the first) - my understanding from `default_all` machinery is yes, but worth verifying.

## docs/patterns/attribution-patching.md

1. The doc accesses gradients with `for L, hs in enumerate(hidden_refs)` (forward order) inside the backward context. The text warns that backward access is reverse and tells users to flip the loop if needed. Is the example actually broken (will hang) or does the gradient hook system handle forward-order access fine? Need to test.
2. The metric `(logits[:, paris] - logits[:, rome]).sum().backward()` - is the `.sum()` necessary? `tensor.backward()` requires a scalar; `.sum()` on a `[B]` tensor gives that. Worth a one-line explanation.
3. Should we show the per-(layer, position) heatmap with a matplotlib snippet? Probably out of scope for a recipe doc but useful.

## docs/patterns/sae-and-auxiliary-modules.md

1. Pattern B mounts the SAE via `model.transformer.h[LAYER]._module.add_module("sae", sae)`. Does the surrounding Envoy tree pick this up automatically on the next trace, or do you need to call something like `model._refresh()` / re-wrap? Need to verify; the doc currently says "Re-trace to get the new submodule wrapped (If you mounted before any trace, this is automatic)" which is hand-wavy.
2. The `eproperty` "first-class hookable values" example uses `self._module._last_input`, which is not a real attribute on torch modules. Either provide a concrete working example or remove the snippet. Probably should point at `tests/test_transform.py` instead.
3. Are there published SAE checkpoints that load cleanly into nnsight that we could reference (Goodfire / EleutherAI / etc.)? Useful for runnable examples.

## docs/patterns/per-head-attention.md

1. Pattern A in-place writes via `.view(...)`. Confirm that `attn_out.view(B, S, n_heads, head_dim)[:, :, HEAD, :] = 0` actually propagates to `model.transformer.h[L].attn.output[0]` (the tuple element) without a tuple replacement. The doc claims it does because `view` shares storage; need to check on a real GPT-2.
2. The custom Envoy example uses `self._module.n_heads`. For GPT-2's `GPT2Attention`, the attribute is `num_heads` (in some versions) or `nh` (in older ones). Should the doc lift this from `model.config.n_head` instead and pass it via `__init__`?
3. The `envoys={GPT2Attention: AttnHeadsEnvoy}` reference - confirm that the import path is `transformers.models.gpt2.modeling_gpt2.GPT2Attention` and that NNsight's MRO match works on it.

## docs/patterns/multi-prompt-comparison.md

1. The doc says "empty invokes do not trigger `_batch()`". Confirm that statement against `src/nnsight/intervention/batching.py` - is it `_batch` specifically, or `_prepare_input` + `_batch`?
2. The "Pre-batched input (no invokes)" example uses `model.trace(["Hello", "World"])`. For a `LanguageModel`, this should batch via the tokenizer; for a base `NNsight` it would not. Worth being explicit about which models support this.

## docs/patterns/gradient-based-attribution.md

1. Integrated gradients example replaces `model.transformer.wte.output = scaled`. Does writing to `wte.output` on a wrapped HF model actually feed through, given that internally the model also reads embeds via positional embeddings, dropouts, etc.? Need to test on GPT-2 specifically.
2. "Gradient surgery" example writes to `hs.grad[:, :, 100:200] = 0`. Is the gradient tensor a proper leaf such that this in-place write affects subsequent backprop chains, or is it a captured value at a specific moment? Want to be precise.
3. `with metric.sum().backward(retain_graph=True):` - is `retain_graph` a valid kwarg on the nnsight backward context wrapper, or only on `torch.Tensor.backward`? Verify in `src/nnsight/__init__.py` patches.
