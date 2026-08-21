---
title: Integration Pitfalls
one_liner: Wrapper-specific traps — deprecated model aliases, .source can't drill into submodule calls, auxiliary modules need hook=True, vLLM specifics.
tags: [gotcha, transformers, source, vllm, sae, lora]
related: [docs/models/transformers-model.md, docs/usage/source.md, docs/models/vllm.md]
sources: [src/nnsight/modeling/language.py:19, src/nnsight/intervention/source.py:536, src/nnsight/intervention/envoy.py:726, src/nnsight/modeling/vllm/vllm.py]
---

# Integration Pitfalls

## TL;DR
- Use `TransformersModel(repo_id, task=...)`. `LanguageModel` / `VisionLanguageModel` are **deprecated** thin subclasses that warn on construction — the tokenizer/processor comes from the task's pipeline, so there's no "pass a tokenizer" step.
- `.source` cannot drill *into* a call that is a **submodule** — it raises `SourceNotAvailable` telling you to call `.source` on that submodule's envoy directly.
- Auxiliary modules (SAEs, LoRA adapters) you attach and call inside a trace need `module(x, hook=True)` for their internals to be observable at `.submodule.output` — and the observation happens in a *later* trace (apply via `edit()`).
- vLLM: pass sampling settings (`temperature`, `top_p`, `max_tokens`, ...) to `trace`/`invoke`; read/edit `model.logits` and `model.samples`; choose `VLLM(..., mode="sync"|"async")`.

---

## Prefer `TransformersModel`; aliases warn

### Symptom
```
DeprecationWarning: LanguageModel is deprecated; use TransformersModel(repo_id, task='text-generation') instead.
```

### Cause
`LanguageModel` (`src/nnsight/modeling/language.py:19`) and `VisionLanguageModel` are backwards-compatible names over `TransformersModel`. `TransformersModel` is backed by a `transformers.pipeline`, which loads the tokenizer/processor itself — so `model.tokenizer` is populated without you passing one.

### Right code
```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
with model.trace("Hello world"):
    out = model.output.logits.save()
```

`TransformersModel` has three run modes: `trace` (one forward), `generate` (returns token ids — read `tracer.result`), and `pipe` (runs the whole task pipeline, returns decoded records). See [docs/models/transformers-model.md](../models/transformers-model.md).

### Mitigation
- A pre-loaded HF model still works: `TransformersModel(model)` wraps it in a pipeline, inferring the task and sourcing the tokenizer from the model's `name_or_path` (pass `task=` / `tokenizer=` if it can't be inferred — e.g. a model built without a `name_or_path`). A bare non-HF `torch.nn.Module` goes through `NNsight(module)`.

---

## `.source` can't drill into a submodule call

### Symptom
```
SourceNotAvailable: 'self_c_proj_0' calls a submodule; call `.source` on that submodule directly instead of drilling into the call
```

### Cause
`.source` decomposes a `forward` into its operations. You can drill into a plain *function* call with `op.source`, but if the call target is a `torch.nn.Module` (e.g. `self.c_proj(x)`), source refuses to descend — that module has its own envoy and hooks (`SourceEnvoy.source`, `src/nnsight/intervention/source.py:536`). Operation names include the full dotted path joined with `_` (`self.c_proj(x)` → `self_c_proj_0`).

### Wrong / Right
```python
# inspect available ops first
with model.trace("Hello world"):
    ...
# print(model.transformer.h[0].mlp.source)   ->  self_c_fc_0, self_act_0, self_c_proj_0, self_dropout_0

# wrong — c_proj is a submodule
with model.trace("Hello world"):
    model.transformer.h[0].mlp.source.self_c_proj_0.source     # SourceNotAvailable

# right — go to the submodule's own envoy
with model.trace("Hello world"):
    act = model.transformer.h[0].mlp.source.self_act_0.output.save()   # a plain-fn op
    proj = model.transformer.h[0].mlp.c_proj.output.save()            # the submodule directly
```

### Mitigation
- To inspect ops inside a submodule's forward, walk to that submodule (`...mlp.c_proj`) and use *its* `.source`. See [docs/usage/source.md](../usage/source.md).

---

## Auxiliary modules need `hook=True` (and observe in a later trace)

### Symptom
You attach an SAE/adapter, call it inside a trace, and reading `aux.submodule.output` raises `OutOfOrderError` / never fires.

### Cause
`Envoy.__call__` defaults to `hook=False`: while interleaving it runs the module normally but stands the trace down, so the module's submodules aren't observable. Pass `hook=True` to let the trace watch the call. Apply the aux module in an `edit()` (a default intervention replayed on every trace) and observe its internals in a subsequent trace.

### Right code
```python
model.transformer.h[0].adapter = MyAdapter().to(model.device)

with model.edit() as (tracer, edited):
    acts = edited.transformer.h[0].output
    edited.transformer.h[0].output[:] = edited.transformer.h[0].adapter(acts, hook=True)

with edited.trace("Hello world"):
    inner = edited.transformer.h[0].adapter.inner.output.save()   # observable
```

To apply it on *every* generation step, put the passthrough under an `iter` loop in an `inplace=True` edit (see the `Envoy.__call__` docstring).

### Mitigation
- `OutOfOrderError`/missed-value mentioning a path of a module you called manually → you likely forgot `hook=True`. Modules the model itself calls (blocks, attn, mlp) are hooked automatically.

---

## vLLM specifics

### Cause / usage
On `VLLM`, each invoke is its own vLLM request, so sampling settings are passed to `trace`/`invoke` rather than configured on the model, and the per-step intervention points are `model.logits` (pre-sample logits) and `model.samples` (drawn token ids):

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("gpt2", dispatch=True)                    # mode="sync" default; "async" for streaming
with model.trace("The Eiffel Tower is in", temperature=0.0) as tracer:
    model.transformer.h[8].output[:] = 0
    logits = model.logits.save()
    ids = model.samples.save()
```

Under `tracer.iter`, each pass sees the next decoded step's `logits`/`samples`.

### Mitigation
- Sampling kwargs (`temperature`, `top_p`, `top_k`, `max_tokens`, ...) go to `trace`/`invoke`; invalid ones raise when vLLM builds its sampling params.
- For engine/parallelism setup (tensor parallelism, sync vs async), see [docs/models/vllm.md](../models/vllm.md).

---

## Related
- [docs/models/transformers-model.md](../models/transformers-model.md) — the primary HF wrapper.
- [docs/usage/source.md](../usage/source.md) — source-tracing reference.
- [docs/models/vllm.md](../models/vllm.md) — vLLM model reference.
- [docs/gotchas/save.md](save.md) — `.save()` mechanics (universal).
