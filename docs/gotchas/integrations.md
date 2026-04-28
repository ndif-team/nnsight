---
title: Integration Pitfalls
one_liner: Wrapper-specific traps — LanguageModel needs a tokenizer for pre-loaded models, .source can't be called on submodule fns, auxiliary modules need hook=True, vLLM specifics.
tags: [gotcha, language-model, source, vllm, sae, lora]
related: [docs/models/language-model.md, docs/usage/source.md]
sources: [src/nnsight/modeling/language.py:205, src/nnsight/intervention/source.py:658, src/nnsight/intervention/envoy.py:239, src/nnsight/modeling/vllm/vllm.py:139]
---

# Integration Pitfalls

## TL;DR
- `LanguageModel(hf_model)` on a pre-loaded HF model raises `AttributeError: Tokenizer not found` — you must pass `tokenizer=`.
- `.source` cannot be called on a *module* fn from inside another `.source`; the chained access raises with a message telling you to call `.source` on the submodule's envoy directly.
- Auxiliary modules (SAEs, LoRA adapters) called inside a trace need `module(x, hook=True)` if you want `.input`/`.output` to be accessible on that aux module afterwards. The default routes through `.forward(...)` and bypasses hooks.
- vLLM `tracer.invoke(prompt, temperature=..., top_p=..., max_tokens=...)` forwards these to vLLM's `SamplingParams`.
- vLLM pipeline parallelism is *not* supported. `pipeline_parallel_size` is forced to 1 (`src/nnsight/modeling/vllm/vllm.py:139`). Tensor parallelism (TP) and data parallelism (DP) are supported.

---

## `LanguageModel` on a pre-loaded HF model needs `tokenizer=`

### Symptom
```
AttributeError: Tokenizer not found. If you passed a pre-loaded model to `LanguageModel`,
you need to provide a tokenizer when initializing: `LanguageModel(model, tokenizer=tokenizer)`.
```

### Cause
When you give `LanguageModel` a repo id (string), it loads both the model and tokenizer from HuggingFace via `from_pretrained`. When you give it an already-loaded model object, it has nothing to derive the tokenizer from. The `LanguageModel.__init__` raises this exception (`src/nnsight/modeling/language.py:205`) the first time tokenization is needed.

### Wrong code
```python
from transformers import AutoModelForCausalLM
from nnsight import LanguageModel

hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
model = LanguageModel(hf_model)   # no tokenizer

with model.trace("Hello"):        # AttributeError here
    out = model.lm_head.output.save()
```

### Right code
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel

hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = LanguageModel(hf_model, tokenizer=tokenizer)

with model.trace("Hello"):
    out = model.lm_head.output.save()
```

### Mitigation / how to spot it early
- If you wrap a pre-loaded model, always pass `tokenizer=`.
- If you pass a repo id string, you don't need to.

---

## Don't call `.source` on a module fn from inside another `.source`

### Symptom
```
ValueError: Don't call .source on a module (...) from within another .source.
Call it directly with: <path>.source
```

### Cause
`.source` rewrites the module's forward and registers per-operation hooks. Recursive `.source` works for *function* calls inside a forward, but if one of those operations is itself a *module call* (e.g. `self.c_proj(x)`), the source machinery refuses to descend into it. The error fires from `OperationEnvoy.source` (`src/nnsight/intervention/source.py:658`) when it detects the captured fn is a `torch.nn.Module`.

The fix is to access that submodule's envoy directly — its own `.source` works as a top-level entry.

### Wrong code
```python
with model.trace("Hello"):
    # attempting to chain .source through a submodule call
    out = (
        model.transformer.h[0].attn
        .source.self_c_proj_0       # this is a module call
        .source.<something>         # ValueError
        .output.save()
    )
```

### Right code
```python
with model.trace("Hello"):
    # access the submodule's envoy directly, then its .source
    out = model.transformer.h[0].attn.c_proj.source.<some_op>.output.save()
```

### Mitigation / how to spot it early
- If you need to inspect operations inside a submodule's forward, walk to that submodule's envoy first.
- See [docs/usage/source.md](../usage/source.md) for full source-tracing details.

---

## Auxiliary modules need `hook=True` for `.input`/`.output` access

### Symptom
You add an SAE, LoRA adapter, or any auxiliary `nn.Module` to your model, call it inside a trace, and try to access `aux.output` afterward. Either the access deadlocks waiting for a value that never comes, or you get a missed-provider error.

### Cause
`Envoy.__call__` has a `hook=False` default (`src/nnsight/intervention/envoy.py:239`):

```python
def __call__(self, *args, hook=False, **kwargs):
    return (
        self._module.forward(*args, **kwargs)
        if self.interleaver.current is not None and not hook
        else self._module(*args, **kwargs)
    )
```

When you call an envoy *inside* an interleaving session and don't pass `hook=True`, it routes to `.forward(...)` directly, bypassing PyTorch's `__call__` hook dispatch. That means the sentinel hook (which keeps PyTorch in the dispatch path) is bypassed and dynamically registered one-shot hooks for `.input`/`.output` never fire.

For module calls where you want post-call inspection of `.input`/`.output`, pass `hook=True` to route through the hook-dispatching path.

### Wrong code
```python
# model.sae is an SAE you've added to the envoy tree
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        hs = model.transformer.h[5].output
        recon = model.sae(hs)         # default: hook=False, routes through .forward
        model.transformer.h[5].output[:] = recon

    with tracer.invoke():
        sae_out = model.sae.output.save()   # never fires — no hook was registered
```

### Right code
```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        hs = model.transformer.h[5].output
        recon = model.sae(hs, hook=True)   # routes through __call__, hooks fire
        model.transformer.h[5].output[:] = recon

    with tracer.invoke():
        sae_out = model.sae.output.save()   # OK — hook fired in invoke 1
```

### Mitigation / how to spot it early
- If a deadlock or missed-provider error mentions a path matching an auxiliary module you called manually, check whether you passed `hook=True`.
- Plain modules that the model itself calls (transformer blocks, attention, MLP) are hooked automatically by the model's own forward pass — you only need `hook=True` for *your* explicit calls.

---

## vLLM sampling kwargs route through `SamplingParams`

### Symptom
You write `tracer.invoke("Hello", temperature=0.8, top_p=0.95)` on a vLLM model and it works. You wonder where these kwargs go.

### Cause
The `VLLM` wrapper's batcher forwards sampling-related kwargs to vLLM's `SamplingParams` so each invoke gets its own sampling configuration. Common keys: `temperature`, `top_p`, `top_k`, `max_tokens`, `min_tokens`, `frequency_penalty`, `repetition_penalty`, `n`, `seed`.

### Right code
```python
from nnsight.modeling.vllm import VLLM

model = VLLM("gpt2", tensor_parallel_size=1, gpu_memory_utilization=0.1, dispatch=True)

with model.trace(max_tokens=3) as tracer:
    with tracer.invoke("Hello", temperature=0.8, top_p=0.95):
        samples = list().save()
        for step in tracer.iter[:]:
            samples.append(model.samples.item())
```

### Mitigation / how to spot it early
- If a kwarg you pass to `tracer.invoke(...)` looks like sampling configuration, it's going to `SamplingParams`.
- For values that aren't valid `SamplingParams` fields, vLLM raises at the SamplingParams construction step.

---

## vLLM: pipeline parallelism is not supported

### Symptom
`pipeline_parallel_size > 1` is silently overridden to 1. Or: code expecting different behavior on different stages doesn't work as expected.

### Cause
The `VLLM` wrapper forces `kwargs["pipeline_parallel_size"] = 1` (`src/nnsight/modeling/vllm/vllm.py:139`). nnsight's intervention model assumes a single mediator thread can reach every module — pipeline parallelism splits modules across stages on different GPUs, so no single worker has the full model. Tensor parallelism (TP) and data parallelism (DP) are supported because the batcher gathers/re-shards transparently.

### Wrong assumption
```python
model = VLLM("meta-llama/Llama-3.1-70B", pipeline_parallel_size=2)  # silently overridden
```

### Right approach
- Use TP (`tensor_parallel_size=N`) for sharding a single model across GPUs.
- Use DP for replicating across GPUs.
- For very large models that exceed a single node's TP capacity, the current advice is to use a different deployment topology (e.g. an NDIF deployment) rather than vLLM PP.

### Mitigation / how to spot it early
- If you set `pipeline_parallel_size > 1`, expect it to be 1 in practice.
- See `src/nnsight/modeling/vllm/DISCUSSION.md` for the architectural reason and `IDEAS.md` for any future plans.

---

## Related
- [docs/models/language-model.md](../models/language-model.md) — `LanguageModel` reference.
- [docs/usage/source.md](../usage/source.md) — source-tracing docs.
- [docs/concepts/envoy-and-eproperty.md](../concepts/envoy-and-eproperty.md) — `Envoy.__call__` and the `hook=` flag.
- [docs/gotchas/save.md](save.md) — `.save()` mechanics (universal, including for vLLM).
