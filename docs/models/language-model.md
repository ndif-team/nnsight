---
title: LanguageModel
one_liner: Wraps HF transformers AutoModelForCausalLM with tokenization, batching, generation, and meta-loading.
tags: [models, language, transformers]
related: [docs/models/index.md, docs/models/nnsight-base.md, docs/models/vision-language-model.md, docs/models/vllm.md, docs/remote/index.md]
sources: [src/nnsight/modeling/language.py:21, src/nnsight/modeling/transformers.py:11, src/nnsight/modeling/huggingface.py:14, src/nnsight/modeling/mixins/loadable.py:8, src/nnsight/modeling/mixins/meta.py:12]
---

# LanguageModel

## What this is for

`nnsight.LanguageModel` wraps a HuggingFace `transformers` causal LM (or any model loadable via an `AutoModel*` class) with:

- Repo-id loading via `AutoModelForCausalLM.from_pretrained()` (or any `automodel=` you choose)
- Automatic tokenization of string inputs
- Padded batching across multiple `tracer.invoke(...)` calls
- Multi-token `.generate()` with token-by-token access via `model.generator`
- Meta-tensor lazy loading + automatic dispatch on first trace

It is the recommended wrapper for any HuggingFace transformers language model.

## When to use / when not to use

Use `LanguageModel` when:
- You want to load a HuggingFace causal LM by repo ID (`"openai-community/gpt2"`, `"meta-llama/Llama-3.1-8B"`, etc.).
- You want automatic tokenization — pass strings, get hidden states.
- You want HF-style `.generate()` for multi-token output with intervention support.
- You want to batch multiple prompts in one forward pass (left-padded by default).

Do not use `LanguageModel` when:
- You have a vision-language model — use [`VisionLanguageModel`](vision-language-model.md) instead. It extends `LanguageModel` with an `AutoProcessor`.
- You need vLLM-grade throughput / continuous batching / TP serving — use [`VLLM`](vllm.md).
- You have a non-HF custom architecture — use [`NNsight`](nnsight-base.md) with a pre-loaded module.

## Loading

```python
from nnsight import LanguageModel

model = LanguageModel(
    "openai-community/gpt2",
    device_map="auto",
    dispatch=True,
)
```

### Constructor

```python
LanguageModel(
    repo_id_or_module,        # str repo ID, or a pre-loaded torch.nn.Module
    *,
    tokenizer=None,           # PreTrainedTokenizer; required when wrapping a pre-loaded model
    automodel=AutoModelForCausalLM,
    config_model=None,        # PretrainedConfig override
    revision=None,            # git revision / branch / commit
    rename=None,              # dict of module-path aliases
    envoys=None,              # Envoy class or dict (see NNsight base)
    dispatch=False,           # True = load weights now; False = lazy meta tensors
    meta_buffers=True,        # buffers also on meta device when dispatch=False
    import_edits=False,       # restore previously exported edits
    tokenizer_kwargs=None,    # forwarded to AutoTokenizer.from_pretrained
    **kwargs,                 # forwarded to AutoModelForCausalLM.from_pretrained / from_config
)
```

#### Key parameters

| Parameter | Description |
|-----------|-------------|
| First positional arg | Either a HuggingFace repo ID string, or an already-instantiated `torch.nn.Module` (in which case you **must** pass `tokenizer=`). See `language.py:200` and `loadable.py:30`. |
| `tokenizer` | A `PreTrainedTokenizer`. Required when wrapping a pre-loaded model; otherwise auto-loaded from the repo. |
| `automodel` | The `AutoModel` class to use for `from_pretrained` / `from_config`. Defaults to `AutoModelForCausalLM`. Can be an `AutoModel` subclass or a string name resolvable from `transformers.models.auto.modeling_auto` (e.g. `"AutoModelForSeq2SeqLM"`). See `transformers.py:36`. |
| `dispatch` | If `True`, load real weights immediately. If `False` (default), build the model with meta tensors via `from_config`; weights load on first `.trace()` / `.generate()` or when you call `.dispatch()` explicitly. See `mixins/meta.py:38`. |
| `device_map` | Forwarded to HF Accelerate. `"auto"` distributes layers across all available GPUs (and offloads to CPU if needed). Other options: `"cuda"`, `"cpu"`, or a custom dict. |
| `torch_dtype` | Forwarded to HF (e.g. `torch.float16`, `torch.bfloat16`). |
| `trust_remote_code` | Forwarded to HF for repos with custom modeling code. |
| `attn_implementation` | Forwarded to HF (`"eager"`, `"flash_attention_2"`, `"sdpa"`). |
| `rename` | Dict of module-path aliases. E.g. `{"transformer.h": "layers", "mlp": "feedforward"}`. Both original and aliased paths work. See `Envoy` docs. |
| `envoys` | Class attribute or constructor kwarg for descendant Envoy types. See [docs/models/nnsight-base.md](nnsight-base.md). |
| `revision` | Git branch / tag / commit to pin. |
| `tokenizer_kwargs` | Forwarded to `AutoTokenizer.from_pretrained`. Padding side defaults to `"left"` if unspecified (set in `language.py:168`). |

All other `**kwargs` are forwarded to `automodel.from_pretrained()` / `from_config()`. So anything HF accepts here works.

### Dispatch behavior

- `dispatch=False` (default): Only the config is downloaded. The model architecture is built with `from_config(...)` inside `accelerate.init_empty_weights()`, so all parameters are on the `meta` device. The Envoy tree is fully usable for writing intervention code, but weights aren't loaded yet.
- `dispatch=True`: Real weights are downloaded and loaded via `from_pretrained(...)` immediately during `__init__`.
- Either way, the first `.trace()` / `.generate()` call automatically calls `.dispatch()` if needed (see `mixins/meta.py:97`).

You can call `model.dispatch()` manually at any point to force loading.

### Wrapping a pre-loaded model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel

hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = LanguageModel(hf_model, tokenizer=tokenizer)   # tokenizer is required
```

#### Common error: `AttributeError: Tokenizer not found.`

```
AttributeError: Tokenizer not found. If you passed a pre-loaded model to `LanguageModel`,
you need to provide a tokenizer when initializing: `LanguageModel(model, tokenizer=tokenizer)`.
```

`LanguageModel._tokenize` (`src/nnsight/modeling/language.py:200`) needs a tokenizer to convert string input into token IDs:

```python
if self.tokenizer is None:
    if self.repo_id is not None:
        self._load_tokenizer(self.repo_id, **kwargs)
    else:
        raise AttributeError(
            "Tokenizer not found. If you passed a pre-loaded model to "
            "`LanguageModel`, you need to provide a tokenizer when initializing: "
            "`LanguageModel(model, tokenizer=tokenizer)`."
        )
```

`LanguageModel` only auto-loads a tokenizer when it was constructed from a HuggingFace **repo id string**. When you pass an already-loaded `nn.Module` instance there is no repo id to look up, so the tokenizer must be supplied explicitly.

**Common triggers**

- Building the model yourself and wrapping it: `LanguageModel(AutoModelForCausalLM.from_pretrained(...))` without `tokenizer=`.
- Passing a custom `nn.Module` that isn't a HuggingFace model to `LanguageModel`.
- Loading the wrapper from a checkpoint that didn't persist `repo_id` and re-using it with string inputs.

**Fixes**

```python
# WRONG — pre-loaded model, no tokenizer supplied
hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
model = LanguageModel(hf_model)
with model.trace("Hello"):                   # AttributeError on tokenize
    ...

# FIXED — pass tokenizer alongside the pre-loaded model
hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
tok = AutoTokenizer.from_pretrained("gpt2")
model = LanguageModel(hf_model, tokenizer=tok)

# ALSO FIXED — let LanguageModel load both from the repo id
model = LanguageModel("gpt2")                # tokenizer auto-loaded
```

If you have no tokenizer at all, you can still pass tensor input directly — `_tokenize` short-circuits on `torch.Tensor`:

```python
input_ids = torch.tensor([[1, 2, 3]])
with model.trace(input_ids):
    ...
```

**How to avoid**

- When wrapping a pre-loaded model, always pair it with a tokenizer in the constructor.
- If you subclass `LanguageModel` for a custom architecture, override `_load_tokenizer` so future `repo_id`-based loads work.
- For tensor-only pipelines (already-tokenized data), pass tensors and the error never fires.

## Canonical pattern

```python
from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

with model.trace("The Eiffel Tower is in"):
    hidden = model.transformer.h[-1].output.save()
    logits = model.lm_head.output.save()

print(hidden.shape)
print(model.tokenizer.decode(logits.argmax(dim=-1)[0]))
```

### Multi-token generation

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    output = tracer.result.save()        # preferred

print(model.tokenizer.decode(output[0]))
```

`tracer.result` is the recommended path for grabbing the final generated token IDs. `model.generator.output.save()` still works (and is what the streamer-based machinery writes to internally), but new code should prefer `tracer.result`.

### Iterate over generation steps

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    tokens = list().save()
    for step in tracer.iter[:]:
        tokens.append(model.lm_head.output[0][-1].argmax(dim=-1))
```

### Batching multiple prompts

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        out_a = model.lm_head.output[:, -1].save()

    with tracer.invoke(["Madison Square Garden is in", "The Colosseum is in"]):
        out_b = model.lm_head.output[:, -1].save()    # batch of 2
```

### Pre-tokenized inputs

```python
ids = model.tokenizer("Hello", return_tensors="pt")["input_ids"]

with model.trace(input_ids=ids):
    out = model.lm_head.output.save()
```

`_prepare_input()` (`language.py:241`) accepts strings, lists of strings, lists of token IDs, tensors, dicts, or `BatchEncoding` objects.

## Special properties

| Attribute / property | Description | Source |
|----------------------|-------------|--------|
| `model.tokenizer` | The `PreTrainedTokenizer`. Loaded from the repo or supplied via `tokenizer=`. Padding side defaults to `"left"`. | `language.py:76`, `language.py:164` |
| `model.config` | The HuggingFace `PretrainedConfig`. Available after construction even with `dispatch=False`. | `transformers.py:38` |
| `model.automodel` | The `AutoModel` class used for loading (default `AutoModelForCausalLM`). | `transformers.py:40` |
| `model.repo_id` | The HuggingFace repo ID string (or `name_or_path` of a pre-loaded module). | `huggingface.py:46` |
| `model.revision` | The git revision pinned at load time. | `huggingface.py:51` |
| `model.dispatched` | Boolean — whether real weights are loaded. | `mixins/meta.py:47` |
| `model.generator` | A `LanguageModel.Generator` wrapper module that captures the final generation output. The inner `model.generator.streamer` receives tokens as they are generated (used internally by `.generate()`). Prefer `tracer.result` over `model.generator.output` in new code. | `language.py:44`, `language.py:80` |
| `model._model` / `model._module` | The underlying HuggingFace `PreTrainedModel`. | base `NNsight` |

`tracer.result` is the **preferred** way to read the final generated token IDs (it's provided by `Envoy.interleave()`). `model.generator.output` is kept for backwards compatibility and still works.

### Module renaming

```python
model = LanguageModel(
    "openai-community/gpt2",
    rename={
        "transformer.h": "layers",
        "mlp": "feedforward",
    },
    dispatch=True,
)

with model.trace("Hello"):
    out1 = model.layers[0].feedforward.output.save()       # alias
    out2 = model.transformer.h[0].mlp.output.save()        # original still works
```

## Limitations

- **Generation kwargs go to HF `generate()`.** Anything passed to `.generate()` after the prompt (e.g. `max_new_tokens`, `do_sample`, `temperature`) is forwarded to `transformers.GenerationMixin.generate()` via `__nnsight_generate__` (see `language.py:140`). For non-string config kwargs that the tokenizer also accepts, name collisions can be tricky — see `_TOKENIZER_KWARGS` at `language.py:219` for the list of names that are routed to the tokenizer.
- **CUDA graph compile config is patched.** When `generation_config.compile_config` exists, NNsight forces `fullgraph=False` and `dynamic=True` (`language.py:119`). This is required for hook firing to work and is silent.
- **Padding side is left by default.** Tokenizer padding side is forced to `"left"` if not explicitly set (`language.py:168`). This matches causal-LM conventions but can surprise people loading encoder-only models.
- **One AutoModel class per instance.** If your model needs `AutoModelForSeq2SeqLM` or `AutoModel`, pass `automodel=AutoModelForSeq2SeqLM` (or the class name as a string).
- **No vLLM / continuous batching.** For high-throughput serving, use [`VLLM`](vllm.md).

## Gotchas

- **Tokenizer required when wrapping pre-loaded models.** `AttributeError: Tokenizer not found.` at first `.trace()` if you forgot. See [Wrapping a pre-loaded model](#wrapping-a-pre-loaded-model) above for the full error message and fixes.
- **`dispatch=False` means weights load lazily.** A long pause on first `.trace()` is expected — that's the actual model download. Pass `dispatch=True` if you want loading to happen during `__init__`.
- **`device_map="auto"` requires `accelerate`.** `accelerate` is already a hard dep of nnsight via `meta.py`, so this works out of the box.
- **`tracer.result` vs `.generator.output`.** Both refer to the final generation output. `tracer.result` is the newer, preferred path. `model.generator.output` is kept for backwards compatibility and for the streamer hook.
- **Cross-invoke variable sharing.** Variables from one invoke are visible in later invokes (`CONFIG.APP.CROSS_INVOKER` is `True` by default). When two invokes both access the same module, you need a `tracer.barrier(n)` to synchronize.

## Related

- [docs/models/index.md](index.md) — pick the right wrapper
- [docs/models/nnsight-base.md](nnsight-base.md) — the underlying `NNsight` class
- [docs/models/vision-language-model.md](vision-language-model.md) — extends `LanguageModel` with image support
- [docs/models/vllm.md](vllm.md) — high-throughput alternative
- [docs/remote/](../remote/) — running `LanguageModel` traces on NDIF (`remote=True`)
- `src/nnsight/modeling/language.py` — `LanguageModel` source
- `src/nnsight/modeling/transformers.py` — `TransformersModel` (AutoConfig / AutoModel base)
