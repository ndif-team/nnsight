---
title: TransformersModel
one_liner: Primary wrapper for any HuggingFace transformers task; trace/generate/pipe/scan with tokenization and batching handled by the task's pipeline.
tags: [models, transformers, primary]
related: [docs/models/index.md, docs/models/nnsight-base.md, docs/models/language-model.md, docs/models/vision-language-model.md, docs/models/vllm.md]
sources: [src/nnsight/modeling/transformers.py:161, src/nnsight/modeling/huggingface.py:15, src/nnsight/modeling/mixins/meta.py:116, tests/test_language.py, tests/test_encoder.py, tests/test_vision.py, tests/test_vlm.py, tests/test_modeling.py]
---

# TransformersModel

## What this is for

`nnsight.TransformersModel` is the **primary** wrapper for any HuggingFace `transformers` model. It is backed by a `transformers.pipeline`, so it works for *any* task — text generation, fill-mask, text/image classification, image-text-to-text (VLM), and so on — and leans on the pipeline to tokenize, featurize, template chat, and pad batches. You pick the task (or let it be inferred from the checkpoint) and get the full NNsight tracing API.

Use it for anything you'd load from the HuggingFace Hub with an `AutoModel*` / `pipeline`. `LanguageModel` and `VisionLanguageModel` are now thin deprecated aliases over this class (see [language-model.md](language-model.md) / [vision-language-model.md](vision-language-model.md)).

## Loading

```python
from nnsight import TransformersModel

# task is inferred from the checkpoint when omitted
model = TransformersModel("openai-community/gpt2", dispatch=True)

# or pin it explicitly
model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
```

### Constructor

```python
TransformersModel(
    repo_id,                    # HF repo id string, or a pre-loaded torch.nn.Module
    *,
    task=None,                  # pipeline task; inferred from the checkpoint if None
    tokenizer=None,             # supply one instead of letting the pipeline load it
    processor=None,             # multimodal processor (VLMs)
    image_processor=None,       # vision tasks
    feature_extractor=None,     # audio tasks
    peft=None,                  # HF repo id of a PEFT/LoRA adapter to apply
    revision=None,              # git branch / tag / commit
    dispatch=False,             # True = load real weights now; False = lazy meta build
    rename=None,                # dict of module-path aliases
    **kwargs,                   # forwarded to transformers.pipeline / from_pretrained
)
```

| Parameter | Description |
|-----------|-------------|
| `repo_id` | A HuggingFace repo id string, or an already-instantiated `torch.nn.Module`. |
| `task` | The pipeline task (`"text-generation"`, `"fill-mask"`, `"text-classification"`, `"image-classification"`, `"image-text-to-text"`, ...). If `None`, inferred from the checkpoint. |
| `tokenizer` / `processor` / `image_processor` / `feature_extractor` | Pass one to adopt it instead of letting the pipeline load it. Which of them a task uses varies; the unused ones stay `None` (`transformers.py:266`). |
| `peft` | Repo id of a PEFT adapter grafted onto the base model at load. See `tests/test_language.py:637` for verified PEFT usage. |
| `dispatch` | `True` loads real weights during `__init__`; `False` (default) builds the architecture on the `meta` device and loads weights lazily on the first `trace`/`generate`/`pipe`. |
| `device_map`, `torch_dtype`, `trust_remote_code`, `attn_implementation`, ... | Forwarded to the pipeline / `from_pretrained`. |
| `rename` | Module-path aliases (see [Module renaming](#module-renaming)). |

`kwargs` are split between the `pipeline(...)` factory's own parameters and `model_kwargs` automatically (`transformers.py:97`), so anything HF accepts works.

### Preprocessor attributes

The pipeline and its preprocessors are exposed as attributes. Which are populated depends on the task:

| Task | `tokenizer` | `processor` | `image_processor` | `feature_extractor` |
|------|-------------|-------------|-------------------|---------------------|
| text-generation, fill-mask, text-classification | yes | — | — | — |
| image-classification | — | — | yes | — |
| image-text-to-text (VLM) | yes (from processor) | yes | — | — |

Any of them may be `None`. `model.pipeline` is always the underlying `transformers.Pipeline`. `model.config` / `model.repo_id` / `model.revision` / `model.dispatched` are available too.

## The three ways to run it

`trace`, `generate`, and `pipe` all capture the `with` block and interleave your interventions with the real forward — they differ in *what runs* and *what comes back*.

| Method | What runs | Returns (`tracer.result`) |
|--------|-----------|---------------------------|
| `model.trace(...)` | **one forward** | the model's raw output (e.g. `model.output.logits`) |
| `model.generate(...)` | the model's **`generate`** (decode loop) | generated **token ids** `[batch, seq]` |
| `model.pipe(...)` | the **whole task pipeline** | the pipeline's postprocessed **records** (decoded text, labels, ...) |
| `model.scan(...)` | one forward under **fake tensors** | shapes only, no weights loaded |

> Note the OLD→NEW change: old nnsight's `generate` returned the pipeline's decoded records. That is now `pipe`. `generate` returns raw token ids.

### trace — one forward

```python
model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("The Eiffel Tower is in the city of"):
    hidden = model.transformer.h[-1].output.save()
    logits = model.output.logits.save()

print(hidden.shape)                                   # torch.Size([10, 768])
print(model.tokenizer.decode(logits[0, -1].argmax()))
```

### generate — token ids out

```python
with model.generate("The Eiffel Tower is in the city of", max_new_tokens=3) as tracer:
    ids = tracer.result.save()

print(ids.shape)                          # torch.Size([1, 13])  (10 prompt + 3 new)
print(model.tokenizer.decode(ids[0]))     # "The Eiffel Tower is in the city of Paris, and"
```

`generate` goes through the model with the checkpoint's own settings — **greedy by default** (no `do_sample`), unlike `pipe`, which folds in the checkpoint's `task_specific_params`. Ask for sampling explicitly with `do_sample=True`, `temperature=`, etc. Read the ids off `tracer.result` (preferred). `**kwargs` are forwarded to the model's `generate`, e.g. `max_new_tokens`, `num_return_sequences`, `generation_config=`.

Called directly (no `with`), it just generates and returns the ids:

```python
ids = model.generate("Hello", max_new_tokens=3)       # torch.Tensor [1, N]
```

### pipe — pipeline records out

```python
with model.pipe("The Eiffel Tower is in the city of", max_new_tokens=3) as tracer:
    records = tracer.result.save()

print(records)          # [{'generated_text': 'The Eiffel Tower is in the city of Paris, and'}]
```

`pipe` runs the whole task pipeline end to end (it tokenizes and collates its own input) and returns whatever that task postprocesses to. For a classifier that's label records; for fill-mask, ranked fills; for image-classification, `{"label", "score"}` records. Interventions in the block still land on the real forward:

```python
# image-classification: pipe returns label records
with vit.pipe(image) as tracer:
    result = tracer.result.save()      # [{'label': ..., 'score': ...}]
```

### scan — shapes without weights

```python
model = TransformersModel("openai-community/gpt2")     # not dispatched (meta)

with model.scan("Hello World"):
    shape = model.transformer.h[0].output.shape     # torch.Size([2, 768])

assert model.dispatched is False                        # scan never loads weights
```

`scan` runs the forward under a fake-tensor mode: only shapes/dtypes propagate, no real weights are loaded, and the model is never dispatched. Read `.shape` / `.dtype` inside the block rather than saving tensors out (the values are fakes valid only there).

## Input forms `trace` / `generate` accept

Every invoke is normalized to per-row `input_ids` and left-pad batched into one forward, so mixed formats and unequal lengths combine freely (`transformers.py:570`, verified in `tests/test_language.py:693`):

```python
model.trace("a single prompt")                          # str -> 1 row
model.trace(["prompt one", "prompt two"])               # list[str] -> one row each
model.trace(model.tokenizer("hi").input_ids)            # list[int] -> 1 row
model.trace([ids_a, ids_b])                             # list[list[int]] -> one row each
model.trace(torch.tensor([1, 2, 3]))                    # 1-D tensor -> 1 row
model.trace(model.tokenizer("hi", return_tensors="pt")) # BatchEncoding (positional)
model.trace(input_ids=ids, attention_mask=mask)         # keyword tensors
model.trace(**model.tokenizer("hi", return_tensors="pt"))  # unpacked encoding
```

Chat messages are detected and templated automatically (as `Pipeline.__call__` would):

```python
messages = [{"role": "user", "content": "Who wrote 'Beloved'?"}]
with model.trace(messages):
    ...
```

A raw float feature tensor or a multimodal encoding (one carrying `pixel_values`, `input_features`, ...) is **opaque** — it passes straight to the model untouched and cannot be batched with others (`transformers.py:759`).

### Batching across invokes

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        a = model.output.logits[:, -1].save()
    with tracer.invoke(["Madison Square Garden is in", "The Colosseum is in"]):
        b = model.output.logits[:, -1].save()      # batch of 2
```

Causal decoders left-pad and get mask-derived `position_ids`; encoders (BERT, DistilBERT) keep right padding and need no correction (`transformers.py:851`, `tests/test_encoder.py`). An empty `tracer.invoke()` sees the whole padded batch.

## Generation internals: `generator` / streamer

Generated ids are passed through `model.generator`, a standalone module:

- `tracer.result` — the **preferred** way to read the final ids.
- `model.generator.output` — the same ids; **deprecated**, kept for backwards compat.
- `model.generator.streamer.output` — per-step tokens as they decode (no `tracer.result` equivalent).

```python
# per-step tokens via the streamer
with model.generate(PROMPT, max_new_tokens=3) as tracer:
    for _ in tracer.iter[:3]:
        step = model.generator.streamer.output.save()   # prompt, then 1 token/step
```

## Iterating over generation steps

```python
import nnsight

with model.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
    per_step = nnsight.save([])
    for step in tracer.iter[:3]:
        per_step.append(model.transformer.h[0].output)
# per_step[0] is the full prompt; per_step[1:] are one cached token each
```

## Interventions

```python
# in-place zeroing
with model.trace(PROMPT):
    model.transformer.h[-1].output[:] = 0
    logits = model.output.logits.save()

# replacement / logit lens (ad-hoc module call, out of order)
with model.trace("The Eiffel Tower is in the city of"):
    hidden = model.transformer.h[-1].output
    logits = model.lm_head(model.transformer.ln_f(hidden))
    tokens = logits.argmax(dim=-1).save()

# gradients
with model.trace(PROMPT):
    a = model.transformer.h[0].output
    loss = model.output.logits.sum()
    with loss.backward():
        grad = a.grad.save()

# skip a module
with model.trace(PROMPT):
    model.transformer.h[0].mlp.skip(torch.zeros_like(mlp_out))

# cache activations
with model.trace(PROMPT) as tracer:
    cache = tracer.cache(include_inputs=True)
# cache["model.transformer.h.0"].output
```

## Module renaming

```python
model = TransformersModel(
    "openai-community/gpt2", task="text-generation", dispatch=True,
    rename={"transformer.h": "layers", "mlp": "my_mlp"},
)
assert model.layers[0] is model.transformer.h[0]
with model.trace(PROMPT):
    out = model.layers[0].my_mlp.output.save()      # alias
    ref = model.transformer.h[0].mlp.output.save()  # original still works
```

Aliases are honored in `tracer.cache()` keys too (`tests/test_language.py:487`).

## Dispatch behavior

- `dispatch=False` (default): only configs download; the architecture is built on the `meta` device. The Envoy tree is fully usable for writing intervention code.
- `dispatch=True`: real weights load during `__init__`.
- First `trace`/`generate`/`pipe` auto-dispatches if needed (`meta.py:177`). `scan` does **not** dispatch.
- Call `model.dispatch()` to force loading.

## Remote

`TransformersModel` is remoteable. `model.to_model_key()` identifies the checkpoint (repo id + revision, canonicalized via the Hub), and `trace(..., remote=True)` runs on NDIF. The deprecated `LanguageModel` / `VisionLanguageModel` aliases share this class's remote key (`language.py:51`), so a model deployed as a `TransformersModel` is reachable when wrapped as either. See [docs/remote/](../remote/).

## Gotchas

- **`generate` vs `pipe`.** `generate` returns token ids and is greedy by default; `pipe` returns decoded records and folds in the checkpoint's sampling `task_specific_params`. If old code relied on `generate` returning text, switch it to `pipe`.
- **`save()` outside a trace raises.** `.save()` / `nnsight.save(...)` now raises if there's no active trace (it was a silent no-op in old nnsight).
- **`scan` needs `dispatch=False` to be cheap** but works either way; it never loads weights.
- **Opaque inputs can't be batched.** A multimodal encoding (with `pixel_values`) or a raw float tensor must be a lone invoke — batching several raises `NotImplementedError`.
- **`tracer.result` is preferred over `model.generator.output`** for finished ids; the latter is deprecated.

## Related

- [docs/models/index.md](index.md) — decision tree
- [docs/models/nnsight-base.md](nnsight-base.md) — the base `NNsight` wrapper
- [docs/models/language-model.md](language-model.md) / [vision-language-model.md](vision-language-model.md) — deprecated aliases
- [docs/models/vllm.md](vllm.md) — high-throughput serving path
- `src/nnsight/modeling/transformers.py` — source
- `tests/test_language.py`, `tests/test_encoder.py`, `tests/test_vision.py`, `tests/test_vlm.py` — runnable examples
