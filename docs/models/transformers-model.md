---
title: TransformersModel
one_liner: Primary wrapper for any HuggingFace transformers task; trace/generate/pipe/scan with tokenization and batching handled by the task's pipeline.
tags: [models, transformers, primary]
related: [docs/models/index.md, docs/models/nnsight-base.md, docs/models/language-model.md, docs/models/vision-language-model.md, docs/models/vllm.md]
sources: [src/nnsight/modeling/transformers.py, src/nnsight/modeling/huggingface.py, src/nnsight/modeling/mixins/meta.py, tests/test_language.py, tests/test_encoder.py, tests/test_vision.py, tests/test_vlm.py, tests/test_chunked_tasks.py, tests/test_modeling.py]
---

# TransformersModel

## What this is for

`nnsight.TransformersModel` is the **primary** wrapper for any HuggingFace `transformers` model. It is backed by a `transformers.pipeline`, so it works for any task the pipeline factory knows — text generation, fill-mask, text/image classification, token classification, ASR, image-text-to-text (VLM) — and leans on the pipeline to tokenize, featurize, template chat, and pad batches. You pick the task (or let it be inferred from the checkpoint) and get the full NNsight tracing API. Two tasks are the exceptions under `trace`: `mask-generation`, whose preprocessing runs the model itself, and `keypoint-matching`, whose unit input is a pair of images that the list convention would split. Each is refused with a message naming its escape hatches — `model.pipe(...)` for the whole task, or a forward on an encoding you build with the model's own processor; see [Chunked tasks](#chunked-tasks).

Use it for anything you'd load from the HuggingFace Hub with an `AutoModel*` / `pipeline`. `LanguageModel` and `VisionLanguageModel` are now thin deprecated aliases over this class (see [language-model.md](language-model.md) / [vision-language-model.md](vision-language-model.md)).

## Loading

```python
from nnsight import TransformersModel

# task is inferred from the checkpoint when omitted
model = TransformersModel("openai-community/gpt2", dispatch=True)

# or pin it explicitly
model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
```

Inferring the task asks the Hub for the checkpoint's metadata, and a fully cached
checkpoint does not change that. Under `HF_HUB_OFFLINE=1` the first form raises
`RuntimeError: You cannot infer task automatically within 'pipeline' when using
offline mode`, so pass `task=` on an air-gapped machine or a cluster node with no
outbound network.

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
                                # (dtype, device_map, attn_implementation, ...)
)
```

| Parameter | Description |
|-----------|-------------|
| `repo_id` | A HuggingFace repo id string, or an already-instantiated `torch.nn.Module`. |
| `task` | The pipeline task (`"text-generation"`, `"fill-mask"`, `"text-classification"`, `"image-classification"`, `"image-text-to-text"`, ...). If `None`, inferred from the checkpoint — which asks the Hub, so pass it explicitly when you are offline (see [Loading](#loading)). |
| `tokenizer` / `processor` / `image_processor` / `feature_extractor` | Pass one to adopt it instead of letting the pipeline load it. Which of them a task uses varies; the unused ones stay `None` (`transformers.py`, `_preprocessor_sources`). |
| `peft` | Repo id of a PEFT adapter grafted onto the base model at load. See `tests/test_language.py` for verified PEFT usage. |
| `dispatch` | `True` loads real weights during `__init__`; `False` (default) builds the architecture on the `meta` device and loads weights lazily on the first `trace`/`generate`/`pipe`. |
| `dtype` | Forwarded. A torch dtype, or a quantization name (`"nf4"`, `"int8"`, ...) — see [quantization.md](quantization.md). The transformers 4 spelling `torch_dtype` is still accepted. |
| `device_map`, `trust_remote_code`, `attn_implementation`, ... | Forwarded to the pipeline / `from_pretrained`. |
| `rename` | Module-path aliases (see [Module renaming](#module-renaming)). |

`kwargs` are split between the `pipeline(...)` factory's own parameters and `model_kwargs` automatically (`transformers.py`, `_split_pipeline_kwargs`), so anything HF accepts works.

### Preprocessor attributes

The pipeline and its preprocessors are exposed as attributes. Which are populated depends on the task:

| Task | `tokenizer` | `processor` | `image_processor` | `feature_extractor` |
|------|-------------|-------------|-------------------|---------------------|
| text-generation, fill-mask, text-classification | yes | — | — | — |
| image-classification | — | — | yes | — |
| image-text-to-text (VLM) | yes (from processor) | yes | — | — |

A task that has no text side leaves `tokenizer` as `None` — `image-classification` and `image-feature-extraction` populate only `image_processor`. A VLM has both: its `processor` is the object that pairs image and text, and `tokenizer` is the text half of that processor rather than a separately loaded one. `model.pipeline` is always the underlying `transformers.Pipeline`, and `model.config` / `model.repo_id` / `model.revision` / `model.dispatched` are available too.

## The three ways to run it

`trace`, `generate`, and `pipe` all capture the `with` block and interleave your interventions with the real forward — they differ in *what runs* and *what comes back*.

| Method | What runs | Returns (`tracer.result`) |
|--------|-----------|---------------------------|
| `model.trace(...)` | **one forward** | the model's raw output (e.g. `model.output.logits`) |
| `model.generate(...)` | the model's **`generate`** (decode loop) | generated **token ids** `[batch, seq]` |
| `model.pipe(...)` | the **whole task pipeline** | the pipeline's postprocessed **records** (decoded text, labels, ...) |
| `model.scan(...)` | one forward under **fake tensors** | shapes only, no weights loaded |


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

Every invoke is normalized to per-row `input_ids` and left-pad batched into one forward, so mixed formats and unequal lengths combine freely (`transformers.py`, `_preprocess_invoke`; verified in `tests/test_language.py`):

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

A raw float feature tensor or a multimodal encoding (one carrying `pixel_values`, `input_features`, ...) is **opaque** — it passes straight to the model untouched and cannot be batched with others (`transformers.py`, `_is_opaque`).

### Batching across invokes

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        a = model.output.logits[:, -1].save()
    with tracer.invoke(["Madison Square Garden is in", "The Colosseum is in"]):
        b = model.output.logits[:, -1].save()      # batch of 2
```

Causal decoders left-pad and get mask-derived `position_ids`; encoders (BERT, DistilBERT) keep right padding and need no correction (`transformers.py`, `_supply_position_ids`; `tests/test_encoder.py`). An empty `tracer.invoke()` sees the whole padded batch.

## Chunked tasks

Some tasks split one input into several encodings and forward each on its own: token windows past the model's length limit in `token-classification`, one entailment pair per candidate label in `zero-shot-classification`, a long recording's windows in `automatic-speech-recognition`, and the same shape in `document-question-answering` and `zero-shot-object-detection`. Those encodings become **rows of the trace's one forward**, which is what the pipeline itself does at a `batch_size` of its chunk count, so a read inside the block sees one row per chunk in the order the task yields them:

```python
ner = TransformersModel(NER_REPO, task="token-classification", dispatch=True)

with ner.trace("John lives in Paris"):
    logits = ner.output.logits.save()
assert logits.shape[0] == 1                        # one window, one row

with ner.trace(" ".join(["John lives in Paris"] * 200), stride=16):
    logits = ner.output.logits.save()
assert logits.shape[0] == 7                        # seven windows, seven rows
```

An edit inside the block reaches every row, because the rows are the forward's rather than something the trace assembled afterwards.

Two consequences:

- **A chunked invoke is the whole batch.** The row count belongs to the task and is only known after preprocessing, while the batcher counts one row per invoke before that — so a second invoke would name rows belonging to the first. It is refused with `NotImplementedError: task='zero-shot-classification' splits this invoke into 3 forward rows, and a batched trace gives an invoke the rows its input has ...`.
- **`mask-generation` has no forward to trace.** Its preprocessing runs the model to embed the image and then yields one input per batch of candidate points, so the encoder would run outside the trace. It is refused with a message pointing at `model.pipe(image)` for the whole task, or at a forward on an encoding you build with `model.image_processor`.

`keypoint-matching` is the other refused task, for a different reason: its unit input is a *pair* of images, which a trace's list convention (one prompt per element) would split. Run the whole task with `model.pipe([image_a, image_b])`, or trace one forward on an encoding you build yourself: `model.image_processor(images=[image_a, image_b], return_tensors='pt')`.

A task whose input is a dict — `{"image": ..., "question": ..., "word_boxes": [...]}` for `document-question-answering`, `{"image": ..., "candidate_labels": [...]}` for `zero-shot-object-detection`, `{"table": ..., "query": ...}` for `table-question-answering` — goes through the task's own preprocessing, including the `_args_parser` step `Pipeline.__call__` would run (which is where `table-question-answering` builds its `pd.DataFrame`). A mapping carrying tensors is still read as a model encoding (`transformers.py`, `_is_task_input`). A dual-encoder zero-shot task (`zero-shot-image-classification`, `zero-shot-audio-classification`) nests the candidate labels' text encoding inside its preprocess row; the trace merges those tensors into its one forward, so `logits_per_image` / `logits_per_audio` reads `(1, n_labels)` — one text row per candidate label against the single image/audio row.

Row counts across the five, measured on the checkpoints `tests/test_chunked_tasks.py` uses plus `openai/whisper-tiny`:

| Task | Input | Read inside the block |
|---|---|---|
| `token-classification` | `"John lives in Paris"` | `logits` `(1, 18, 2)` |
| `token-classification` | the same 200x, `stride=16` | `logits` `(7, 512, 2)` |
| `zero-shot-classification` | one sequence, 3 `candidate_labels=` | `logits` `(3, 2)` |
| `zero-shot-object-detection` | `{"image", "candidate_labels"}`, 2 labels | `logits` `(2, 256, 1)` |
| `document-question-answering` | `{"image", "question", "word_boxes"}` | `start_logits` `(1, 25)` |
| `automatic-speech-recognition` | 70 s of audio, `chunk_length_s=30` | `encoder.layers[0].output` `(3, 1500, 384)` |

**An encoder-decoder ASR trace needs `decoder_input_ids=`.** `trace` runs one forward, and a seq2seq model with no decoder ids builds decoder embeddings for itself, then rejects the pair it just made: `ValueError: You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time`, which names nothing the caller wrote. Pass one row of `decoder_start_token_id` per chunk:

```python
ids = torch.tensor([[asr.config.decoder_start_token_id]] * 3)
with asr.trace(audio, chunk_length_s=30, decoder_input_ids=ids):
    encoded = asr.model.encoder.layers[0].output.save()   # (3, 1500, 384)
```

`generate` and `pipe` run the decode loop themselves and need none of this.

## Finding the image span in a VLM

An `image-text-to-text` model's prompt is mostly image: the processor expands the
template's single `<image>` placeholder into one token per image patch, and those
positions are where an intervention on the image goes. Find them by matching the
config's image token id:

```python
enc = model.processor(images=image, text=prompt, return_tensors="pt")
span = (enc["input_ids"] == model.config.image_token_id).nonzero()
```

**Only processor-built ids carry that span.** On `llava-hf/llava-interleave-qwen-0.5b-hf`
the processor returns 742 ids of which **729** are the image token; the same text
through `model.tokenizer` alone returns 14 ids with the placeholder still a single
token. Building ids one way and `pixel_values` the other gives
`ValueError: Image features and image tokens do not match, tokens: 1, features: 746496`,
which reports the mismatch in features rather than in what you passed.

Build the encoding with `model.processor` and pass it whole, and the count lines
up. The same encoding is what `trace` gives the model: a multimodal encoding is
opaque, so it goes straight through and cannot share a batch with another invoke.

## Generation internals: `generator` / streamer

Generated ids are passed through `model.generator`, a standalone module:

- `tracer.result` — the **preferred** way to read the final ids.
- `model.generator.output` — the same ids; **deprecated**, and reading or writing it warns.
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

Aliases are honored in `tracer.cache()` keys too (`tests/test_language.py`).

## Dispatch behavior

- `dispatch=False` (default): only configs download; the architecture is built on the `meta` device. The Envoy tree is fully usable for writing intervention code.
- `dispatch=True`: real weights load during `__init__`.
- First `trace`/`generate`/`pipe` auto-dispatches if needed (`mixins/meta.py`, `Meta.interleave`). `scan` does **not** dispatch.
- Call `model.dispatch()` to force loading.

## Remote

`TransformersModel` is remoteable. `model.to_model_key()` identifies the checkpoint (repo id + revision, canonicalized via the Hub), and `trace(..., remote=True)` runs on NDIF. The deprecated `LanguageModel` / `VisionLanguageModel` aliases share this class's remote key (`language.py`, `LanguageModel._remoteable_class`), so a model deployed as a `TransformersModel` is reachable when wrapped as either. See [docs/remote/](../remote/).

## Gotchas

- **`generate` vs `pipe`.** `generate` returns token ids and is greedy by default; `pipe` returns decoded records and folds in the checkpoint's sampling `task_specific_params`.
- **`save()` outside a trace raises.** `.save()` / `nnsight.save(...)` raises if there's no active trace.
- **`scan` needs `dispatch=False` to be cheap** but works either way; it never loads weights.
- **Opaque inputs can't be batched.** A multimodal encoding (with `pixel_values`) or a raw float tensor must be a lone invoke — batching several raises `NotImplementedError`.
- **Chunked inputs can't be batched either.** A task that splits one input into several forward rows takes the whole batch; see [Chunked tasks](#chunked-tasks).
- **`tracer.result` is preferred over `model.generator.output`** for finished ids; reading or writing the latter warns with `nnsight.NNsightDeprecationWarning`.

## Related

- [docs/models/index.md](index.md) — decision tree
- [docs/models/nnsight-base.md](nnsight-base.md) — the base `NNsight` wrapper
- [docs/models/language-model.md](language-model.md) / [vision-language-model.md](vision-language-model.md) — deprecated aliases
- [docs/models/vllm.md](vllm.md) — high-throughput serving path
- `src/nnsight/modeling/transformers.py` — source
- `tests/test_language.py`, `tests/test_encoder.py`, `tests/test_vision.py`, `tests/test_vlm.py` — runnable examples
