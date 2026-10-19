---
title: TransformersModel
one_liner: Primary wrapper for any HuggingFace transformers task; trace/generate/pipe/scan with tokenization and batching handled by the task's pipeline.
tags: [models, transformers, primary]
related: [docs/models/index.md, docs/models/nnsight-base.md, docs/models/language-model.md, docs/models/vision-language-model.md, docs/models/vllm.md]
sources: [src/nnsight/modeling/transformers.py, src/nnsight/modeling/processing.py, src/nnsight/modeling/huggingface.py, src/nnsight/modeling/mixins/meta.py, tests/test_language.py, tests/test_encoder.py, tests/test_vision.py, tests/test_vlm.py, tests/test_chunked_tasks.py, tests/test_modeling.py]
---

# TransformersModel

## What this is for

`nnsight.TransformersModel` is the **primary** wrapper for any HuggingFace `transformers` model. It is backed by a `transformers.pipeline`, so it works for any task the pipeline factory knows — text generation, fill-mask, text/image classification, token classification, ASR, image-text-to-text (VLM) — and leans on the pipeline to tokenize, featurize, template chat, and pad batches. You pick the task (or let it be inferred from the checkpoint) and get the full NNsight tracing API. Two tasks cannot be traced; see [Chunked tasks](#chunked-tasks).

`LanguageModel` and `VisionLanguageModel` are deprecated aliases over this class (see [language-model.md](language-model.md) / [vision-language-model.md](vision-language-model.md)).

## Loading

```python
from nnsight import TransformersModel

# task is inferred from the checkpoint when omitted
model = TransformersModel("openai-community/gpt2", dispatch=True)

# or pin it explicitly
model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)

# choose the device with device=, not device_map=
model = TransformersModel("openai-community/gpt2", device="cpu", dispatch=True)
```

### Choosing a device

Pass `device=` (`"cpu"`, `"cuda:1"`, `0`) to put the whole model on one device; `device_map=` is for spreading it across several. The model loads through `transformers.pipeline(...)`, which moves a single-device model to its own `device` (the first accelerator by default), silently overriding a single-device `device_map`. This holds for `dispatch=True` and lazy dispatch alike; check with `next(model.parameters()).device`.

| Argument (one GPU present) | Model ends up on |
|---|---|
| none | `cuda:0` (CPU only when no accelerator exists) |
| `device="cpu"` | `cpu` |
| `device="cuda"` / `device=0` | `cuda:0` |
| `device_map="cpu"` / `device_map={"": "cpu"}` | `cuda:0` |
| `device_map="cuda"` / `device_map="auto"` | `cuda:0` |
| `device_map=` dict naming two or more devices | as mapped (the pipeline leaves a multi-device model where it is) |

`DiffusionModel` loads through diffusers instead and works the other way round: there `device_map="cuda"` places the pipeline and `device=` is ignored (see [diffusion-model.md](diffusion-model.md)).

Inferring the task asks the Hub for the checkpoint's metadata, and a fully cached
checkpoint does not change that. Under `HF_HUB_OFFLINE=1` the first form raises
`RuntimeError: You cannot infer task automatically within 'pipeline' when using
offline mode`, so pass `task=` on an air-gapped machine or a cluster node with no
outbound network.

**Seq2seq checkpoints (T5, BART).** transformers 5 has no seq2seq pipeline task, and `text-generation` from a repo id silently builds the decoder-only class (`BartForCausalLM`, no encoder). Load the model yourself and pass the module:

```python
from transformers import AutoTokenizer, BartForConditionalGeneration

repo = "facebook/bart-base"
model = TransformersModel(
    BartForConditionalGeneration.from_pretrained(repo),
    tokenizer=AutoTokenizer.from_pretrained(repo),
)
```

`pipe` on such a model echoes the prompt rather than the completion, so measure interventions through `generate` / `tracer.result`.

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
                                # (dtype, device, device_map, attn_implementation, ...)
)
```

| Parameter | Description |
|-----------|-------------|
| `repo_id` | A HuggingFace repo id string, or an already-instantiated `torch.nn.Module`. |
| `task` | The pipeline task (`"text-generation"`, `"fill-mask"`, `"text-classification"`, `"image-classification"`, `"image-text-to-text"`, ...). If `None`, inferred from the checkpoint — which asks the Hub, so pass it explicitly when you are offline (see [Loading](#loading)). |
| `tokenizer` / `processor` / `image_processor` / `feature_extractor` | Pass one to adopt it instead of letting the pipeline load it. Which of them a task uses varies; the unused ones stay `None` (`transformers.py`, `_preprocessor_sources`). |
| `peft` | Repo id of a PEFT adapter grafted onto the base model at load. `model.load_adapter(repo_id)` does the same after construction, swaps to another adapter, or removes the current one with `None`. The adapter is attached in place, so every path of the base model still names the same module (`model.transformer.h[0].attn.c_attn` is now the adapter's layer, holding the original as `.base_layer`), and a trace written for the base model runs unchanged with an adapter. Prompt-learning adapters (prefix/prompt tuning), which wrap the forward instead, are refused. See `tests/test_language.py` for verified PEFT usage. |
| `dispatch` | `True` loads real weights during `__init__`; `False` (default) builds the architecture on the `meta` device and loads weights lazily on the first `trace`/`generate`/`pipe`. |
| `dtype` | Forwarded. A torch dtype, or a quantization name (`"nf4"`, `"int8"`, ...) — see [quantization.md](quantization.md). The transformers 4 spelling `torch_dtype` is still accepted. |
| `device` | The single device to load onto (`"cpu"`, `"cuda:1"`, `0`). Defaults to the first accelerator. See [Choosing a device](#choosing-a-device). |
| `device_map`, `trust_remote_code`, `attn_implementation`, ... | Forwarded to the pipeline / `from_pretrained`; `trust_remote_code` also joins the remote model key (see [Remote](#remote)). `device_map` spreads a model across devices; a single-device map such as `"cpu"` is overridden by the pipeline's `device`. `attn_implementation` (`"eager"` when attention weights are wanted) is read back as `model.attn_implementation` and travels with every remote request so the server runs the same one (see [Remote](#remote)). |
| `config` | A config object the model is built from in place of the checkpoint's own, on the meta build and the real load alike. Keyword overrides are not merged into it, as with `from_pretrained`. |
| `rename` | Module-path aliases (see [rename-modules.md](../usage/rename-modules.md)). |

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

`generate` decodes with the checkpoint's own `generation_config`, which on many instruct checkpoints samples — pass `do_sample=False` for determinism (see [generate](../usage/generate.md#decoding-follows-the-checkpoints-generation_config)). Read the ids off `tracer.result`. `**kwargs` are forwarded to the model's `generate`, e.g. `max_new_tokens`, `num_return_sequences`, `generation_config=`.

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

## Input forms

Every invoke is normalized to per-row model inputs and padded into one forward, so mixed formats and unequal lengths combine freely (`processing.py`, `preprocess_invoke`; verified in `tests/test_language.py`):

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

An encoding, keywords, and ids with keywords are the same input written three
ways. Its per-row fields are split by row, padded and batched with whatever the
other invokes pass:

| Field | Padded with |
|---|---|
| `input_ids`, `decoder_input_ids` | the pad token |
| `inputs_embeds`, `decoder_inputs_embeds` | zeros |
| `attention_mask`, `decoder_attention_mask`, `global_attention_mask`, `token_type_ids`, `position_ids`, `decoder_position_ids` | 0 |
| `special_tokens_mask` | 1 |
| `labels` | -100, which the loss ignores |
| `start_positions`, `end_positions`, `next_sentence_label` | one per row, no padding |

- **Padding side.** Fields the length of the input are padded on the side the
  batch is (the tokenizer's, or the feature extractor's when it names one). The
  decoder's fields, and `labels` on an encoder-decoder, have a length of their
  own and are padded on the right.
- **Unbatched ids take unbatched fields.** `trace([5, 6, 7], labels=[5, 6, 7])`
  is one row with a label per token, and `labels=1` beside it is one class label.
- **One row of `token_type_ids` or `position_ids`** stands for every row of a
  batch, as the models broadcast them. Any other field needs a row per input row.
- **A field one invoke leaves out** gets a default: segment 0, positions counted
  from the row's first token, and an ignored label. A field with no default (a
  decoder input, a float label) has to be in every invoke or none. Segment 0 is
  what an encoder assumes; GPT-2 adds an embedding for any `token_type_ids` it is
  given, so pass them to every invoke of a GPT-2 batch or to none.
- **`inputs_embeds`** stands in for `input_ids`, in `trace` and `generate`.
- **Positions** are derived from the mask row by row, on a left-padding
  tokenizer: a row that is padding then tokens is counted from its first token,
  and any other row is left as the model counts it. Positions you pass are kept,
  and a mask wider than the input (a key-value cache continuation) gets none.

Other keywords are forward arguments, passed on as they are.

**Opaque inputs don't batch.** A *tensor* outside the table (`pixel_values`,
`input_features`, `cache_position`, ...) or a raw float feature tensor is not
known to be per-row, so the input is handed to the model whole and must be a
lone invoke; a second invoke raises `NotImplementedError: Can't batch these
inputs; pass text or token ids.` (`processing.py`, `preprocess_invoke`).

Chat messages are detected and templated automatically (as `Pipeline.__call__` would):

```python
messages = [{"role": "user", "content": "Who wrote 'Beloved'?"}]
with model.trace(messages):
    ...
```

### Batching across invokes

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        a = model.output.logits[:, -1].save()
    with tracer.invoke(["Madison Square Garden is in", "The Colosseum is in"]):
        b = model.output.logits[:, -1].save()      # batch of 2
```

Causal decoders left-pad and get mask-derived `position_ids`; encoders (BERT, DistilBERT) keep right padding and need no correction (`processing.py`, `supply_position_ids`; `tests/test_encoder.py`). An empty `tracer.invoke()` sees the whole padded batch.

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

**A chunked invoke is the whole batch.** The row count belongs to the task and is only known after preprocessing, while the batcher counts one row per invoke before that — so a second invoke would name rows belonging to the first. It is refused with `NotImplementedError: task='zero-shot-classification' splits this invoke into 3 forward rows, and a batched trace gives an invoke the rows its input has ...`.

**Two tasks are refused under `trace`**, each with a message naming the escape hatches — `model.pipe(...)` for the whole task, or a forward on an encoding you build with `model.image_processor`:

- `mask-generation` — its preprocessing runs the model itself (to embed the image), so the encoder would run outside the trace.
- `keypoint-matching` — its unit input is a *pair* of images, which the list convention (one prompt per element) would split. Use `model.pipe([image_a, image_b])` or `model.image_processor(images=[image_a, image_b], return_tensors='pt')`.

A task whose input is a dict — `{"image": ..., "question": ..., "word_boxes": [...]}` for `document-question-answering`, `{"image": ..., "candidate_labels": [...]}` for `zero-shot-object-detection`, `{"table": ..., "query": ...}` for `table-question-answering` — goes through the task's own preprocessing, including the `_args_parser` step `Pipeline.__call__` would run (which is where `table-question-answering` builds its `pd.DataFrame`). A mapping carrying tensors is still read as a model encoding (`processing.py`, `is_task_input`). A dual-encoder zero-shot task (`zero-shot-image-classification`, `zero-shot-audio-classification`) nests the candidate labels' text encoding inside its preprocess row; the trace merges those tensors into its one forward, so `logits_per_image` / `logits_per_audio` reads `(1, n_labels)` — one text row per candidate label against the single image/audio row.

**An encoder-decoder ASR trace needs `decoder_input_ids=`.** `trace` runs one forward, and a seq2seq model with no decoder ids builds decoder embeddings for itself, then rejects the pair it just made: `ValueError: You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time`, which names nothing the caller wrote. Pass one row of `decoder_start_token_id` per chunk:

```python
ids = torch.tensor([[asr.config.decoder_start_token_id]] * 3)
with asr.trace(audio, chunk_length_s=30, decoder_input_ids=ids):
    encoded = asr.model.encoder.layers[0].output.save()   # (3, 1500, 384)
```

`generate` and `pipe` run the decode loop themselves and need none of this.

## Vision-language models

An `image-text-to-text` model takes a prompt plus `images=` (or `text=` and `images=`), or chat messages with the image inline (`{"type": "image", "image": img}`); the processor runs for you in `trace`, `scan` and `generate`, and `generate` returns token ids. Generation arguments (`max_new_tokens`, `do_sample`, `temperature`, ...) reach the model's `generate` as plain keywords; the pipeline's spelling, `generate_kwargs={...}`, is unpacked into them. `max_length` is the exception: the pipeline sends it to the processor as a truncation length, so bound generation with `max_new_tokens`.

### Finding the image span

The processor expands the template's single `<image>` placeholder into one token per image patch, and those positions are where an intervention on the image goes. Build the encoding yourself and match the config's image token id:

```python
enc = model.processor(images=image, text=prompt, return_tensors="pt")
span = (enc["input_ids"] == model.config.image_token_id).nonzero()
with model.trace(enc):
    logits = model.output.logits.save()
```

**Only processor-built ids carry that span.** The same text through `model.tokenizer` alone leaves the placeholder as a single token, and pairing those ids with processor-built `pixel_values` raises `ValueError: Image features and image tokens do not match, tokens: 1, features: ...`. Passing the processor's encoding whole is equivalent to passing the prompt and `images=`. Either way the input is opaque and must be a lone invoke (see [Input forms](#input-forms)).

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

Per-step reads and edits during generation use `tracer.iter`; see [iteration.md](../usage/iteration.md).

## Interventions, renaming, caching

Reading, editing, gradients, `skip`, `tracer.cache` and `rename=` work as on any wrapper; see [access-and-modify.md](../usage/access-and-modify.md), [backward-and-grad.md](../usage/backward-and-grad.md), [skip.md](../usage/skip.md), [cache.md](../usage/cache.md) and [rename-modules.md](../usage/rename-modules.md).

## Dispatch behavior

- `dispatch=False` (default): only configs download; the architecture is built on the `meta` device and the Envoy tree is fully usable for writing intervention code. The meta config takes the same load kwargs as the real load (`attn_implementation`, `experts_implementation`, `dtype`, config overrides), so `model.config` is the same before and after dispatch; placement kwargs (`device_map`, `max_memory`, ...) wait for dispatch.
- `dispatch=True`: real weights load during `__init__`.
- First `trace`/`generate`/`pipe` auto-dispatches if needed (`mixins/meta.py`, `Meta.interleave`). `scan` does **not** dispatch.
- Call `model.dispatch()` to force loading.

## Remote

`TransformersModel` is remoteable (`trace(..., remote=True)`; see [docs/remote/](../remote/index.md)). `model.to_model_key()` is the repo id + revision, canonicalized via the Hub, plus the resolved pipeline task and `trust_remote_code` when set (the fields are defined in [backends.md](../developing/backends.md#model-identity-to_model_key)), so each task of a checkpoint is its own remote model and the server rebuilds the exact pipeline you traced. The deprecated aliases use this class's key with their pinned task, so they reach a deployment only when its task is `text-generation` / `image-text-to-text` respectively.

A request also carries what is layered on the base model, and the server applies it before running: the PEFT adapter (`peft=`), when there is one, and the attention implementation. The implementation is always sent, as the model resolved it locally (`model.attn_implementation`), because the server's model is shared with other callers: the previous request may have switched it to `"eager"`, and a request that said nothing would otherwise run with that. So a model loaded with `attn_implementation="eager"` gets attention weights out of `attn.output[1]` remotely as it does locally, and one loaded with the default gets the default. A model that cannot switch dynamically (one not written against transformers' attention interface) refuses the request rather than silently keeping its implementation.

## Related

- [docs/models/index.md](index.md) — decision tree
- [docs/models/nnsight-base.md](nnsight-base.md) — the base `NNsight` wrapper
- [docs/models/language-model.md](language-model.md) / [vision-language-model.md](vision-language-model.md) — deprecated aliases
- [docs/models/vllm.md](vllm.md) — high-throughput serving path
- `src/nnsight/modeling/transformers.py` — source
- `tests/test_language.py`, `tests/test_encoder.py`, `tests/test_vision.py`, `tests/test_vlm.py` — runnable examples
