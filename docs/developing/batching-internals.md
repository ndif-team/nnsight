---
title: Batching Internals
one_liner: How NNsight combines multiple invokes into one batch and slices activations per invoke.
tags: [internals, dev]
related: [docs/developing/architecture-overview.md, docs/developing/interleaver-internals.md, docs/developing/vllm-integration.md]
sources: [src/nnsight/intervention/batching.py:1, src/nnsight/modeling/language.py:241, src/nnsight/modeling/diffusion.py:350, src/nnsight/modeling/vllm/batching.py:1]
---

# Batching Internals

## What this covers

When a user defines multiple `tracer.invoke(...)` blocks inside a single trace, NNsight runs them in one forward pass. The `Batcher` accumulates inputs across invokes, and during interleaving it slices the per-invoke chunk out of each activation so the user's intervention code sees only the rows that belong to its invoke. This document walks through the `Batchable` interface model classes implement, the `Batcher` base class that drives narrow/swap, and the `DiffusionBatcher` / `VLLMBatcher` overrides that handle non-standard tensor layouts.

## Architecture / How it works

### Two halves: `Batchable` (model) and `Batcher` (per-trace)

Batching is split into a model-side mixin and a per-trace state object:

- `Batchable` (`src/nnsight/intervention/batching.py:35`) — abstract mixin on the model class. Defines `_prepare_input` and `_batch`, plus `_batcher_class()` returning the `Batcher` subclass to instantiate.
- `Batcher` (`src/nnsight/intervention/batching.py:114`) — instantiated once per trace (constructed in `InterleavingTracer.__init__` at `src/nnsight/intervention/tracing/tracer.py:300`). Accumulates inputs from each invoke, and during interleaving narrows / swaps activations on the way to and from intervention code.

`Envoy` inherits from `Batchable` but does not override `_prepare_input` / `_batch`, so the base `NNsight` class only supports a single input invoke. Calling a second input invoke on a base model raises `NotImplementedError` from `Batchable._batch` (`src/nnsight/intervention/batching.py:104`).

### The `needs_batching` flag

`Batcher.needs_batching` (`src/nnsight/intervention/batching.py:135`) is set to `True` only once a second input invoke has been merged. With one invoke, `narrow` and `swap` are no-ops — there is nothing to slice. The flag is also forced when the vLLM model runner registers more than one mediator (`src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:381`), since vLLM's flat token tensor always needs slicing across mediators.

### `_prepare_input` and `_batch` on the model

- `_prepare_input(*inputs, **kwargs) -> (args, kwargs, batch_size)` (`src/nnsight/intervention/batching.py:53`) — called once per invoke. Returns normalized args / kwargs ready to pass to the model, plus a `batch_size` integer. A `batch_size` of `0` marks an empty invoke. The base implementation returns `batch_size=1` if any input is given.
- `_batch(batched_input, *args, **kwargs) -> (combined_args, combined_kwargs)` (`src/nnsight/intervention/batching.py:78`) — called from the second input invoke onward. Receives the previously combined `(args, kwargs)` tuple plus the new invoke's prepared args/kwargs and returns the merged version.

### `batch_group = [start, length]` per mediator

`Batcher.batch()` (`src/nnsight/intervention/batching.py:146`) records each invoke as it arrives:

- First input invoke gets `batch_group = [0, batch_size]` and seeds `batched_args` / `batched_kwargs`.
- Second + input invokes call `_batch` to merge with what already exists, then update `last_batch_group = [previous_total, new_batch_size]` and set `needs_batching = True` (`src/nnsight/intervention/batching.py:191`).
- Empty invokes (no args/kwargs) get `batch_group = None`. They do not call `_prepare_input` or `_batch`, so they work even on classes without a `_batch` implementation.

The mediator is later associated with this `batch_group` (`src/nnsight/intervention/tracing/invoker.py` invoker setup), and the `Interleaver` uses it to drive `narrow` / `swap`.

### `narrow`, `swap`, `current_value`, `current_provider`

Inside an interleaving step, the interleaver puts the activation into `batcher.current_value` and a string identifying which provider produced it (e.g. `"output"`, `"input"`) into `batcher.current_provider`. The interleaver then iterates mediators, and for each one:

1. `narrow(mediator.batch_group)` (`src/nnsight/intervention/batching.py:198`) returns the slice of `current_value` for this invoke. For `batch_group=None` (empty invoke) or `needs_batching=False`, it returns `current_value` unchanged. Otherwise it calls `_narrow` per `torch.Tensor` via `util.apply`, narrowing `dim=0` from `start` to `start+length` — but only when the tensor's first dim equals `total_batch_size` (`src/nnsight/intervention/batching.py:274`), so non-batch tensors pass through.
2. The mediator runs the user's intervention code on the narrowed value.
3. `swap(mediator.batch_group, new_value)` (`src/nnsight/intervention/batching.py:226`) splices the (possibly modified) value back. If the tensor needs concat (leaf with `requires_grad`, or has `_base`, i.e. a view), it builds a fresh tensor with `torch.cat([pre, swap_value, post])`. Otherwise it does in-place slice assignment.

### `total_batch_size`

`Batcher.total_batch_size` (`src/nnsight/intervention/batching.py:140`) is `sum(last_batch_group)` — i.e. `start + length` of the most recent input invoke, which is the combined batch dimension. Both `_narrow` and `_swap` use this to detect "is dim 0 actually the batch dimension on this tensor?" so that activations like position embeddings (where dim 0 is sequence length) don't get incorrectly sliced.

### How `LanguageModel` implements batching

`LanguageModel._prepare_input` (`src/nnsight/modeling/language.py:241`):
- Splits `kwargs` between tokenizer kwargs (a hardcoded set at `src/nnsight/modeling/language.py:219`) and model kwargs.
- Accepts string, list of strings, list of ints, tensor, dict, or `BatchEncoding`.
- Tokenizes via `self._tokenize(...)`, returning a `BatchEncoding` with `input_ids` and `attention_mask`.
- Returns `((), {**inputs, "labels": labels, ...}, len(inputs["input_ids"]))`.

`LanguageModel._batch` (`src/nnsight/modeling/language.py:309`):
- Concatenates `input_ids` from the previous batch and the new invoke, re-padding via `tokenizer.pad(...)`.
- Builds a fresh combined attention mask, padding-side aware. If `padding_side == "left"`, masks are slotted into the right portion of each row; otherwise left.
- Concatenates `labels` if present.

### `DiffusionBatcher` overrides

`DiffusionBatcher` (`src/nnsight/intervention/batching.py:325`) handles three different effective batch sizes that show up inside diffusion pipelines:

1. `total_batch_size` — one row per prompt
2. `total_batch_size * num_images_per_prompt` — image-level batch (each prompt fans out)
3. `total_batch_size * num_images_per_prompt * 2` — classifier-free guidance, with concatenated unconditional + conditional halves

When recording a new invoke, `DiffusionBatcher.batch` (`src/nnsight/intervention/batching.py:356`) also computes a parallel `image_batch_group = (start * num_images, length * num_images)` and stores it in `image_batch_groups[batch_start]`. `_narrow` (`src/nnsight/intervention/batching.py:394`) inspects `acts.shape[0]` to detect which scenario applies and slices accordingly. The guided-diffusion case slices the unconditional half and the conditional half separately and concatenates them. `_swap` (`src/nnsight/intervention/batching.py:437`) mirrors this — for guided diffusion it splits `swap_value` in half via `chunk(2, dim=0)` and writes both halves back.

`DiffusionModel._batcher_class` (`src/nnsight/modeling/diffusion.py:282`) returns `DiffusionBatcher`. The `num_images_per_prompt` argument flows in through the trace kwargs.

### `VLLMBatcher` overrides

`VLLMBatcher` (`src/nnsight/modeling/vllm/batching.py:15`) handles two orthogonal concerns:

1. **Tensor-parallel gather/split.** `wrap(model)` (`src/nnsight/modeling/vllm/batching.py:33`) registers four PyTorch hooks on every `ColumnParallelLinear` and `RowParallelLinear` module. Pre-hooks (mediator_idx `-inf`, `src/nnsight/modeling/vllm/batching.py:112`) record `current_module` and whether the input/output is sharded; post-hooks (mediator_idx `+inf`) re-shard before vLLM resumes.
2. **`check_gathered()`** (`src/nnsight/modeling/vllm/batching.py:124`) is called on the way into `narrow` or `swap`. If the tracked module is a parallel layer with a sharded value at the current access point, it gathers via `tensor_model_parallel_all_gather` (column out / row in) or `tensor_model_parallel_all_reduce` (row out). After the gather, mediator code sees the full unsharded tensor.

vLLM's flat-token tensor format also means `narrow` operates on `[start_token, num_tokens]` during the forward pass and on `[start_prompt, num_prompts]` after. The transition is driven by `NNsightRequestHelper.unflatten()` (`src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:132`), which rewrites `mediator.batch_group` after `super().execute_model()` returns. See [vllm-integration.md](./vllm-integration.md) for details.

## Key files / classes

- `src/nnsight/intervention/batching.py:35` — `Batchable` mixin (`_prepare_input`, `_batch`, `_batcher_class`)
- `src/nnsight/intervention/batching.py:114` — `Batcher` base class (`batch`, `narrow`, `swap`, `_narrow`, `_swap`)
- `src/nnsight/intervention/batching.py:325` — `DiffusionBatcher`
- `src/nnsight/modeling/language.py:241` — `LanguageModel._prepare_input` (tokenization split)
- `src/nnsight/modeling/language.py:309` — `LanguageModel._batch` (padded concat + attention mask merge)
- `src/nnsight/modeling/diffusion.py:282` — `DiffusionModel._batcher_class`
- `src/nnsight/modeling/diffusion.py:350` — `DiffusionModel._prepare_input` (prompt list)
- `src/nnsight/modeling/diffusion.py:375` — `DiffusionModel._batch` (prompt list extension)
- `src/nnsight/modeling/vllm/vllm.py:220` — `VLLM._prepare_input` (one-prompt-per-invoke enforcement)
- `src/nnsight/modeling/vllm/vllm.py:330` — `VLLM._batch`
- `src/nnsight/modeling/vllm/batching.py:15` — `VLLMBatcher` (TP gather/scatter)
- `src/nnsight/intervention/tracing/tracer.py:300` — where the per-trace `Batcher` is instantiated

## Lifecycle

Per trace:

1. `InterleavingTracer.__init__` instantiates `model._batcher_class()(...)` — one `Batcher` per trace.
2. For each invoke, `Invoker.__exit__` calls `batcher.batch(model, *args, **kwargs)`. First input invoke seeds; subsequent input invokes merge via `_batch` and update `last_batch_group`; empty invokes return `batch_group=None`.
3. Once all invokes are recorded, `tracer.execute(fn)` runs the model with `batched_args` / `batched_kwargs`.
4. During interleaving, the interleaver fills `batcher.current_value` from each provider, then iterates mediators calling `narrow(batch_group)` / `swap(batch_group, ...)` so each invoke sees only its slice.
5. After interleaving the batcher is discarded with the tracer.

## Extension points

- **New runtime with custom batching**: subclass `Batchable`, override `_prepare_input` and `_batch`, return your own `Batcher` subclass from `_batcher_class()`. See [adding-a-new-runtime.md](./adding-a-new-runtime.md).
- **Same input format, different tensor layout**: subclass `Batcher` and override `_narrow` / `_swap` (and optionally `narrow` / `swap` if you need to gather/scatter before slicing, like `VLLMBatcher.check_gathered`). Override `total_batch_size` if your batch dimension isn't `sum(last_batch_group)`.
- **Single-input-only model**: leave `_batch` unimplemented. Users get a clear `NotImplementedError` (`src/nnsight/intervention/batching.py:104`) and can still use one input invoke + any number of empty invokes.

## Related

- [interleaver-internals.md](./interleaver-internals.md) — how `narrow` / `swap` are driven from the mediator loop
- [vllm-integration.md](./vllm-integration.md) — how `VLLMBatcher` plugs into vLLM's flat-token tensor format
- [adding-a-new-runtime.md](./adding-a-new-runtime.md) — recipe for implementing batching on a new model class
