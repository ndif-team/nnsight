---
title: Adding a New Runtime
one_liner: Recipe for integrating a new inference engine (vLLM-style) into NNsight.
tags: [internals, dev]
related: [docs/developing/vllm-integration.md, docs/developing/batching-internals.md, docs/developing/eproperty-deep-dive.md]
sources: [src/nnsight/modeling/base.py, src/nnsight/modeling/language.py, src/nnsight/modeling/vllm/README.md, src/nnsight/modeling/vllm/vllm.py:1]
---

# Adding a New Runtime

## What this covers

vLLM was integrated as a new "runtime" — an inference backend with its own process model, tensor format, and batching scheme. Wrapping a similar engine (TGI, TensorRT-LLM, JAX/Flax, a custom training loop, etc.) follows the same template: subclass `NNsight` (or `LanguageModel` if your runtime hosts HuggingFace-style transformers), implement input prep / batching, expose runtime-specific values via eproperties, and provide a way to dispatch model loading.

This page describes the parts you need to fill in. For a working reference, read the vLLM integration's README — it's the most thorough example of a non-PyTorch runtime under NNsight: [`src/nnsight/modeling/vllm/README.md`](../../src/nnsight/modeling/vllm/README.md). Don't reproduce that document — use it as the canonical example and link to it.

## Architecture / How it works

### When you actually need a new runtime

If you're wrapping a stock PyTorch model that does standard `forward(...) -> tensor`, plain `NNsight(model)` already works. You only need a new runtime class when one of these is true:

- The model lives in a different process (vLLM, Ray actors, a remote service).
- The activation tensor layout differs from `[batch, seq, hidden]` (vLLM's flat `[total_tokens, hidden]`, diffusion's `[batch * num_images_per_prompt * 2, ...]`).
- The forward / generate path is split across multiple stages that you want to hook independently (vLLM: forward, logits, sampling).
- Inputs need non-trivial batching (re-padding, attention mask merging, image batching).
- You expose engine-internal values that aren't `nn.Module` outputs (`model.logits`, `model.samples`).

### What you need to implement

| Piece | Required? | Reference |
|-------|-----------|-----------|
| Subclass of `NNsight` / `LanguageModel` / `RemoteableMixin` | Yes | `src/nnsight/modeling/vllm/vllm.py:43` |
| `_load_meta(repo_id, **kwargs)` | Yes (for lazy loading) | `src/nnsight/modeling/vllm/vllm.py:135` |
| `_load(repo_id, **kwargs)` | Yes (for real loading) | `src/nnsight/modeling/vllm/vllm.py:171` |
| `_prepare_input(*inputs, **kwargs)` | Required if multi-input invokes are supported | `src/nnsight/modeling/vllm/vllm.py:220` |
| `_batch(batched_input, *args, **kwargs)` | Required if `_prepare_input` returns `batch_size > 0` for multiple invokes | `src/nnsight/modeling/vllm/vllm.py:330` |
| Custom `Batcher` subclass | Required if your tensor layout differs from `[batch, ...]` | `src/nnsight/modeling/vllm/batching.py:15`, `src/nnsight/intervention/batching.py:325` |
| `_batcher_class()` classmethod | Required if you have a custom batcher | `src/nnsight/modeling/diffusion.py:282` |
| `__call__(...)` | Yes — called by the interleaving tracer to actually run a forward pass | `src/nnsight/modeling/vllm/vllm.py:409` |
| `__nnsight_generate__(...)` | Optional — separate generate path | `src/nnsight/modeling/language.py:140`, `src/nnsight/modeling/diffusion.py:474` |
| `interleave(fn, *args, **kwargs)` | Optional — override if your runtime needs custom dispatch logic | `src/nnsight/modeling/vllm/vllm.py:456` |
| eproperties for engine-internal values | As needed | `src/nnsight/modeling/vllm/vllm.py:102` (`logits`, `samples`) |
| Custom tracer subclass | Only if you need async or non-standard tracing semantics | `src/nnsight/modeling/vllm/async_tracer.py` |

### Step-by-step

#### 1. Pick a base class

- `NNsight` — base class for arbitrary `nn.Module` wrapping.
- `LanguageModel` — adds tokenization and HuggingFace integration.
- `HuggingFaceModel` / `TransformersModel` — useful as a halfway point if you load via `from_pretrained` but customize execution.
- `RemoteableMixin` — adds `_remoteable_*` hooks for NDIF support. The vLLM class extends this so it can be sent over remote.

vLLM extends `RemoteableMixin` directly (`src/nnsight/modeling/vllm/vllm.py:43`); diffusion extends `HuggingFaceModel` (`src/nnsight/modeling/diffusion.py:221`); LanguageModel extends `TransformersModel`.

#### 2. Implement `_load_meta` and `_load`

`_load_meta(repo_id, **kwargs)` should load a meta-tensor model so users can build the Envoy tree without GPU memory. `_load(repo_id, **kwargs)` should load real weights and connect the engine. Both return the wrapped `nn.Module` (or a wrapper — vLLM uses a meta vLLM config-only model in `_load_meta`; diffusion uses `init_empty_weights()` to create meta-tensor diffusion components).

#### 3. Implement `_prepare_input`

Normalize whatever the user can pass to `model.trace(...)` or `tracer.invoke(...)` into a `(args, kwargs, batch_size)` tuple consumed by your `__call__`. Return `batch_size = 0` for empty inputs.

vLLM enforces one prompt per invoke and accepts strings, token ID lists, or HuggingFace tokenizer dicts (`src/nnsight/modeling/vllm/vllm.py:220`). `LanguageModel` accepts batched prompts in a single invoke and handles tokenization (`src/nnsight/modeling/language.py:241`).

#### 4. Implement `_batch`

Called from the second input invoke onward. Combine `batched_input` (the running tuple from previous invokes) with the new invoke's prepared args/kwargs. Return the merged `(args, kwargs)`.

vLLM extends prompt / params / lora_request lists (`src/nnsight/modeling/vllm/vllm.py:330`). `LanguageModel` re-pads `input_ids` and merges attention masks (`src/nnsight/modeling/language.py:309`). Diffusion extends a flat prompt list (`src/nnsight/modeling/diffusion.py:375`).

If multi-invoke isn't supported, leave `_batch` unimplemented — `Batchable._batch` raises `NotImplementedError` with a helpful message (`src/nnsight/intervention/batching.py:104`).

#### 5. Custom `Batcher` (only if tensor layout differs)

If your runtime produces tensors with a non-`[batch, ...]` first dim — flat tokens, image batches, guidance-doubled batches — subclass `Batcher` and override `_narrow` / `_swap` (and possibly `narrow` / `swap` if you need pre-slice gather/scatter like `VLLMBatcher`). Return your subclass from `_batcher_class()` (a classmethod on the model).

References: `DiffusionBatcher` (`src/nnsight/intervention/batching.py:325`) for guidance/image batching, `VLLMBatcher` (`src/nnsight/modeling/vllm/batching.py:15`) for flat tokens + tensor parallelism.

See [batching-internals.md](./batching-internals.md) for the full Batcher contract.

#### 6. Implement `__call__`

This is what `model.interleave(fn, *args, **kwargs)` ends up calling. It runs the forward / generate path on the underlying engine and triggers value provision through your eproperties.

vLLM's `__call__` (`src/nnsight/modeling/vllm/vllm.py:409`) calls `_serialize_mediators` to embed mediators in `SamplingParams.extra_args`, runs `vllm_entrypoint.generate(...)`, and pushes saves back to the user's frame. Diffusion's `__call__` runs the pipeline with `num_inference_steps=1` for fast tracing (`src/nnsight/modeling/diffusion.py:457`). LanguageModel-style models inherit a generic forward call.

If you have a separate generate path, define `__nnsight_generate__` — see `LanguageModel.__nnsight_generate__` (`src/nnsight/modeling/language.py:140`) and `DiffusionModel.__nnsight_generate__` (`src/nnsight/modeling/diffusion.py:474`).

#### 7. eproperties for engine-internal values

Anything that is **not** an `nn.Module` output but should be accessible from intervention code goes through `eproperty`. vLLM exposes `logits` and `samples` this way (`src/nnsight/modeling/vllm/vllm.py:102,112`). On the runtime side, you call `type(self).<name>.provide(self, value)` to feed the value into the interleaver — see `NNsightGPUModelRunner.sample_tokens` (`src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:402`) for the pattern.

See [eproperty-deep-dive.md](./eproperty-deep-dive.md) for the full eproperty mechanics.

#### 8. Custom tracer (only for async or non-standard semantics)

If your runtime is async or needs to defer execution past the trace context, subclass `RemoteInterleavingTracer` (or `InterleavingTracer`) and override `execute()`. vLLM's async path uses `AsyncInterleavingTracer` (see `src/nnsight/modeling/vllm/async_tracer.py`) which prepares mediators without running generation, then `AsyncVLLMBackend` reads the prepared state.

Wire the tracer in by overriding `model.trace(...)` to inject `tracer_cls=YourTracer` (vLLM does this at `src/nnsight/modeling/vllm/vllm.py:445`).

#### 9. Optional: `RemoteableMixin` hooks

If your runtime should be runnable on NDIF, extend `RemoteableMixin` and override:

- `_remoteable_model_key()` — returns the string used by NDIF to identify the model.
- `_remoteable_persistent_objects()` — returns a `{name: object}` dict for persistent ID resolution. Tag the objects with `obj._persistent_id = name` in `__getstate__`. See `LanguageModel.__getstate__` (`src/nnsight/modeling/language.py:383`) and `VLLM.__getstate__` (`src/nnsight/modeling/vllm/vllm.py:474`).

#### 10. `dispatch()` / `dispatched`

Inherited from `MetaMixin`. By default, calling `model.dispatch()` calls your `_load(...)`. If your runtime needs preconditions before dispatch (vLLM destroys distributed environments first, see `src/nnsight/modeling/vllm/vllm.py:175`), override `_load` itself rather than `dispatch`.

If your runtime needs late dispatch (load on first trace), override `interleave(self, fn, *args, **kwargs)` to call `self.dispatch()` if `not self.dispatched`. vLLM does this at `src/nnsight/modeling/vllm/vllm.py:456`.

## Key files / classes (reference implementations)

- `src/nnsight/modeling/vllm/vllm.py:43` — `VLLM` (most complete reference for a non-PyTorch runtime)
- `src/nnsight/modeling/vllm/README.md` — extended narrative for vLLM
- `src/nnsight/modeling/diffusion.py:221` — `DiffusionModel` (custom batcher, custom call/generate split)
- `src/nnsight/modeling/language.py:21` — `LanguageModel` (tokenization + standard PyTorch forward)
- `src/nnsight/modeling/vlm.py` — `VisionLanguageModel` (multimodal extension of `LanguageModel`)
- `src/nnsight/intervention/batching.py:35` — `Batchable` mixin contract
- `src/nnsight/intervention/envoy.py` — `Envoy` and `eproperty` definitions

## Lifecycle of a runtime trace

1. User: `with model.trace(input)`.
2. Tracer captures source.
3. On `__exit__`, `Backend.__call__` compiles the function.
4. `tracer.execute(fn)` runs `_setup_interleaver`, which calls `Batcher.batch(model, *args, **kwargs)` for each invoke. `_prepare_input` and `_batch` run here.
5. `model.interleave(self.fn, *args, **kwargs)` is called. Your `__call__` runs the engine.
6. Inside the engine's forward path, mediator hooks fire. `Batcher.narrow`/`swap` slice activations per invoke. eproperties surface engine-internal values via `<eproperty>.provide(model, value)`.
7. Saved values are pushed back to the user's frame via `tracer.push`.

## Extension points (within your runtime)

- **Per-step hooks for multi-step engines.** Use `interleaver.default_all = num_steps` to set the default iteration count (`LanguageModel.__nnsight_generate__` does this at `src/nnsight/modeling/language.py:151`; `DiffusionModel._run_pipeline` does it at `src/nnsight/modeling/diffusion.py:431`).
- **Cross-step variable sharing.** vLLM grafts globals across mediators in `process_new_reqs` (`src/nnsight/modeling/vllm/model_runners/GPUModelRunner.py:96`). Diffusion handles this through standard `Globals.saves`.
- **Telemetry / profiling.** Hook your runtime's metrics into the eproperty mechanism so users can `.save()` them like any other value.

## Related

- [`src/nnsight/modeling/vllm/README.md`](../../src/nnsight/modeling/vllm/README.md) — canonical vLLM reference (don't reproduce, link)
- [vllm-integration.md](./vllm-integration.md) — vLLM internals beyond the user surface
- [batching-internals.md](./batching-internals.md) — `Batcher` / `Batchable` contract
- [eproperty-deep-dive.md](./eproperty-deep-dive.md) — eproperty mechanics for engine-internal values
- [serialization.md](./serialization.md) — needed if your runtime supports remote
