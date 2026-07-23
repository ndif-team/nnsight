---
title: API Quick Reference
one_liner: Tables of every public model class, tracer method, envoy property, and top-level function.
tags: [reference, api]
---

# API Quick Reference

Signatures use `model` for an `NNsight` / `TransformersModel` / `DiffusionModel` / `VLLM` instance, and `tracer` for the value bound by `as tracer:` on the trace context. Prefer `TransformersModel` over the deprecated `LanguageModel` / `VisionLanguageModel` names.

## Model classes

All are subclasses of `Envoy` (the tree node type). `NNsight` is the thin, named base; the rest add loading/tokenization.

| Class | Import | One-liner |
|-------|--------|-----------|
| `NNsight` | `from nnsight import NNsight` | Wrap any `torch.nn.Module`. Recursively mirrors the module tree as envoys. |
| `TransformersModel` | `from nnsight import TransformersModel` | **Primary** HuggingFace class, backed by a `transformers.pipeline`. Any task via `task=...` (inferred if unset). |
| `DiffusionModel` | `from nnsight import DiffusionModel` | Wraps any `diffusers.DiffusionPipeline`; components (`unet`, `vae`, ...) are envoys. Was `DiffusionModel`. |
| `VLLM` | `from nnsight.modeling.vllm import VLLM` | vLLM-backed model; interventions run inside the engine's worker. `mode="sync"` (default) or `mode="async"`. |
| `LanguageModel` | `from nnsight import LanguageModel` | **Deprecated** — warns on construction. Use `TransformersModel(repo_id, task="text-generation")`. |
| `VisionLanguageModel` | `from nnsight import VisionLanguageModel` | **Deprecated** — warns on construction. Use `TransformersModel(repo_id, task="image-text-to-text")`. |

Common constructor kwargs (via `HuggingFaceModel` / `Meta`): `dispatch=False` (load real weights now vs lazily on first run), `device_map=`, `revision=`, `rename=`. `TransformersModel` also takes `task=`, `tokenizer=`, `processor=`, `image_processor=`, `feature_extractor=`, `peft=<adapter repo_id>`.

## Run methods

Each returns a tracer usable as `with model.<method>(...) as tracer:`. Called directly (no `with`), each just runs and returns the result. Give a run method input directly and the whole block is one implicit invoke; give it no input and define the batch with `tracer.invoke(...)` blocks.

| Method | On | Runs | Returns (`tracer.result`) |
|--------|-----|------|--------------------------|
| `model.trace(*inputs, **kw)` | all | One forward pass. | The forward's return value (e.g. a `CausalLMOutput`). |
| `model.generate(*inputs, max_new_tokens=N, **kw)` | `TransformersModel`, `DiffusionModel` | Generation through the **model** (greedy by default). | **Token ids** `[batch, seq]` (Transformers); pipeline output (Diffusers). |
| `model.pipe(*inputs, **kw)` | `TransformersModel` | The whole task **pipeline** (preprocess + forward + postprocess). | Its **records** — decoded text, labels, etc. (what old `generate` returned). |
| `model.scan(*inputs, **kw)` | all | One forward under fake tensors — shapes/dtypes only, no weights, no dispatch. | (Read shapes inside the block; fake tensors are invalid after it.) |
| `model.edit(*, inplace=False)` | all | Captures interventions as **defaults** replayed on every future trace. | `as (tracer, edited)` when `inplace=False`; `as tracer` when `inplace=True`. |
| `model.session(*, remote=False)` | all | A scope enclosing several traces that share values without `.save()`. | (Only `nnsight.save`d values survive the session.) |
| `with tensor.backward(...):` | any captured tensor | A backward pass; read `.grad` on tensors captured earlier in the forward. | — |

Notes:
- `model.trace(...)` accepts a `trace=False` kwarg to bypass tracing (one-shot forward; only edits apply).
- `VLLM` sampling params (`temperature`, `max_tokens`, `top_p`, ...) go to `trace`/`invoke`, not the constructor.

## Tracer methods and attributes

`tracer` is bound by `with model.trace() as tracer:` (or `.generate()` / `.pipe()` / `.scan()`).

| Item | Signature | One-liner |
|------|-----------|-----------|
| `tracer.invoke` | `tracer.invoke(*args, **kwargs)` | Add one batched input group; its body's interventions see only its rows. Empty `tracer.invoke()` sees the whole batch. |
| `tracer.result` | `tracer.result` | The traced call's return value. |
| `tracer.iter` | `tracer.iter[slice\|int\|list]` | Target occurrences of a location across a repeated run (e.g. generation steps). Loop: `for step in tracer.iter[:3]:`. |
| `tracer.all` | `tracer.all()` | Shorthand for `tracer.iter[:]` — every occurrence. |
| `tracer.cache` | `tracer.cache(modules=None, device=cpu, dtype=None, detach=True, include_output=True, include_inputs=False)` | Record many modules' activations at once; returns a `CacheView` that fills as the run proceeds. |
| `tracer.barrier` | `tracer.barrier(n: int) -> Barrier` | A meeting point for `n` of this trace's blocks; the last to call it releases them all. |
| `tracer.stop` | `tracer.stop()` | Halt the model's forward pass early (raises `EarlyStopException`). |

The `tracer.next()` method from old nnsight is **gone** — advance across occurrences with `tracer.iter` / `tracer.all()` instead.

## Envoy properties

Available on `model` and every wrapped submodule (`model.transformer.h[0].mlp`, ...). Read/written **inside a trace**; reading parks the worker until the model produces the value. Assigning replaces it in place.

| Item | Returns / accepts | One-liner |
|------|-------------------|-----------|
| `.output` | module's forward return | Read or overwrite the module's output. |
| `.input` | first positional arg (or first kwarg) | Read or overwrite the first input. |
| `.inputs` | `(args, kwargs)` | All inputs to the module's forward. |
| `.source` | `Source` | Operation-level access to the module's forward internals (see below). |
| `.device` / `.devices` | `torch.device` / `set` | Device(s) of the module's parameters. |

## Envoy / module methods

| Item | Signature | One-liner |
|------|-----------|-----------|
| `envoy.skip` | `envoy.skip(replacement)` | Bypass this module's forward, using `replacement` as its output. |
| `envoy(...)` | `envoy(*args, hook=False, **kwargs)` | Ad-hoc apply the module to a value (e.g. logit lens). `hook=True` fires the module's own hooks — for adapters/SAEs/LoRA attached to the tree. |
| `envoy.get` | `envoy.get("transformer.h.0.mlp")` | Fetch a descendant envoy by dotted path. |
| `envoy.modules` | `envoy.modules(include_fn=None, names=False)` | List all descendant envoys (optionally filtered / with paths). |
| `envoy.named_modules` | `envoy.named_modules(include_fn=None)` | `modules(names=True)` — `(path, envoy)` pairs. |
| `envoy.to` / `.cpu` / `.cuda` | `envoy.to(device)` | Move the underlying module; returns the envoy. |
| `envoy.clear_edits` | `envoy.clear_edits()` | Drop all edits accumulated by `edit(inplace=True)`. |
| `envoy[i]` / `for c in envoy` / `len(envoy)` | — | Index / iterate direct children (e.g. a `ModuleList`'s blocks). |
| `envoy.source.<op>_<n>` | e.g. `.source.relu_0` | A `SourceEnvoy` for the n-th call of `<op>` in the forward; same `.input`/`.output`/`.skip`/`.source` interface. `print(envoy.source)` lists them. |

Deprecated aliases (warn; use the `tracer.*` forms): `model.iter`, `model.all()`.

## Model-specific handles

| Item | On | One-liner |
|------|-----|-----------|
| `model.tokenizer` / `.processor` / `.image_processor` / `.feature_extractor` | `TransformersModel` | The preprocessors the task loaded (any may be `None`). |
| `model.pipeline` | `TransformersModel`, `DiffusionModel` | The underlying `transformers.pipeline` / `DiffusionPipeline`. |
| `model.generator.output` | `TransformersModel` | Generated ids passthrough — **deprecated**; use `tracer.result`. |
| `model.generator.streamer.output` | `TransformersModel` | Per-step generated tokens during decoding. |
| `model.logits` | `VLLM` | This request's pre-sampling logits for the step. |
| `model.samples` | `VLLM` | The token ids the sampler drew for the step. |

## Top-level functions

Imported from the top-level `nnsight` package.

| Item | Signature | One-liner |
|------|-----------|-----------|
| `nnsight.save` | `nnsight.save(obj) -> obj` | Mark `obj` to survive past the outermost trace. Same as `obj.save()`. **Raises if called outside a trace.** |
| `nnsight.register` | `nnsight.register(module \| "name")` | Ship a local module's source with remote requests (cloudpickle by value). |
| `nnsight.status` | `nnsight.status(raw=False)` | Query NDIF; `print()` shows deployed models and state. |
| `nnsight.ndif_status` | `nnsight.ndif_status(raw=False)` | **Deprecated** alias for `status()`. |
| `nnsight.is_model_running` | `nnsight.is_model_running(repo_id, revision="main") -> bool` | Whether a model is currently RUNNING on NDIF. |
| `nnsight.compare` | `nnsight.compare() -> EnvComparison` | Diff local vs NDIF Python/package versions; `print()` for the table. |
| `nnsight.CONFIG` | `Config` | The config singleton (see [config.md](./config.md)). |
| `nnsight.Object` | type | Tensor-like static type for values read inside a trace (typing hints). |

(`get_local_env` / `get_remote_env` live on `nnsight.ndif`, not the top level. `nnsight.session`, `nnsight.apply`, `nnsight.cond`, `nnsight.log`, `nnsight.local`, and the `nnsight.list/dict/int/...` wrappers are **removed** — use plain Python and `model.session()`.)

## Remote execution

Pass `remote=` to `model.trace(...)` / `model.generate(...)` / `model.session(...)`.

| `remote=` value | Backend | Behavior |
|-----------------|---------|----------|
| `True` | `RemoteBackend` | Ship to the configured NDIF host (`CONFIG.API.HOST`). `blocking=True` (default) holds one websocket until COMPLETED; `blocking=False` submits and returns a job to `poll()`. |
| `"local"` | `LocalSimulationBackend` | Serialize/deserialize and run in-process — an offline dry run of the remote path. |
| `"<host url>"` | `RemoteBackend` | Like `True`, overriding the host for this call. |

`RemoteBackend` extra kwargs: `blocking`, `job_id`, `verbose`. `AsyncRemoteBackend` (built by `VLLM` async traces and available directly) supports `await backend` → the saves dict, and `async for update in backend` → status updates then the saves dict last.

Model identity for remote: `model.to_model_key()` → `"import.path.Class:model_key"`; `Class.from_model_key(key)` reconstructs it. Deprecated aliases (`LanguageModel`, `VisionLanguageModel`) share `TransformersModel`'s key.
