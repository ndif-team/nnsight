---
title: API Quick Reference
one_liner: Tables of every public context manager, tracer method, envoy property, top-level function, and model class.
tags: [reference, api]
---

# API Quick Reference

All entries link to deeper docs where they exist. Signatures use `model` for an `NNsight` / `LanguageModel` / `VLLM` / `DiffusionModel` / `VisionLanguageModel` instance, and `tracer` for the value bound by `as tracer:` on the context manager.

## Context managers

Entered with `with ...:` to set up a tracing session.

| Item | Signature | One-liner | Doc |
|------|-----------|-----------|-----|
| `model.trace` | `model.trace(*args, **kwargs)` | Single forward-pass tracing context. With a positional/input kwarg, an implicit invoker is created. | [../usage/trace.md](../usage/trace.md) |
| `model.generate` | `model.generate(*args, max_new_tokens=N, **kwargs)` | Multi-token generation context. Use with `tracer.iter[...]` for per-step intervention. | [../usage/generate.md](../usage/generate.md) |
| `model.scan` | `model.scan(*args, **kwargs)` | Like `.trace()` but runs with fake tensors for shape inference / validation. Does not dispatch the model. | [../usage/scan.md](../usage/scan.md) |
| `model.session` | `model.session(remote=False)` | Group multiple traces; with `remote=True`, bundle them into a single NDIF request. | [../usage/session.md](../usage/session.md) |
| `model.session(remote=True)` | `model.session(remote=True)` | Single-request remote session: each inner trace runs on NDIF, values flow between them without `.save()`. | [../remote/remote-session.md](../remote/remote-session.md) |
| `model.edit` | `model.edit(*, inplace=False)` | Capture interventions into a persistent edited model copy. | [../usage/edit.md](../usage/edit.md) |
| `model.edit(inplace=True)` | `model.edit(inplace=True)` | Apply persistent interventions in place; all subsequent traces include them until `model.clear_edits()`. | [../usage/edit.md](../usage/edit.md) |
| `tensor.backward` | `with tensor.backward(retain_graph=False, ...):` | Open a backward-tracing session to access `tensor.grad` on intermediate tensors. | [../usage/backward-and-grad.md](../usage/backward-and-grad.md) |
| `tracer.invoke` | `tracer.invoke(*args, **kwargs)` | Add an invocation to a trace; each invoke runs in its own worker thread. | [../usage/invoke-and-batching.md](../usage/invoke-and-batching.md) |

## Tracer methods

`tracer` is the value bound by `with model.trace() as tracer:` (or `.generate()` / `.session()`).

| Item | Signature | One-liner | Doc |
|------|-----------|-----------|-----|
| `tracer.invoke` | `tracer.invoke(*args, **kwargs)` | Define an invocation (worker thread) that runs intervention code on the given input. Empty `tracer.invoke()` operates on the full batch. | [../usage/invoke-and-batching.md](../usage/invoke-and-batching.md) |
| `tracer.barrier` | `tracer.barrier(n_participants: int) -> Barrier` | Create a synchronization barrier; calling `barrier()` in `n` invokes pauses until all reach it. | [../gotchas/cross-invoke.md](../gotchas/cross-invoke.md) |
| `tracer.cache` | `tracer.cache(modules=None, device=cpu, dtype=None, detach=True, include_output=True, include_inputs=False) -> CacheDict` | Register persistent post-intervention cache hooks; populated during execution. | [../usage/cache.md](../usage/cache.md) |
| `tracer.stop` | `tracer.stop()` | Raise an `EarlyStopException` to halt the model forward pass early. | [../usage/stop-and-early-exit.md](../usage/stop-and-early-exit.md) |
| `tracer.iter` | `tracer.iter[slice|int|list]` | Iteration cursor for multi-token generation. Use as `for step in tracer.iter[:]`. | [../usage/generate.md](../usage/generate.md) |
| `tracer.all` | `tracer.all()` | Shorthand for `tracer.iter[:]` — iterate every generation step. | [../usage/generate.md](../usage/generate.md) |
| `tracer.next` | `tracer.next(step: int = 1)` | Manually advance the iteration cursor by `step` (default 1). | [../usage/generate.md](../usage/generate.md) |
| `tracer.result` | `tracer.result` | The traced function's final return value (e.g., HuggingFace generation output). | [../usage/trace.md](../usage/trace.md) |

## Envoy properties

Available on every `Envoy` (i.e. on `model` and every wrapped submodule). Reading these inside a trace blocks the worker thread until the value is delivered.

| Item | Returns | One-liner | Doc |
|------|---------|-----------|-----|
| `.output` | Module's forward return value | `eproperty` that fires a one-shot output hook; reads block until value arrives. | [../concepts/envoy-and-eproperty.md](../concepts/envoy-and-eproperty.md) |
| `.input` | First positional arg (or first kwarg) | `eproperty` that fires a one-shot input hook. | [../concepts/envoy-and-eproperty.md](../concepts/envoy-and-eproperty.md) |
| `.inputs` | `(args_tuple, kwargs_dict)` | All inputs to the module. Shares the `"input"` key with `.input`. | [../concepts/envoy-and-eproperty.md](../concepts/envoy-and-eproperty.md) |
| `.source` | `SourceEnvoy` | Access intermediate operations inside the module's forward (rewrites the forward to install op hooks). | [../usage/source.md](../usage/source.md) |
| `.skip` | (method, see below) | Skip the module's forward and return a replacement value as its output. | [../usage/skip.md](../usage/skip.md) |
| `.next` | (method, see below) | Advance the iteration cursor for this module — used to access later generation steps. | [../usage/generate.md](../usage/generate.md) |

## Envoy / Module methods

| Item | Signature | One-liner | Doc |
|------|-----------|-----------|-----|
| `Envoy.skip` | `envoy.skip(replacement: Any)` | Replace this module's output with `replacement` and bypass its forward. | [../usage/skip.md](../usage/skip.md) |
| `Envoy.next` | `envoy.next(step: int = 1)` | Advance the iteration cursor by `step` so subsequent reads target a later generation step. | [../usage/generate.md](../usage/generate.md) |
| `Envoy.clear_edits` | `envoy.clear_edits()` | Drop all `_default_mediators` (edits) accumulated by `model.edit(inplace=True)`. | [../usage/edit.md](../usage/edit.md) |
| `Envoy.__call__` | `envoy(*args, hook: bool = False, **kwargs)` | Ad-hoc apply the module to a tensor. Default bypasses interleaving hooks; `hook=True` lets the call participate in interleaving (useful for SAE / LoRA modules). | [../usage/extending.md](../usage/extending.md) |
| `Envoy.to` / `.cpu` / `.cuda` | `envoy.to(device)` | Move the underlying module; returns the envoy. | [../concepts/envoy-and-eproperty.md](../concepts/envoy-and-eproperty.md) |
| `Envoy.modules` / `.named_modules` | `envoy.modules(include_fn=None)` | Iterate all descendant Envoys (optionally filtered). | [../concepts/envoy-and-eproperty.md](../concepts/envoy-and-eproperty.md) |
| `Envoy.get` | `envoy.get(path: str)` | Fetch a descendant Envoy by dotted path (e.g. `"transformer.h.0.mlp"`). | [../concepts/envoy-and-eproperty.md](../concepts/envoy-and-eproperty.md) |

## Top-level functions

Imported from the top-level `nnsight` package.

| Item | Signature | One-liner | Doc |
|------|-----------|-----------|-----|
| `nnsight.save` | `nnsight.save(obj) -> obj` | Mark `obj` to persist past the trace boundary. Preferred over `obj.save()`. | [../usage/save.md](../usage/save.md) |
| `nnsight.session` | `nnsight.session(*args, **kwargs) -> Tracer` | Construct a bare `Tracer` (used as a session container). | [../usage/session.md](../usage/session.md) |
| `nnsight.register` | `nnsight.register(module: ModuleType | str)` | Register a local module for `cloudpickle` serialization-by-value when running remotely on NDIF. | [../remote/register-local-modules.md](../remote/register-local-modules.md) |
| `nnsight.ndif_status` | `nnsight.ndif_status(raw=False)` | Deprecated alias for `nnsight.status()`. | [../remote/ndif-overview.md](../remote/ndif-overview.md) |
| `nnsight.is_model_running` | `nnsight.is_model_running(repo_id, revision="main") -> bool` | Check whether a specific model is currently `RUNNING` on NDIF. | [../remote/ndif-overview.md](../remote/ndif-overview.md) |
| `nnsight.compare` | `nnsight.compare()` | Print a table comparing local vs NDIF Python and package versions. | [../remote/ndif-overview.md](../remote/ndif-overview.md) |
| `nnsight.get_local_env` | `nnsight.get_local_env() -> dict` | Build the local Python-version + installed-packages dict used by `compare()`. | [../remote/ndif-overview.md](../remote/ndif-overview.md) |
| `nnsight.get_remote_env` | `nnsight.get_remote_env(force_refresh=False) -> dict` | Fetch (and cache) NDIF's Python + package environment from `CONFIG.API.HOST/env`. | [../remote/ndif-overview.md](../remote/ndif-overview.md) |

Note: `nnsight.status()` is also available (the non-deprecated form of `ndif_status`).

## Model classes

All are subclasses of `NNsight` (which itself is a subclass of `Envoy`).

| Class | Import | One-liner | Doc |
|-------|--------|-----------|-----|
| `NNsight` | `from nnsight import NNsight` | Root envoy for any `torch.nn.Module`. Recursively wraps every child module. | [../models/nnsight-base.md](../models/nnsight-base.md) |
| `LanguageModel` | `from nnsight import LanguageModel` | HuggingFace-Transformers wrapper with `AutoModelForCausalLM` + tokenizer; supports `.trace()` and `.generate()`. | [../models/language-model.md](../models/language-model.md) |
| `VisionLanguageModel` | `from nnsight import VisionLanguageModel` | `LanguageModel` subclass that adds an `AutoProcessor` for VLMs (LLaVA, Qwen2-VL, etc.); accepts `images=` kwarg. | [../models/vision-language-model.md](../models/vision-language-model.md) |
| `DiffusionModel` | `from nnsight import DiffusionModel` | Wrapper for any `diffusers.DiffusionPipeline`; `.trace()` is single-step, `.generate()` runs the full pipeline. | [../models/diffusion-model.md](../models/diffusion-model.md) |
| `VLLM` | `from nnsight.modeling.vllm import VLLM` | vLLM-backed wrapper supporting single GPU, multi-GPU TP, Ray, and `mode="async"` streaming. Exposes `.logits` and `.samples` eproperties. | [../models/vllm.md](../models/vllm.md) |

## Configuration

`CONFIG` is a singleton `ConfigModel` instance.

| Item | Signature | One-liner |
|------|-----------|-----------|
| `nnsight.CONFIG` | `ConfigModel` | Singleton config object loaded from `src/nnsight/config.yaml`. |
| `CONFIG.set_default_api_key` | `CONFIG.set_default_api_key(apikey: str)` | Persist an NDIF API key into `config.yaml`. |
| `CONFIG.set_default_app_debug` | `CONFIG.set_default_app_debug(debug: bool)` | Persist `APP.DEBUG` into `config.yaml`. |
| `CONFIG.save` | `CONFIG.save()` | Write current config to `config.yaml`. |

See [config.md](./config.md) for every individual setting.
