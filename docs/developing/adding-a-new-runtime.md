---
title: Adding a New Runtime
one_liner: How a new model type or inference engine plugs into NNsight via the modeling mixins.
tags: [internals, dev]
related: [docs/developing/vllm-integration.md, docs/developing/extending-envoy.md, docs/developing/batching-internals.md]
sources: [src/nnsight/modeling/base.py, src/nnsight/modeling/mixins/loadable.py, src/nnsight/modeling/mixins/meta.py, src/nnsight/modeling/mixins/remotable.py, src/nnsight/modeling/huggingface.py, src/nnsight/modeling/transformers.py, src/nnsight/modeling/vllm/vllm.py]
---

# Adding a New Runtime

## What this covers

A "runtime" is a model type with its own loading, batching, or execution model —
HuggingFace transformers, diffusers, vLLM, or a new engine you're integrating. They
all sit on the same short mixin chain over `Envoy`. This page is which mixin to
start from and which underscore-prefixed extension points to fill in. Extension
points are methods with working defaults (no ABCs, no `Mixin` suffix — see
`STYLE.md`); the docstring on each states the base default and points at the
reference override.

## The class chain

```
Envoy                      intervention/envoy.py   the tree; hooks; trace/interleave/__call__
 └─ Loadable               mixins/loadable.py      _load(...): construct from a spec, not a module
     └─ Meta               mixins/meta.py          meta-device build + dispatch(); scan()
         └─ Remotable      mixins/remotable.py     remote key/env; remote & local backends
             └─ HuggingFaceModel   huggingface.py  from_pretrained loading; model-key from repo id
                 ├─ TransformersModel  transformers.py   PRIMARY HF class (pipeline-backed)
                 │   └─ LanguageModel        (deprecated alias)
                 │       └─ VisionLanguageModel (deprecated alias)
                 └─ DiffusionModel     diffusers.py
             └─ VLLM               vllm/vllm.py     a non-PyTorch engine, straight off Remotable
```

`NNsight` (`base.py`) is a separate leaf: `class NNsight(Envoy)` with an empty body —
a thin, named `Envoy`. It wraps an **already-instantiated** `nn.Module` and has no
`_load`, no `dispatch`, no `scan` (those come from the mixins).

## Pick a base class

- **`NNsight(module)`** — you already have the `nn.Module`. Nothing to add.
- **`Loadable`** — you construct the model from a spec (`_load(*args)`), not a passed
  module.
- **`Meta`** — you want a meta-device tree built up front (so users build the Envoy
  tree without weights) and real weights loaded lazily on first run via `dispatch()`.
- **`Remotable`** — the model should be runnable on NDIF or via `remote="local"`.
- **`HuggingFaceModel`** — it loads from the HF Hub by repo id.
- **`VLLM`** shows the deepest case: a non-PyTorch engine off `Remotable` directly.

## Extension points

### Loading — `_load_meta` and `_load`

- `_load(*args, **kwargs) -> nn.Module` (`loadable.py:19`, `NotImplementedError` by
  default) constructs and returns the real model. `Loadable.__init__` calls it unless
  the first arg is already an `nn.Module`.
- `_load_meta(*args, **kwargs) -> nn.Module` (`meta.py:136`) builds a **meta-device**
  version so the Envoy tree exists without GPU memory. `Meta.__init__` runs it inside
  `with MetaDevice():` (which forces every tensor onto the meta device, `meta.py:31`)
  unless `dispatch=True` or an `nn.Module` was passed. `MetaDevice.real()` suspends
  the forcing for parts of a build that need real tensors.
- `dispatch()` (`meta.py:139`) calls `_load(*self.args, **self.kwargs)` then
  `_update(model)` to re-point the meta tree at real weights. It runs automatically on
  the first `interleave` if not already dispatched and not under fake tensors
  (`meta.py:177`). Override `_load`, not `dispatch`, if loading needs preconditions —
  vLLM tears down its meta process group inside `_load` (`vllm.py:191`).

`HuggingFaceModel` implements both from a repo id: `_load_meta` (`huggingface.py:51`)
via `AutoConfig` + `from_config`; `_load` (`:57`) via `from_pretrained`.

### Input & batching — `_batch_size` and `_batch`

The standard tracer always uses `Batcher(self.envoy)` (`intervention/tracer.py:245`),
which calls back into your model:

- `_batch_size(*inputs, **kwargs) -> int` (`envoy.py:588`) — how many batch rows an
  invoke contributes. **Base default:** `1` if there's any input else `0`. Override to
  report the true row count (`TransformersModel._batch_size`, `transformers.py:570`,
  counts a prompt/list/tensor).
- `_batch(invokes, fn) -> (args, kwargs)` (`envoy.py:597`) — combine multiple invokes'
  inputs into one call. **Base default:** pass a single invoke straight through; two or
  more raise `NotImplementedError`. Override to merge (`TransformersModel._batch`,
  `transformers.py:633`, dispatches by `fn.__name__` and pads/collates; `VLLM._batch`,
  `vllm.py:245`, extends prompt/params/lora lists — one request per invoke).

For an **exotic tensor layout** (a first dim that isn't the batch — vLLM's flat
`[total_tokens, hidden]`, tensor-parallel shards), subclass `Batcher`
(`intervention/batching.py:66`) and override `batching`/`narrow`/`widen`, then wire
your subclass onto the tracer's `self.batcher` (handed to `interleave(batcher=...)`)
inside your own execution path.
`VLLMBatcher` (`vllm/batching.py:32`) is the reference — it's installed by the vLLM
model runner, not by the generic tracer. See
[batching-internals.md](./batching-internals.md).

### Execution — the forward the tracer runs

`trace(...)` runs `fn`, defaulting to `"__call__"`. A runtime usually points it at its
own method:

- Override `trace` to set `kwargs.setdefault("fn", self._call)` (and to inject a
  custom backend/tracer — see below). `TransformersModel.trace` (`transformers.py:373`)
  and `VLLM.trace` (`vllm.py:330`) both do this.
- `_call(...)` runs the actual forward/engine request. `TransformersModel._call`
  (`transformers.py:546`) preprocesses inputs and calls the module; `VLLM._call`
  (`vllm.py:371`) serializes mediators onto the requests and drives the engine.
- Separate `generate` / `pipe` paths are just more `@traceable` methods that set a
  different `fn` (`TransformersModel.generate` runs the model and returns token ids;
  `pipe` runs the whole pipeline).
- Override `interleave` (`envoy.py:612`) only if your runtime doesn't run a local
  forward. `VLLM.interleave` (`vllm.py:434`) starts no local workers — they're
  serialized onto the engine's requests and started on the other side.

### Runtime-internal values — `eproperty`

To expose an engine value that isn't a module output (logits, samples, telemetry),
add an `eproperty` to your model class. The decorated stub is the read-side
preprocess — `def logits(self, value): return value` for an identity view — and its
location is `"{self.path}.{key}"`. Serve it where the value is produced, inside an
open interleaver context, with the eproperty's `.provide`:
`type(model).logits.provide(model, value)` (which forwards to
`self.interleaver.handle(location, value)` and returns it, edited if a worker wrote
back). Give it a `description=` to surface it in the model's repr. vLLM's
`logits`/`samples` (`vllm.py:144`, served in `GPUModelRunner.py:450`/`:472`) are the
reference. Full recipe in [extending-envoy.md](./extending-envoy.md).

### Remote support — `Remotable` hooks

If the runtime should run on NDIF, extend `Remotable` and implement:

- `_remoteable_model_key() -> str` (`remotable.py:108`) and classmethod
  `_remoteable_from_model_key(cls, key, **kwargs)` (`:111`) — the server-side identity.
  `to_model_key()` combines them with the class import path (`:125`).
- `_remoteable_persistent_objects() -> dict` (`:97`) — the `{id: object}` map the
  server resolves persistent IDs against (base: `{"Interleaver": ...}` plus a
  `Module:<path>` per envoy). Add your tokenizer/preprocessors here and tag them with
  `obj._persistent_id = name` in `__getstate__` so they're referenced, not pickled
  (`TransformersModel`, `transformers.py:460`/`:533`; `VLLM`, `vllm.py:456`/`:461`).
- `_remoteable_get_env()` / `_remoteable_set_env(env)` (`:79`/`:88`) — per-request
  environment applied server-side (e.g. `TransformersModel` transports a PEFT adapter).
- `_remoteable_class()` (`:115`) — return the canonical class if yours is a deprecated
  alias, so it shares one server key (`LanguageModel` returns `TransformersModel`).

`__getstate__` on `Envoy` (`envoy.py:248`) already tags the interleaver and modules;
runtimes with a live engine handle null it out (`VLLM.__getstate__` nulls
`vllm_entrypoint`; `DiffusionModel.__getstate__` pops `pipeline`). See
[serialization.md](./serialization.md).

### Custom backend / tracer

Inject a backend or `tracer_cls` from your `trace` override for non-standard
execution (async streaming, HTTP serve). vLLM injects `AsyncVLLMBackend` +
`VLLMTracer` for `mode="async"` and `LocalServeBackend` for `serve=url`
(`vllm.py:330`). See [vllm-integration.md](./vllm-integration.md) and
[adding-a-new-backend.md](./adding-a-new-backend.md).

## Lifecycle of a runtime trace

1. `with model.trace(input)` — the tracer captures the block.
2. On `__exit__`, the backend compiles the block; `Batcher(self.envoy)` runs each
   invoke's `_batch_size`/`_batch` to build the call input and per-invoke groups.
3. `model.interleave(fn, *args, **kwargs)` runs (dispatching first if needed). `fn`
   (your `_call`/`generate`/pipeline) executes the forward; hooks + `handle` serve
   parked interventions; the batcher narrows/widens per invoke.
4. The run's return is served at `"result"`; saved values are pushed back into the
   caller's frame.

## Reference implementations

- `src/nnsight/modeling/vllm/vllm.py` — the deepest case (non-PyTorch, two processes, async, serve)
- `src/nnsight/modeling/transformers.py` — pipeline-backed HF, full custom batching
- `src/nnsight/modeling/diffusion.py` — pipeline-as-module, custom loading
- `src/nnsight/modeling/base.py` — the trivial `NNsight(Envoy)` leaf

## Related

- [extending-envoy.md](./extending-envoy.md) — exposing runtime-internal values
- [vllm-integration.md](./vllm-integration.md) — vLLM as the reference non-PyTorch runtime
- [batching-internals.md](./batching-internals.md) — the `Batcher` contract
- [serialization.md](./serialization.md) — the `Remotable` persistent-object protocol
