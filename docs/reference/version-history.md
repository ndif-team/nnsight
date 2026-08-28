---
title: Version History
one_liner: What the pipeline rewrite changed and how it maps to the old API.
tags: [reference, history]
---

# Version History

For release notes, see the [GitHub Releases page](https://github.com/ndif-team/nnsight/releases).

## 0.8 — the pipeline rewrite

Version 0.8 is a ground-up rewrite of nnsight's execution model. If you know the older API, here is what changed and where things moved.

### Interleaving: greenlets, not threads

Intervention code and the model's forward pass now run in **greenlets** (cooperative, single-threaded coroutines), not OS worker threads. Each block runs in its own worker greenlet (`Mediator`) that switches control back to the model side whenever it parks on a location. Because only one greenlet runs at a time, there are no locks or queues.

The worker/model event protocol is `VALUE` / `SWAP` / `SKIP` / `BARRIER` (the `Event` enum). The old `END` / `EXCEPTION` events are gone.

### Model classes

- `NNsight(module)` — base wrapper for any `torch.nn.Module`.
- `TransformersModel("repo/id", task=...)` — the **primary** HuggingFace class, backed by a `transformers.pipeline`; supports any task.
- `DiffusionModel` — any `diffusers` pipeline (UNet- or transformer-based).
- `VLLM(..., mode="sync"|"async")` — vLLM-backed, interventions run inside the engine worker.
- `LanguageModel` / `VisionLanguageModel` — now **deprecated** thin subclasses that warn on construction; use `TransformersModel(task=...)`.

### `generate` vs `pipe`

`model.generate(...)` now generates through the **model** and returns **token ids** (read `tracer.result`), greedy by default. `model.pipe(...)` runs the whole task **pipeline** and returns its records (decoded text, labels, ...) — that is what the old `generate` returned. `model.trace(...)` runs one forward; `model.scan(...)` runs one forward under fake tensors for shape inference.

### `eproperty` reintroduced

The `eproperty` descriptor is back, with a new API. Decorate a stub with `@eproperty` (or `@eproperty(key=..., description=...)`) to define a hookable value; the stub is the *preprocess* mapping the served value to what the user reads, refined by `.postprocess` (a written value before the swap), `.transform` (write an edited preprocess view back to the model's layout), and `.provide` (serve the value from the model side via `interleaver.handle`). A `description=` surfaces the value in the Envoy repr tree. `Envoy.input` / `.inputs` / `.output`, `tracer.result`, and `VLLM.logits` / `.samples` are all built on it; add your own on a model subclass. See [extending.md](../usage/extending.md).

### `save` is guarded

`nnsight.save(x)` / `x.save()` now **raises** if called outside a trace (it used to be a silent no-op). It marks a value to survive past the enclosing `with model.trace(...):` block, so it only makes sense inside one.

### Iteration

Target occurrences across a repeated run with `tracer.iter[...]` (loop form: `for step in tracer.iter[:3]:`) or `tracer.all()`. The `with tracer.iter[...]:` block still works but is deprecated. `tracer.next()`, `model.iter`, and `model.all()` are gone or deprecated.

### Config

`CONFIG.API.HOST` / `APIKEY` / `COMPRESS` and `CONFIG.APP.DEBUG` / `REMOTE_LOGGING` / `PYMOUNT`. Loaded from a user file (`~/.config/nnsight/config.yaml`) over shipped defaults, then env (`NDIF_API_KEY`, `NDIF_HOST`, `NNSIGHT_DEBUG`). The old `CROSS_INVOKER`, `CACHE_DIR`, and `TRACE_CACHING` settings are gone. See [config.md](./config.md).

### Remote

- `remote=True` → `RemoteBackend` (blocking over one websocket, or `blocking=False` submit/poll).
- `remote="local"` → `LocalSimulationBackend` (serialize/deserialize dry run, offline).
- `AsyncRemoteBackend` — `await backend` for the saves dict, `async for` for streamed status updates.
- Model identity via `model.to_model_key()` / `Class.from_model_key(...)`.
- The old `tracer.local()` hybrid streaming is **not** ported.

### Removed v0.4-era namespace

`nnsight.apply()`, `nnsight.log()`, `nnsight.local()`, `nnsight.cond()`, `nnsight.iter()`, `nnsight.session()`, and the `nnsight.list/dict/int/...` type wrappers are removed — use plain Python and `model.session()`.

### `.source`: assignments are operations; decorated forwards are instrumented

Every assignment in an instrumented forward is an operation, `{target}_{n}`, on
the same per-name counter as calls; its `.output` is the assigned value. Values
that are not a call's return — a product, a loop's running state — are reachable
by name, one fire per loop iteration. **Labels shift where a forward binds a name
and then calls it**: GPT-2's attention call is now `attention_interface_1`
(`attention_interface_0` is the line that chooses the implementation). Requesting
the old label raises nothing — it returns the assigned value instead — so update
any code that used it.

Decorated forwards no longer raise `SourceNotAvailable`: a wrapper that calls the
function it closes over is peeled and rebuilt around the instrumented function; a
dispatching wrapper (transformers' `experts_implementation`) is instrumented as it
is and shows the dispatch, so `experts.source.experts_forward_1.source` reaches the
implementation that ran. Closures and `super()` forwards are instrumented too.
`SourceNotAvailable` now means only "no Python source" (a builtin or C function).

## Where to read more

- README: [`../../README.md`](../../README.md)
- Documentation site: [https://nnsight.net](https://nnsight.net)
