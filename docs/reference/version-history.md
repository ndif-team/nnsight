---
title: Version History
one_liner: Pointer to release notes; brief summary of what each major version brought.
tags: [reference, history]
---

# Version History

For full release notes, see [`0.6.0.md`](https://github.com/ndif-team/nnsight/blob/main/0.6.0.md) in the repo root and the [GitHub Releases page](https://github.com/ndif-team/nnsight/releases).

## Upcoming: `refactor/transform` highlights

Currently in development on the `refactor/transform` branch (pre-release). Major themes:

- **Lazy hook execution.** Modules no longer carry permanent input/output hooks. Each module gets a thin skippable forward + a sentinel output hook; actual interception is done by **one-shot hooks** registered on demand by each mediator and self-removed after firing. Untouched modules incur effectively zero hook overhead.
- **`eproperty` extension API.** The `eproperty` descriptor formalizes how custom Envoy subclasses expose new hookable properties. Subclasses can now define `.heads`, `.logits`, etc. by stacking a pre-setup decorator (`@requires_output`, `@requires_operation_output`, …) on a stub method. Optional `preprocess` / `postprocess` / `transform` hooks let preprocessed views write back into the running model on in-place edit.
- **Source split.** Source tracing internals split into `SourceAccessor` (global per-fn rewrite + cache) and `OperationAccessor` (per-call-site hook lists), separating the user-facing `SourceEnvoy` / `OperationEnvoy` from the global hook bookkeeping. Multiple Envoys / Interleavers / sessions touching the same operation now share one accessor cleanly.
- **`envoys=` kwarg.** `NNsight(...)` (and any subclass) accepts `envoys=...` to control which `Envoy` class wraps descendants — either a single subclass for everything, or a `Dict[Type[nn.Module], Type[Envoy]]` matched via MRO. Subclasses can also set a class-level `envoys` attribute as a default.
- **Backwards-compat for `SourceAccessor` across module replacement.** When a wrapped module is replaced (e.g. weights dispatched, edits applied, hot-swapped), the existing `SourceAccessor`'s injected forward keeps working — the accessor is keyed by the unwrapped fn, not the module instance.

## Released versions

### v0.6.0 — see [`0.6.0.md`](https://github.com/ndif-team/nnsight/blob/main/0.6.0.md)

Headline features:

| Area | Change |
|------|--------|
| NDIF | Seamless serialization of local code via cloudpickle by-value (`nnsight.register(...)` for pip-installed packages; auto-registration for editable installs and your script's local imports). Python 3.9+ clients now work regardless of NDIF's Python version. |
| vLLM | Full vLLM integration: single GPU, multi-GPU tensor parallelism, Ray distributed executor, multi-node Ray, `mode="async"` with streaming. |
| Iteration | `tracer.iter` now supports plain `for` loops in addition to `with` blocks (faster — no code capture overhead). |
| NDIF utilities | `nnsight.compare()`, `nnsight.status()`, `nnsight.is_model_running(...)` for env diffing and deployment checks. |
| Diagnostics | Cleaner traceback reconstruction; nnsight internals hidden by default; `python -d` re-enables them. |
| Trace dispatch | Smarter trace-vs-invoker detection — kw-only inputs (`input_ids=...`) now correctly create implicit invokers. |
| Performance | 2.4–3.9x faster traces vs v0.5.15 via always-on trace caching, persistent pymount, removed `torch._dynamo.disable` wrappers, batched `PyFrame_LocalsToFast`, and filtered globals copy. `TRACE_CACHING` config is deprecated (always on). |
| New models | `VisionLanguageModel` (LLaVA / Qwen2-VL / etc.), `DiffusionModel` (UNet + transformer pipelines, with `DiffusionBatcher`). |
| Robustness | `python -c "..."` works; multiple `NNsight` wrappers on the same model coexist; reference-cycle memory leaks fixed. |
| Wire format | NDIF results compressed with zstandard. |

Breaking changes:

- Removed deprecated v0.4 namespace items: `nnsight.apply()`, `nnsight.log()`, `nnsight.local()`, `nnsight.cond()`, `nnsight.iter()`, `nnsight.stop()`, `nnsight.trace()`, and the `nnsight.list/dict/int/...` type wrappers (use plain Python builtins).
- Removed the `trace=` parameter on `.trace()` / `.generate()`.
- `obj.stop()` on arbitrary objects removed — use `tracer.stop()`.
- Newly deprecated (still works, will be removed): `model.iter`, `model.all()`, `model.next()` (use the `tracer.*` versions); `with tracer.iter[...]:` (use `for step in tracer.iter[...]:`).
- Custom `_prepare_input` / `_batch` implementations may need updating to match the `Batchable` interface in `nnsight/intervention/batching.py`.

### v0.5.x

Introduced the thread-based deferred-execution architecture (Tracer + Interleaver + Mediator + Envoy + eproperty). Standard Python `if`/`for` statements work inside traces, replacing the v0.4 `nnsight.cond()` / `session.iter()` DSL. v0.4 namespace items emit `DeprecationWarning`s.

### v0.4.x and earlier

Proxy-based deferred execution. Used `nnsight.apply()`, `nnsight.cond()`, `nnsight.iter()`, `nnsight.list()`, etc. as a tracing DSL. Replaced by the v0.5 architecture; documented here only for migration context.

## Where to read more

- Full v0.6 notes: [`0.6.0.md`](https://github.com/ndif-team/nnsight/blob/main/0.6.0.md)
- README: [`README.md`](https://github.com/ndif-team/nnsight/blob/main/README.md)
- Internals deep-dive: [`NNsight.md`](https://github.com/ndif-team/nnsight/blob/main/NNsight.md)
- Documentation site: [https://nnsight.net](https://nnsight.net)
