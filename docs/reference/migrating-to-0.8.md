---
title: Migrating to 0.8
one_liner: Porting code written for nnsight before 0.8 — removed names and their errors, deprecated names and their warnings, and the behavior changes that bite.
tags: [reference, migration, deprecation]
---

# Migrating to 0.8

Release notes are on the [GitHub Releases page](https://github.com/ndif-team/nnsight/releases). This page lists only what breaks or changes behavior in code written for earlier versions.

## Removed — raise `AttributeError`

`nnsight.apply()`, `nnsight.log()`, `nnsight.local()`, `nnsight.cond()`, `nnsight.iter()`, `nnsight.session()`, and the `nnsight.list/dict/int/...` type wrappers are gone — use plain Python and `model.session()`. Each reports itself the same way, naming no replacement:

```
AttributeError: module 'nnsight' has no attribute 'list'
```

`CONFIG.APP.CROSS_INVOKER`, `CACHE_DIR` and `TRACE_CACHING` are gone the same way (`AttributeError: 'AppConfig' object has no attribute 'CROSS_INVOKER'`). `tracer.local()` (hybrid streaming, not ported), `tracer.next()` and `module.next()` report a missing attribute on the tracer or envoy. A saved value is the value itself, so `.value` raises `AttributeError: 'Tensor' object has no attribute 'value'`.

## Deprecated — warn under `NNsightDeprecationWarning`

`nnsight.NNsightDeprecationWarning` is a `FutureWarning`, so it shows wherever the call is, including inside imported modules. Silence nnsight's alone with `warnings.filterwarnings("ignore", category=nnsight.NNsightDeprecationWarning)`; nnsight registers no filters of its own.

| Old | Message | Use instead |
|---|---|---|
| `LanguageModel(repo)` | `LanguageModel is deprecated; use TransformersModel(repo_id, task='text-generation') instead.` | `TransformersModel(repo, task="text-generation")` |
| `VisionLanguageModel(repo)` | `VisionLanguageModel is deprecated; use TransformersModel(repo_id, task='image-text-to-text') instead.` | `TransformersModel(repo, task="image-text-to-text")` |
| `model.iter[...]` | `model.iter is deprecated; use tracer.iter instead.` | `tracer.iter[...]` |
| `model.all()` | `model.all() is deprecated; use tracer.all() instead.` | `tracer.all()` |
| `with tracer.iter[...]:` / `with tracer.all():` | ``The `with tracer.iter[...]:` / `with tracer.all():` block form is deprecated; use `for step in tracer.iter[...]:` instead.`` | `for step in tracer.iter[...]:` |
| `model.generator.output` (read or write) | `model.generator.output is deprecated; use tracer.result instead (model.generator.streamer.output still gives per-step tokens).` | `tracer.result` |
| `nnsight.ndif_status()` | `nnsight.ndif_status() is deprecated; use nnsight.status() instead.` | `nnsight.status()` |

`model.generator.output` and `tracer.result` carry the same tensor (prompt ids followed by generated ones). Per-step tokens are `model.generator.streamer.output`, which is not deprecated.

## Behavior changes

- **`generate` returns token ids; `pipe` returns what `generate` used to.** `model.generate(...)` generates through the model and returns ids on `tracer.result`, decoding with the checkpoint's own `generation_config` (many instruct checkpoints sample by default — pass `do_sample=False` for determinism). `model.pipe(...)` runs the task pipeline and returns its records (decoded text, labels). See [generate.md](../usage/generate.md).
- **`save()` raises outside a trace** instead of silently doing nothing. Move every save inside the `with` block.
- **A loop that outruns the run is cut short.** Old nnsight bounded `tracer.all()` internally. Now a `tracer.iter` loop that asks for a step the run does not make — including every open `iter[:]` / `all()` — warns, keeps what it saved, and drops the statements after the loop. Hold the run to the count (`min_new_tokens=`; `min_tokens=` or `ignore_eos=True` on vLLM) or move trailing statements into a separate `tracer.invoke()`. See [iteration.md](../usage/iteration.md).
- **PEFT adapters attach in place.** `peft=` / `model.load_adapter` no longer go through `PeftModel`, so module paths are the base model's, not `model.base_model.model...`.
- **A submodule named `output` or `input` moved.** `.nns_output` and `.nns_input` are gone (`AttributeError`). The submodule is at `.E_output` / `.E_input`, and `.output` is the module's own value, as on every other module. See [Overloaded names](../usage/access-and-modify.md#overloaded-names).
- **`automodel=` no longer picks the class.** `LanguageModel` and `TransformersModel` ignore it without an error, and the task decides: `TransformersModel(repo, task="fill-mask")` builds the masked-LM head. `DiffusionModel` still takes `automodel=`.
