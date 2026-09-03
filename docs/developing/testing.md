---
title: Testing
one_liner: How to run NNsight's test suite offline, the C extension it exercises, and what each test file covers.
tags: [internals, dev]
related: [docs/developing/contributing.md, docs/developing/performance.md]
sources: [tests/conftest.py, tests/vllm/conftest.py, pyproject.toml, setup.py, tests/]
---

# Testing

## What this covers

`tests/` mirrors the source modules — one test file per concept, named for it (see
`STYLE.md`, "Tests mirror modules"). This page is how to run it offline, the C
extension a couple of tests depend on, the small HuggingFace models used, and a map
of the files.

## Running the suite

The whole offline suite (everything except the two subtrees with hardware or
version requirements):

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ --ignore=tests/vllm --ignore=tests/tp
```

- **`LD_LIBRARY_PATH`** points at the conda env's libs so the compiled C extension
  (`nnsight._c.py_mount`, see below) and torch load their shared objects.
- **`CUDA_VISIBLE_DEVICES=""`** forces CPU so the tests are deterministic and run
  without a GPU.
- **`--ignore=tests/vllm`** skips the vLLM integration tests, which need a GPU and a
  `vllm` install.
- **`--ignore=tests/tp`** skips the tensor-parallel tests. Drop it on transformers
  **>= 5.16** and keep it below that: 5.16 rebuilt tensor parallelism on DTensor,
  and `nnsight.modeling.tp` targets that build
  (`MINIMUM_TRANSFORMERS`, `src/nnsight/modeling/tp/fragments.py`). On an older
  transformers `tests/tp/` errors at collection and
  `tests/test_tensor_parallel_rules.py` fails — 45 red results that say nothing
  about your change.

That collects 933 tests; 1031 with `tests/tp/` included. Config is minimal —
`pyproject.toml` has only:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

There are no custom pytest CLI flags and no registered markers beyond stock
`parametrize`/`skipif`, and no `pytest-timeout` — `--timeout` is not available.
Select tests the usual ways:

```bash
python -m pytest tests/test_source.py -q                 # one file
python -m pytest tests/test_language.py -k generation    # by name
python -m pytest tests/test_saving.py -x                 # stop on first failure
```

`tests/conftest.py` inserts `src/` on `sys.path` and defines one shared fixture:
`gpt2` (module-scoped, a dispatched `TransformersModel("openai-community/gpt2")`).
Heavy models are shared fixtures; small ones are constructed inline per test file.

**Name a subtree, or ignore the ones you are not running.** `tests/` has no
`__init__.py` files, so pytest imports each test module by its bare basename, and
`tests/test_tracing.py` and `tests/vllm/test_tracing.py` claim the same one. A
plain `pytest tests/` therefore aborts with `import file mismatch` on whichever it
reaches second, whether or not `vllm` is installed. Running the two trees
separately — or with the `--ignore` flags above — avoids it. `tests/vllm/` also
has one file that reads a `vllm` submodule at import (`test_moe_batching.py`), so
it errors at collection rather than skipping when `vllm` is absent; the rest of
the subtree collects (207 tests) and skips at run time.

## The C extension (`.save()`)

`obj.save()` is a method mounted onto Python's base `object` type by the optional
`nnsight._c.py_mount` C extension (`src/nnsight/_c/py_mount.c`), built by `setup.py`.
It requires a C compiler at install time; if none is present, setuptools silently
skips it and `obj.save()` is unavailable (`nnsight.save(obj)` still works). The
tests that assert the method form are guarded:

```python
@pytest.mark.skipif(not save_mounted, reason="C .save() mount not built/enabled")
```

If `tests/test_saving.py::TestSaveMethod` is skipped, rebuild the extension
(`pip install -e .` in an env with `gcc`/`libc6-dev`) — server images install those
precisely so `.save()` works remotely.

## Offline models

Tests use small models pulled from the HF cache (all offline once cached):

| Task | Repo id |
|------|---------|
| text-generation | `openai-community/gpt2`, `hf-internal-testing/tiny-random-LlamaForCausalLM` |
| fill-mask | `hf-internal-testing/tiny-random-BertForMaskedLM` |
| text-classification | `hf-internal-testing/tiny-random-DistilBertForSequenceClassification` |
| image-classification | `hf-internal-testing/tiny-random-ViTForImageClassification` |
| image-text-to-text (VLM) | `trl-internal-testing/tiny-LlavaForConditionalGeneration` |
| diffusion | `hf-internal-testing/tiny-stable-diffusion-torch` |

Optional-dependency modules skip themselves at import: `test_diffusion.py`
(`importorskip("diffusers")`), `test_vision.py`/`test_vlm.py` (`importorskip("PIL")`),
IPython-display assertions in `test_tracing.py` (`importorskip("IPython")`), and PEFT
paths in `test_language.py` (`skipif(not peft_installed)`).

## Test map

| File | Covers |
|------|--------|
| `test_tracing.py` | block capture/parse/compile, `Info`, block cache, traceback surgery |
| `test_interleaving.py` | `Mediator`, `Interleaver`, the controller, envoy access/editing, iteration, source iteration, cache, session, the `.save()` mount |
| `test_envoy.py` | Envoy tree, attribute passthrough, repr, setattr, rename, device, `result`, `tracer_cls`, deprecated `iter` aliases, out-of-interleaving errors |
| `test_source.py` | `.source` listing, capture, inputs, editing, skip, install, recursive drill-in |
| `test_batching.py` | invokers, trace/generate batching, multi-invoke skip, barriers, invoke scope, invoker input formats |
| `test_backward.py` | `with tensor.backward():` gradient access |
| `test_saving.py` | `save()` / `.save()` (function and method forms), nested saves, thread safety |
| `test_editing.py` | `model.edit(...)`, edit-with-attachment, edit serialization |
| `test_serialization.py` | source-based `dumps`/`loads`, lambdas, recursion, local env, linecache, `remote="local"` simulation, server execution, model keys |
| `test_remote_backend.py` | `AsyncRemoteBackend` websocket status stream consumption |
| `test_ndif.py` | `nnsight.ndif` helpers (register, status, env comparison, `pull_env`) offline |
| `test_modeling.py` | `_update`/meta device/loadable/meta/scan/import path/remotable/remote env, HF + Transformers construction, status display, backend threading |
| `test_language.py` | `TransformersModel` (gpt2) — generation, activation edits, gradients, ad-hoc modules, input setting, source, early stop, iteration, session, cache |
| `test_encoder.py` | encoder tasks (fill-mask, text-classification), padding, `pipe` |
| `test_vision.py` | image-classification pipeline, `pipe` |
| `test_vlm.py` | vision-language models + deprecated `VisionLanguageModel` |
| `test_chat.py` | chat-formatted (role/content) input |
| `test_diffusion.py` | `DiffusionModel` (tiny stable-diffusion) |
| `test_config.py` | config precedence, Colab userdata fallback, `set_default_api_key` |
| `test_memory.py` | trace teardown leaves no reference cycles (model/saved-object/tracer/exception) |
| `test_multiple_wrappers.py` | one module wrapped by several `NNsight`s at once, incl. source/skip |
| `test_util.py` | `apply`/leaf/container traversal helpers |
| `test_fragments.py` | `Fragments` — the seam a distributed runtime hangs its gather on |
| `test_construction_routing.py` | constructor routing through the `Loadable`/`Meta` mixins |
| `test_chunked_tasks.py` | HuggingFace chunk pipelines (one row per chunk in one forward) |
| `test_deprecations.py` | every declared deprecation, and that each warns from an imported module as well as from `__main__` |
| `test_quantization.py` | naming a quantization in the `dtype` slot, checked without a GPU |
| `test_tensor_parallel_rules.py` | the tensor-parallel rule table, checked without a GPU |
| `test_login.py` | the `nnsight login` CLI entry point |

### `tests/tp/` (transformers >= 5.16; two devices for the GPU files)

Run separately: `python -m pytest tests/tp/`. `test_cpu_gloo.py` (tensor
parallelism over gloo, so CI covers it at all) and `test_cpu_expert_parallel.py`
(experts split rather than tensors) run on CPU; `test_sharded_tracing.py` (sharded
activations read and edit as whole tensors) and `test_real_model.py` (the same
against a real checkpoint) need two GPUs and skip without them. `worker.py` and
`ep_worker.py` are one rank each, launched under `torch.distributed.run` — they
carry no `test_` prefix precisely so pytest does not collect them.

### `tests/vllm/` (GPU + `vllm` required)

Run separately: `python -m pytest tests/vllm/`. Its `conftest.py` sets
`VLLM_ALLOW_INSECURE_SERIALIZATION=1` (only for the request-state tests, which ship a function to the workers through `collective_rpc`; tracing itself needs no such flag).
Files: `test_tracing.py` (logits/generation/sampling/interventions/input forms/early
stop/deferred errors/cache), `test_async.py` (async engine streaming),
`test_async_ray.py` (the async loop driven over Ray), `test_serve.py`
(the `nnsight-serve` HTTP path + GPU-less client), `test_tensor_parallel.py` (sharded
read/edit), `test_deepseek.py` (a second sharded architecture — MLA + expert
parallelism), `test_moe_batching.py` (deferred-reduce partials gather on read),
`test_taps.py` (tracing an engine replaying CUDA graphs, at declared taps),
`test_registration.py` (blocks installed on the engine, applying to every request),
`test_chunked_prefill.py` (a traced prompt is prefilled whole),
`test_preemption.py` (a preempted request comes back whole), `test_lora.py`
(interventions on a request carrying an adapter),
`test_requests.py` (client/worker cleanup, batch isolation, concurrency),
`test_ray.py` (Ray executor). `redteam_repros.py` is deliberately not a `test_`
module — it prints raw observations rather than asserting. TP tests use
Qwen2.5-0.5B; single-rank tests use gpt2.

## Conventions

- **Run on CPU** (`CUDA_VISIBLE_DEVICES=""`); the offline suite is CPU-clean.
- **One test file per source concept**, named for it. Behavior groups into `Test*`
  classes; the test name carries the assertion (`test_assignments_pushed_to_parent`),
  so tests need no docstrings.
- **Prefer inline fakes** — define small `nn.Module`s in the test file. Keep only the
  expensive shared models in `conftest.py`.
- **`importorskip` at the top** of an optional-dependency test file.

## Related

- [contributing.md](./contributing.md) — pre-PR routine and house style
- [performance.md](./performance.md) — the `tests/performance/` benchmark harness
