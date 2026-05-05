---
title: Testing
one_liner: How to run NNsight's test suite, what each test file covers, and pytest conventions used across the repo.
tags: [internals, dev]
related: [docs/developing/contributing.md, docs/developing/agent-evals.md, docs/developing/performance.md]
sources: [tests/conftest.py, pytest.ini, tests/]
---

# Testing

## What this covers

NNsight has a top-level `tests/` directory with pytest-driven tests for each subsystem, plus three nested directories for special purposes. This page is a map: how to invoke pytest, what command-line flags exist, what each test file is for, and conventions you should follow when adding tests.

## Architecture / How it works

### Pytest configuration

`pytest.ini` (project root):

```
[pytest]
pythonpath = tests
markers =
    scan, config, order, source, skips, iter, cache, rename
```

The `pythonpath = tests` line lets test modules import each other by name, and lets `mymethods/` and other helpers be imported as top-level packages from inside tests.

The markers are used to select subsets of tests within `test_lm.py`. For example: `pytest tests/test_lm.py -m iter --device cpu`.

### Custom command-line options

`tests/conftest.py:21` defines:

| Flag | Default | Purpose |
|------|---------|---------|
| `--device` | `cuda:0` if available else `cpu` | Device for model fixtures. Parametrizes any test that takes a `device` argument. |
| `--tp` | `1` | Tensor parallel size for vLLM tests. Read inside vLLM-specific tests. |
| `--test-flux` | False | Opt-in flag for slow Flux diffusion tests that need GPU + a model download. |

Every test fixture that loads a model (`tiny_model`, `gpt2`, `vlm`, `tiny_sd`, `flux`) honors `--device`. Skip GPU-only tests cleanly when no GPU is available by guarding on `--device`.

### Shared fixtures

Defined in `tests/conftest.py`:

- `tiny_model` (scope: module) — two-layer Linear `nn.Sequential` wrapped with `NNsight`. Used by `test_tiny.py`, `test_envoys.py`, etc.
- `tiny_input` — random `[1, 5]` tensor matching `tiny_model`'s input.
- `gpt2` (scope: module) — `LanguageModel("openai-community/gpt2", device_map=device, dispatch=True)`. Used by `test_lm.py`, `test_remote.py`, `test_source.py`, etc.
- `vlm` (scope: module) — `VisionLanguageModel("llava-hf/llava-interleave-qwen-0.5b-hf", ...)` plus `dummy_image`.
- `tiny_sd` (scope: module) — `DiffusionModel("segmind/tiny-sd", ...)` for fast diffusion tests.
- `flux` (scope: module) — `FLUX.1-schnell`, gated behind `--test-flux`.
- `ET_prompt`, `MSG_prompt` — common prompt strings used in tests.

### Top-level test files

| File | Coverage |
|------|----------|
| `test_tiny.py` | Smoke tests on the minimal `NNsight` wrapper. The fastest way to validate a refactor. |
| `test_lm.py` | Comprehensive `LanguageModel` coverage — invokers, generation, gradients, scan, caching, source tracing, module renaming, skipping. Uses pytest markers (`scan`, `iter`, `cache`, `source`, `rename`, `skips`, `order`). |
| `test_vllm.py` | vLLM integration. Requires `--tp` flag for tensor parallelism scenarios. Heavy — needs GPU and vllm install. |
| `test_vllm_dispatch_bug.py` | Specific regression tests for vLLM dispatch ordering. |
| `test_vlm.py` | `VisionLanguageModel` (multimodal). |
| `test_diffusion.py` | `DiffusionModel`. Uses `tiny_sd` fixture (1.4M-param model, runs on CPU). |
| `test_envoys.py` | `Envoy` proxy semantics — module access, alias paths, `.path`, `.modules()`. |
| `test_remote.py` | NDIF remote execution. Requires API key + network. |
| `test_serialization_edge_cases.py` | Stress tests for the source-based pickler — recursive functions, mutual recursion, lambda extraction edge cases, large object graphs. |
| `test_lambda_serialization.py` | Lambda-specific serialization tests (multiple lambdas per line, nested lambdas, etc.). |
| `test_dataclass_serialization.py` | Cross-version dataclass round-trip tests. |
| `test_local_simulation.py` | `remote='local'` round-trip simulation tests. |
| `test_local_recursion.py`, `test_local_mutual_recursion.py`, `test_mutual_recursion.py` | Recursive / mutually recursive function serialization. |
| `test_whitelist_serialization.py` | Allowlist enforcement for `LocalSimulationBackend.SERVER_MODULES`. |
| `test_local_env.py` | Module discovery for the local-package auto-registration path. |
| `test_debug.py` | DEBUG mode (`CONFIG.APP.DEBUG = True`) — verifies internal frames appear. |
| `test_memory_cleanup.py` | Hook leak / weak-reference tests. Verifies wrappers are GC-able. |
| `test_multiple_wrappers.py` | Wrapping the same `nn.Module` with multiple `NNsight` instances coexists correctly. |
| `test_source.py` | Source tracing (`.source.<op_name>`). |
| `test_transform.py` | Tests for the in-progress `transform` refactor on this branch. |
| `test_0516_features.py` | Features added on/around 2024-05-16. Catch-all for features that don't fit other files. |
| `test_tp_stream_fix.py` | Specific regression test for vLLM TP streaming. |
| `test_envoys.py` | Envoy mechanics. |

Helpers in `tests/`:

- `debug_demo.py` — manual demo script.
- `explore_remote.py`, `explore_remote_advanced.py` — manual remote exploration scripts.
- `repro_631.py` — minimal reproduction for issue #631.

### Subdirectories

- `tests/agent-evals/` — LLM agent evaluation suite. See [agent-evals.md](./agent-evals.md). Not run as part of the standard pytest suite.
- `tests/mymethods/` — fixture package referenced by tests that need a real importable user package. `__init__.py` and `stateful.py` define functions that live in a "user module" for serialization tests.
- `tests/performance/` — benchmark scripts (`benchmark_interventions.py`) and saved results. See [performance.md](./performance.md).

### The standard validation suite

The set of test files used in CI / referenced by `CONTRIBUTING.md`:

```bash
pytest tests/test_lm.py tests/test_tiny.py tests/test_0516_features.py \
  tests/test_debug.py tests/test_memory_cleanup.py tests/test_multiple_wrappers.py \
  --device cpu
```

Quick smoke test:

```bash
pytest tests/test_tiny.py --device cpu -x
```

vLLM-only:

```bash
pytest tests/test_vllm.py --tp 1
```

## Conventions

- **Run on CPU when possible.** Most tests should pass with `--device cpu`. GPU-specific tests should `pytest.skip` if no GPU is available.
- **Use `@torch.no_grad()` for inference-only tests.**
- **`pytest.importorskip()` at the top of optional-dep test files.** `test_diffusion.py` imports diffusers; `test_vllm.py` imports vllm. Skip the whole module if missing.
- **Pattern selection.** `pytest tests/test_lm.py -k logit_lens` runs only tests whose names contain "logit_lens".
- **Marker selection.** `pytest tests/test_lm.py -m "iter or cache" --device cpu`.
- **Fixture scope.** Model fixtures are `scope="module"` to avoid reloading a heavy model per test. Don't share mutable state between tests within a module.
- **Conda environment.** Local development typically uses a conda env with nnsight in editable mode (`pip install -e ".[test]"`), plus optional dependencies (`diffusers`, `vllm`) added per-environment as needed.

## Key files / classes

- `pytest.ini` — pytest config + marker registration
- `tests/conftest.py:21` — `pytest_addoption` (CLI flags)
- `tests/conftest.py:43` — `pytest_generate_tests` (parametrize on device)
- `tests/conftest.py:71` onward — fixtures (`tiny_model`, `gpt2`, `vlm`, `tiny_sd`, `flux`)
- `tests/agent-evals/` — agent evaluation suite (separate harness)
- `tests/mymethods/` — importable user-module fixture for serialization tests
- `tests/performance/` — benchmarks (not pytest-driven; see [performance.md](./performance.md))

## Lifecycle (a typical pytest run)

1. `pytest_addoption` registers `--device`, `--tp`, `--test-flux`.
2. `pytest_generate_tests` parametrizes `device`-using tests with the value from `--device`.
3. Module-scoped fixtures load models once per module.
4. Tests run, optionally filtered by markers (`-m`) or names (`-k`).
5. After each module, fixtures tear down (model goes out of scope, weights are freed).

## Extension points

- **Add a new marker.** Append it to `pytest.ini` and tag relevant tests with `@pytest.mark.<name>`. Filter via `-m`.
- **Add a new fixture.** Put it in `conftest.py` with appropriate `scope`. Cross-test fixtures live there; one-test fixtures can live in the test file itself.
- **Add an opt-in heavy test.** Mirror `--test-flux`: register a new flag in `pytest_addoption`, gate the fixture on `request.config.getoption("--test-flux")`.
- **Importable user package fixtures.** Add modules to `tests/mymethods/` if you need code that lives in a "user package" path. See `tests/test_local_env.py` for usage.

## Related

- [contributing.md](./contributing.md) — pre-PR checklist, standard validation suite
- [performance.md](./performance.md) — `tests/performance/` benchmarks
- [agent-evals.md](./agent-evals.md) — `tests/agent-evals/` harness
