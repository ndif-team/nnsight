# Contributing to NNsight

Thanks for your interest in contributing to nnsight! This guide covers everything you need to get started.

## Getting Started

### Prerequisites

- Python 3.10+
- [PyTorch](https://pytorch.org/) >= 2.4.0
- Git

### Development Setup

1. Fork and clone the repository:

```bash
git clone https://github.com/<your-username>/nnsight.git
cd nnsight
```

2. Install in development mode:

```bash
pip install -e ".[test]"
```

This installs nnsight in editable mode along with test dependencies (pytest, pytest-cov).

3. Verify your setup:

```bash
pytest tests/test_tiny.py --device cpu -x
```

### Optional Dependencies

- **diffusers**: Required for `DiffusionModel` support. Install with `pip install diffusers`.
- **vllm**: Required for vLLM integration. Install with `pip install vllm>=0.12`.

These are not required for core development or running the main test suite.

## Project Structure

```
src/nnsight/
├── _c/                          # C extensions (pymount)
├── intervention/
│   ├── batching.py              # Batchable interface
│   ├── envoy.py                 # Core Envoy wrapper (module proxy)
│   ├── interleaver.py           # Hook dispatch, thread management
│   ├── backends/                # Execution backends
│   ├── tracing/
│   │   ├── base.py              # Tracer base class, source extraction
│   │   ├── tracer.py            # InterleavingTracer, ScanningTracer
│   │   ├── invoker.py           # Invoker for batched execution
│   │   ├── iterator.py          # Step iteration (tracer.iter)
│   │   ├── globals.py           # Global state, pymount lifecycle
│   │   ├── backwards.py         # Gradient tracing
│   │   └── util.py              # Exception handling, frame utils
│   └── serialization.py         # NDIF serialization
├── modeling/
│   ├── base.py                  # NNsight base class
│   ├── language.py              # LanguageModel
│   ├── vlm.py                   # VisionLanguageModel
│   ├── diffusion.py             # DiffusionModel
│   ├── huggingface.py           # HuggingFace model mixin
│   ├── transformers.py          # Transformers integration
│   └── mixins/                  # Meta loading, dispatch, remote
├── schema/                      # Config schema
├── ndif.py                      # Remote execution (NDIF)
└── util.py                      # Utilities
```

### Key Architectural Concepts

NNsight uses **deferred execution with thread-based synchronization**:

1. Code inside `with model.trace(...)` is extracted via AST, compiled, and run in a worker thread.
2. When the thread accesses `.output` or `.input`, it blocks until the model's forward pass provides the value via a PyTorch hook.
3. Each invoke runs in its own thread, executing serially in definition order.

Understanding this architecture is important for working on the intervention system. See [CLAUDE.md](./CLAUDE.md) and [NNsight.md](./NNsight.md) for deep technical documentation.

## Running Tests

### Full Test Suite

```bash
pytest tests/test_lm.py tests/test_tiny.py tests/test_0516_features.py \
  tests/test_debug.py tests/test_memory_cleanup.py tests/test_multiple_wrappers.py \
  --device cpu
```

### Quick Smoke Test

```bash
pytest tests/test_tiny.py --device cpu -x
```

### Diffusion Tests

Requires `diffusers` to be installed. Uses a tiny test model (~1.4M params) that runs on CPU:

```bash
pytest tests/test_diffusion.py --device cpu
```

### GPU Tests

If you have a CUDA GPU available:

```bash
pytest tests/test_lm.py --device cuda:0
```

### Test Markers

Tests are organized with pytest markers: `scan`, `source`, `iter`, `cache`, `rename`, `skips`, `order`. You can run a specific category:

```bash
pytest tests/test_lm.py -m iter --device cpu
```

## Making Changes

### Before You Start

- Check [existing issues](https://github.com/ndif-team/nnsight/issues) to see if someone is already working on what you have in mind.
- For larger changes, open an issue first to discuss the approach.
- For bug fixes, include a minimal reproduction if possible.

### Code Style

- Follow existing patterns in the codebase. NNsight doesn't use a formatter, but consistency with surrounding code is expected.
- Keep changes focused. A bug fix shouldn't include unrelated refactoring.
- Don't add docstrings, comments, or type annotations to code you didn't change.
- Only add comments where the logic isn't self-evident.

### Writing Tests

- Tests go in the `tests/` directory.
- Use the existing fixtures in `tests/conftest.py` where possible.
- All tests should pass on CPU (`--device cpu`). GPU-specific tests should be skipped when no GPU is available.
- Use `@torch.no_grad()` for inference-only tests.
- For optional dependencies (like diffusers), use `pytest.importorskip()` at the top of the test file.

### Common Development Pitfalls

- **Module access order matters.** Within a single invoke, modules must be accessed in forward-pass execution order. Accessing layer 5 then layer 2 will deadlock.
- **`.save()` is required** to persist values outside a trace context. Values without `.save()` are garbage collected.
- **Source cache issues.** If you hit `AttributeError: 'Info' object has no attribute 'pull'`, clear the cache with `Globals.cache.clear()` from `nnsight.intervention.tracing.globals`.
- **Benchmarking.** Define trace functions at module level (not inside loops). Warm up 2-3 iterations before timing. Use `torch.cuda.synchronize()` for GPU timing.

## Submitting Changes

1. Create a branch from the latest `main`:

```bash
git checkout -b my-feature main
```

2. Make your changes and add tests.

3. Run the test suite:

```bash
pytest tests/test_tiny.py tests/test_lm.py --device cpu -x
```

4. Push your branch and open a pull request against `main`.

5. In your PR description:
   - Explain what the change does and why.
   - Reference any related issues.
   - Note any breaking changes.

If your PR isn't getting the attention it deserves, come bug us on [Discord](https://discord.gg/6uFJmCSwW7)!

## Reporting Bugs

Open an issue at [github.com/ndif-team/nnsight/issues](https://github.com/ndif-team/nnsight/issues) with:

- NNsight version (`python -c "import nnsight; print(nnsight.__version__)"`)
- Python and PyTorch versions
- A minimal code example that reproduces the issue
- The full error traceback

## Community

- **Documentation:** [nnsight.net](https://nnsight.net)
- **Forum:** [discuss.ndif.us](https://discuss.ndif.us)
- **Discord:** [discord.gg/6uFJmCSwW7](https://discord.gg/6uFJmCSwW7)

## License

By contributing to nnsight, you agree that your contributions will be licensed under the [MIT License](./LICENSE).
