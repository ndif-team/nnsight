# Contributing to NNsight

Thanks for your interest in contributing! This guide covers getting set up, running
the tests, and the conventions this repo follows.

## Getting started

### Prerequisites

- Python 3.10+
- [PyTorch](https://pytorch.org/)
- Git, and a C compiler for the optional `.save()` mount extension (a failed build
  is a warning, not an error — nnsight falls back to `nnsight.save(value)`).

### Development setup

Clone your fork, then install in editable mode with the dev extras:

```bash
git clone https://github.com/<your-username>/nnsight.git
cd nnsight
pip install -e ".[dev]"
```

`[dev]` adds the test toolchain (pytest) and the extra runtimes the suite
exercises (diffusers, pillow, accelerate) on top of nnsight's core install —
which already includes transformers and the remote (NDIF) deps. The tests import
the **installed** package — there is no `sys.path` shim — so the editable install
is what makes them resolve to your working tree.

Quick smoke test (no model download):

```bash
pytest tests/test_tracing.py -q
```

## Running tests

Run the suite on CPU, skipping the vLLM tests (they need a GPU and the `vllm` extra):

```bash
CUDA_VISIBLE_DEVICES="" pytest tests/ -q --ignore=tests/vllm
```

Notes:

- Many tests download small models from the HuggingFace Hub on first run (gpt2, a few
  tiny random checkpoints, tiny diffusion pipelines); they're cached afterward.
- Diffusion tests `pytest.importorskip("diffusers")`, so they skip cleanly if
  diffusers isn't installed.
- The vLLM suite (`tests/vllm/`) needs a CUDA GPU and `pip install nnsight[vllm]` (or
  `pip install vllm`); it's excluded from the default run above.

## Project structure

```
src/nnsight/
├── _c/                       # optional C extension: mounts .save() on every object
├── tracing/                  # capture → parse → execute (Tracer, Backend, util, hint)
├── intervention/
│   ├── envoy.py              # the Envoy tree (the module proxy)
│   ├── eproperty.py          # descriptor behind .input/.output/tracer.result/...
│   ├── interleaver.py        # the Interleaver + Mediator (greenlets, event protocol)
│   ├── batching.py           # Batcher: narrow/widen, batch groups
│   ├── source.py             # .source AST instrumentation
│   ├── tracer.py             # InterleavingTracer / ScanningTracer / Invoker
│   ├── cache.py              # tracer.cache
│   ├── serialization.py      # source-based serialization for remote
│   └── backends/             # remote.py (blocking / non-blocking / async), local.py
├── modeling/
│   ├── base.py               # NNsight base class
│   ├── huggingface.py        # HuggingFaceModel
│   ├── transformers.py       # TransformersModel (the primary HF wrapper)
│   ├── diffusion.py          # DiffusionModel + DiffusionBatcher
│   ├── language.py / vlm.py  # LanguageModel / VisionLanguageModel (deprecated aliases)
│   ├── mixins/               # Loadable, Meta (lazy build/dispatch), Remotable
│   └── vllm/                 # the vLLM runtime
├── schema/                   # CONFIG (config.py), request/response schemas
└── util.py
```

### Architecture in one paragraph

nnsight runs on **deferred, interleaved execution**. Code inside `with model.trace(...)`
is captured as source, compiled, and run as a **greenlet** interleaved with the
model's forward pass: reading `.output` parks the greenlet until the model reaches
that module and a forward hook hands the value over; assigning to `.output` splices a
value in. The event protocol is `VALUE`/`SWAP`/`SKIP`/`BARRIER`. The full picture is
in [NNsight.md](./NNsight.md); the task docs are under [`docs/`](docs/), routed by
[CLAUDE.md](./CLAUDE.md).

## Making changes

### Code style

Read [STYLE.md](./STYLE.md) — it's the canonical house style. In short: tight,
present-tense prose in comments and docstrings that describe how the code *is* (not
its history); comments only where the logic isn't self-evident; keep changes focused.

### Writing tests

- Tests live in `tests/` and import `nnsight` from the installed package (no
  `sys.path` inserts).
- Keep them CPU-runnable; guard GPU-only paths and use `pytest.importorskip(...)` for
  optional dependencies.
- To collect values across a trace, **save the container** and store raw values —
  `xs = nnsight.save([]); xs.append(model...output)` — never `xs.append(x.save())`
  (see [docs/gotchas/save.md](docs/gotchas/save.md)).

## Submitting changes

1. Branch from the latest `main`: `git checkout -b my-feature main`.
2. Make the change and add tests; run `CUDA_VISIBLE_DEVICES="" pytest tests/ -q --ignore=tests/vllm`.
3. Commit with an `area: summary` subject (e.g. `tracing: ...`, `docs: ...`) and open
   a PR against `main`.
4. In the PR: what the change does and why, related issues, and any breaking changes.

## Reporting bugs

Open an issue at [github.com/ndif-team/nnsight/issues](https://github.com/ndif-team/nnsight/issues) with the
nnsight version (`python -c "import nnsight; print(nnsight.__version__)"`), your Python
and PyTorch versions, a minimal reproduction, and the full traceback.

## Community

- Documentation: [nnsight.net](https://nnsight.net)
- Forum: [discuss.ndif.us](https://discuss.ndif.us)
- Discord: [discord.gg/6uFJmCSwW7](https://discord.gg/6uFJmCSwW7)

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](./LICENSE).
