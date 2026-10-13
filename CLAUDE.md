# nnsight — Agent Guide

**Start at [docs/index.md](docs/index.md)**, the router for every docs section. Find the one page for the task and read it before writing code. Something broke? Start at [docs/errors/symptom-index.md](docs/errors/symptom-index.md); for an exception or warning — including a plain `NameError` on a name used after a block — look its message up in [docs/errors/index.md](docs/errors/index.md).

## Contributing

- **Base PRs on the `dev` branch, not `main`.**
- Conventions, the test-to-run table and the commit format: [docs/developing/contributing.md](docs/developing/contributing.md). Code style: [STYLE.md](STYLE.md). Human-facing setup and the docs standard: [CONTRIBUTING.md](CONTRIBUTING.md).
- Docs change in the same PR as the code that invalidates them.

## By task

| I want to... | Read |
|---|---|
| Read or modify activations on one forward | [usage/trace.md](docs/usage/trace.md), [usage/access-and-modify.md](docs/usage/access-and-modify.md), [usage/save.md](docs/usage/save.md) |
| Generate tokens, or run the task pipeline | [usage/generate.md](docs/usage/generate.md), [usage/pipe.md](docs/usage/pipe.md) |
| Intervene at specific generation steps | [usage/iteration.md](docs/usage/iteration.md) |
| Run several prompts at once, or share values between them | [usage/invoke-and-batching.md](docs/usage/invoke-and-batching.md), [usage/barrier.md](docs/usage/barrier.md) |
| Bundle several traces / share values across them | [usage/session.md](docs/usage/session.md) |
| Reach operations inside a module's forward | [usage/source.md](docs/usage/source.md) |
| Cache activations from many modules | [usage/cache.md](docs/usage/cache.md) |
| Take gradients | [usage/backward-and-grad.md](docs/usage/backward-and-grad.md) |
| Check shapes without running the model | [usage/scan.md](docs/usage/scan.md) |
| Make persistent edits; skip a module; stop early | [usage/edit.md](docs/usage/edit.md), [usage/skip.md](docs/usage/skip.md), [usage/stop-and-early-exit.md](docs/usage/stop-and-early-exit.md) |
| Pick a model class (HF, any `nn.Module`, diffusers, vLLM) | [models/index.md](docs/models/index.md) |
| Fit a model too big for one GPU | [models/tensor-parallel.md](docs/models/tensor-parallel.md), [models/quantization.md](docs/models/quantization.md) |
| Run on NDIF | [remote/index.md](docs/remote/index.md), [remote/remote-trace.md](docs/remote/remote-trace.md) |
| Ablate or sweep attention heads | [patterns/per-head-attention.md](docs/patterns/per-head-attention.md), [patterns/ablation.md](docs/patterns/ablation.md) |
| Sweep many prompts (local or NDIF) | [patterns/remote-dataset-sweep.md](docs/patterns/remote-dataset-sweep.md) |
| Run a research technique (logit lens, patching, ablation, steering, probing, SAEs) | [patterns/index.md](docs/patterns/index.md) |
| Port code written for nnsight before 0.8 | [reference/migrating-to-0.8.md](docs/reference/migrating-to-0.8.md) |
| Look up an exact API name or a `CONFIG` setting | [reference/api-quick-reference.md](docs/reference/api-quick-reference.md), [reference/config.md](docs/reference/config.md) |
| Extend nnsight (custom model, served value, runtime, backend) | [developing/extending.md](docs/developing/extending.md), [developing/extending-envoy.md](docs/developing/extending-envoy.md), [developing/index.md](docs/developing/index.md) |
| Understand why it behaves this way | [concepts/index.md](docs/concepts/index.md) |

## Before writing nnsight code

- **Nothing a trace body assigns survives the block unless it is saved, and a save must be inside the block and bound to a name** (`logits = model.output.logits.save()`). See [usage/save.md](docs/usage/save.md).
- **A module's `.output` is the real object it returns** — a plain tensor for a GPT-2 block, a tuple for attention. Edit a tuple's tensors in place or rebuild the tuple. See [usage/access-and-modify.md](docs/usage/access-and-modify.md).
- **Read modules in forward order within an invoke**, a submodule before the block containing it; out of order raises `OutOfOrderError`. See [errors/out-of-order-error.md](docs/errors/out-of-order-error.md).
- **A `tracer.iter` loop must not ask for more steps than the run makes**; it is cut short with a warning and the code after it is dropped. Check `len()` of what you collect. See [usage/iteration.md](docs/usage/iteration.md).
- **`generate` returns token ids and samples if the checkpoint's `generation_config` says so** — pass `do_sample=False` for determinism; `pipe` returns decoded records. See [usage/generate.md](docs/usage/generate.md).
- **Prefer `TransformersModel` / `NNsight`, and choose one device with `device=`, not `device_map=`.** See [models/transformers-model.md](docs/models/transformers-model.md#choosing-a-device).
