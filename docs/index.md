---
title: nnsight Docs
one_liner: The router for docs/ — which section answers which question, and the pages most tasks start from.
tags: [index]
---

# nnsight Docs

Find the section that answers your question, then the one page inside it. Read that page before writing code.

## Sections

- [usage/](usage/index.md) — How do I use a feature? One page per API: trace, generate, save, invoke, iteration, source, cache, backward, ...
- [patterns/](patterns/index.md) — How do I run a research technique? Logit lens, patching, ablation, steering, attribution, probing, SAEs.
- [models/](models/index.md) — Which model class do I use, and how does it behave? `TransformersModel`, `NNsight`, `DiffusionModel`, `VLLM`, tensor parallelism, quantization.
- [remote/](remote/index.md) — How do I run on NDIF? API key, `remote=True`, sessions, non-blocking jobs, shipping local code.
- [errors/](errors/index.md) — Something broke. The index is keyed by the exception or warning text; [symptom-index.md](errors/symptom-index.md) covers wrong results that raise nothing.
- [concepts/](concepts/index.md) — Why does it behave this way? Mental models: deferred execution, workers, the interleaver, envoys, batching, `.source`.
- [reference/](reference/index.md) — What is the exact name? API tables, `CONFIG`, glossary, migrating from pre-0.8 code.
- [developing/](developing/index.md) — How is nnsight built, and how do I extend or contribute to it? Internals, extension points, testing, conventions.

## Most-used pages

- [usage/trace.md](usage/trace.md) — `model.trace(input)`: one forward with interventions
- [usage/save.md](usage/save.md) — getting values out of a trace
- [usage/invoke-and-batching.md](usage/invoke-and-batching.md) — several prompts in one trace; sharing values between them
- [usage/generate.md](usage/generate.md) — multi-token generation; `generate` vs `pipe`
- [usage/iteration.md](usage/iteration.md) — `tracer.iter[...]` per-step interventions and their pitfalls
- [usage/source.md](usage/source.md) — operations inside a module's forward
- [remote/remote-trace.md](remote/remote-trace.md) — running a trace on NDIF
- [errors/symptom-index.md](errors/symptom-index.md) — something broke
- [reference/api-quick-reference.md](reference/api-quick-reference.md) — every public method and property
- [models/index.md](models/index.md) — picking a model class
