---
title: Developing NNsight
one_liner: Internals reference for contributors and agents extending nnsight.
tags: [internals, dev]
related: [docs/developing/architecture-overview.md, docs/developing/tracing-pipeline.md, docs/developing/interleaver-internals.md]
sources: []
---

# Developing NNsight

This folder is the internals reference for the `refactor/transform` branch of nnsight. It is intentionally one level below the user-facing `docs/usage/` and `docs/concepts/` folders: docs here cite source `file:line`, describe data flow between subsystems, and explain extension points for custom backends, runtimes, and envoys.

Audience: nnsight contributors and AI agents whose users want to debug or extend the library. If you are looking for "how do I run a trace," start in `docs/usage/`.

## What this covers

The internal architecture is organized into layered subsystems:

- **Tracer** — captures the body of a `with` block, parses it via AST, and compiles it into a callable function. Lives under `src/nnsight/intervention/tracing/`.
- **Backend** — compiles the function source to a code object, executes it, and routes results. Lives under `src/nnsight/intervention/backends/`.
- **Interleaver / Mediator** — coordinates the model's forward pass with one worker thread per invoke; routes value/swap/skip/barrier events between threads via PyTorch hooks. Lives in `src/nnsight/intervention/interleaver.py`.
- **Hook system** — lazy, one-shot PyTorch hooks installed on demand by mediators (`src/nnsight/intervention/hooks.py`).
- **Source accessor** — AST-based forward injection for in-module operation tracing (`src/nnsight/intervention/source.py`).
- **Envoy / eproperty** — user-facing proxy and the descriptor protocol that ties everything together (`src/nnsight/intervention/envoy.py`, `interleaver.py`).
- **Batching, serialization, runtimes** — multi-invoke batching, dill-based serialization for remote execution, and pluggable model runtimes such as vLLM.

## Table of contents

### Big picture

- `docs/developing/architecture-overview.md` — top-down "how everything fits": Tracer to Backend to Interleaver to Mediator to PyTorch hooks.
- `docs/developing/tracing-pipeline.md` — capture, parse, compile, execute. The `Tracer.Info` lifecycle and the cache key.

### Interleaver

- `docs/developing/interleaver-internals.md` — `Interleaver`, `Mediator`, the event loop, `handle()` fan-out, `iterate_requester()`.
- `docs/developing/lazy-hook-system.md` — one-shot input/output hooks, `add_ordered_hook`, the sentinel hook, persistent cache and iter hooks.
- `docs/developing/source-accessor-internals.md` — `SourceAccessor`, `OperationAccessor`, `FunctionCallWrapper`, recursive `.source` and `rebind`.

### Extension API

- `docs/developing/eproperty-deep-dive.md` — the `eproperty` descriptor as the formal extension API. `IEnvoy`, the stub idiom, `preprocess`/`postprocess`/`transform`, `provide`.
- `docs/developing/adding-a-new-backend.md` — subclassing `Backend` for custom execution strategies (remote, simulated, async).
- `docs/developing/adding-a-new-runtime.md` — wrapping a non-PyTorch runtime (vLLM-style) with a custom model class, batcher, and provider eproperties.

### Other

- `docs/developing/batching-internals.md` — `Batchable`, `Batcher`, batch groups, narrow/swap, multi-invoke semantics.
- `docs/developing/serialization.md` — dill-based pickling for remote execution; `__getstate__` / `__setstate__` on tracers, mediators, and envoys.
- `docs/developing/backends.md` — the existing backends (`ExecutionBackend`, `EditingBackend`, `RemoteBackend`, `LocalSimulationBackend`).
- `docs/developing/vllm-integration.md` — `VLLM`, `NNsightGPUModelRunner`, `VLLMBatcher`, and how vLLM's scheduler ordering is reconciled with mediator batch groups.
- `docs/developing/testing.md` — running the test suite, conda env, key test files, smoke vs full validation.
- `docs/developing/performance.md` — where overhead lives, the trace cache, `PYMOUNT`, and how to profile.
- `docs/developing/agent-evals.md` — running agent-driven evaluations against the docs and code.
- `docs/developing/contributing.md` — branch policy, commit style, and PR checklist.

## Related

- `NNsight.md` (repo root) — long-form design document. Most internals docs in this folder summarize and update sections of `NNsight.md` for the `refactor/transform` branch.
- `0.6.0.md` (repo root) — release notes for v0.6.0; describes the user-visible surface of the lazy-hook and eproperty refactor.
- `CLAUDE.md` (repo root) — project-level instructions, including the development & testing notes that this folder expands on.
