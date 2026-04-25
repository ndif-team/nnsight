# nnsight — Agent Guide

This file routes you to the right documentation under `docs/` for whatever the user is asking about. The actual content lives in `docs/`. **Read the relevant doc page before writing code** — these docs are tight, recipe-style, and frequently updated to match the current branch.

If you're new to nnsight, read [docs/concepts/index.md](docs/concepts/index.md) once. Otherwise jump directly to the doc that matches the user's task.

---

## How to use this file

1. Find the user's intent in **"By task"** below and follow the link.
2. If the user's request maps to a model class (LanguageModel, VLLM, etc.), check **"By model class"**.
3. If something is broken, check **"Errors"** or **"Gotchas"**.
4. The **"Inline gotcha cheat-sheet"** at the bottom catches the most common agent mistakes — internalize them before writing any nnsight code.

---

## By task

### "I want to read activations / modify them on a single forward pass"
- [docs/usage/trace.md](docs/usage/trace.md) — `model.trace(input)`
- [docs/usage/access-and-modify.md](docs/usage/access-and-modify.md) — `.output`, `.input`, `.inputs`, in-place vs replacement
- [docs/usage/save.md](docs/usage/save.md) — keep values past the trace exit

### "I want multi-token / autoregressive generation"
- [docs/usage/generate.md](docs/usage/generate.md) — `model.generate(input, max_new_tokens=N)`
- [docs/usage/iter.md](docs/usage/iter.md) — step-by-step control with `tracer.iter[...]`
- [docs/usage/all-and-next.md](docs/usage/all-and-next.md) — `tracer.all()`, `module.next()`

### "I want to run multiple prompts at once"
- [docs/usage/invoke-and-batching.md](docs/usage/invoke-and-batching.md) — `tracer.invoke(...)`, batched lists, empty invokes
- [docs/usage/barrier.md](docs/usage/barrier.md) — `tracer.barrier(n)` for cross-invoke value sharing
- [docs/patterns/multi-prompt-comparison.md](docs/patterns/multi-prompt-comparison.md)

### "I want to look inside a module's forward (intermediate operations)"
- [docs/usage/source.md](docs/usage/source.md) — `model.<path>.source.<op_name>.output / .input`
- [docs/concepts/source-tracing.md](docs/concepts/source-tracing.md) — how `.source` rewrites the AST

### "I want to cache activations from many modules"
- [docs/usage/cache.md](docs/usage/cache.md) — `tracer.cache(modules=..., include_inputs=...)`

### "I need gradients / backward pass"
- [docs/usage/backward-and-grad.md](docs/usage/backward-and-grad.md) — `with tensor.backward():` is its own session
- [docs/patterns/gradient-based-attribution.md](docs/patterns/gradient-based-attribution.md)
- [docs/patterns/attribution-patching.md](docs/patterns/attribution-patching.md)

### "I want to run remotely on NDIF"
- [docs/remote/ndif-overview.md](docs/remote/ndif-overview.md) — what NDIF is, job lifecycle
- [docs/remote/api-key-and-config.md](docs/remote/api-key-and-config.md) — set up your API key
- [docs/remote/remote-trace.md](docs/remote/remote-trace.md) — `model.trace(..., remote=True)`
- [docs/remote/remote-session.md](docs/remote/remote-session.md) — bundle multiple traces into one job
- [docs/remote/non-blocking-jobs.md](docs/remote/non-blocking-jobs.md) — submit and poll
- [docs/remote/register-local-modules.md](docs/remote/register-local-modules.md) — ship local code to NDIF

### "I want to verify shapes / inspect dimensions without running the model"
- [docs/usage/scan.md](docs/usage/scan.md) — `model.scan(...)`
- [docs/gotchas/save.md](docs/gotchas/save.md) (`.save()` still required inside scan)

### "I want to make persistent edits to a model"
- [docs/usage/edit.md](docs/usage/edit.md) — `model.edit()` / `model.edit(inplace=True)`

### "I want to run a research pattern (logit lens / patching / steering / SAE...)"
- [docs/patterns/index.md](docs/patterns/index.md) — full cookbook
- Most-asked-for: [logit-lens](docs/patterns/logit-lens.md), [activation-patching](docs/patterns/activation-patching.md), [ablation](docs/patterns/ablation.md), [steering](docs/patterns/steering.md), [attention-patterns](docs/patterns/attention-patterns.md), [sae-and-auxiliary-modules](docs/patterns/sae-and-auxiliary-modules.md), [per-head-attention](docs/patterns/per-head-attention.md)

### "I'm extending nnsight (custom Envoy / new runtime / new value type)"
- [docs/usage/extending.md](docs/usage/extending.md) — `envoys=` kwarg + custom `eproperty`
- [docs/concepts/envoy-and-eproperty.md](docs/concepts/envoy-and-eproperty.md) — mental model
- [docs/developing/eproperty-deep-dive.md](docs/developing/eproperty-deep-dive.md) — full extension API

### "Something is broken / I got an error"
- [docs/errors/index.md](docs/errors/index.md) — exception → cause → fix table
- [docs/gotchas/index.md](docs/gotchas/index.md) — most common ways things go wrong
- [docs/errors/debug-mode.md](docs/errors/debug-mode.md) — turn on full tracebacks

---

## By model class

| Need | Use | Doc |
|---|---|---|
| Any `torch.nn.Module` | `NNsight(module)` | [docs/models/nnsight-base.md](docs/models/nnsight-base.md) |
| HuggingFace causal LM (text) | `LanguageModel("repo/id", ...)` | [docs/models/language-model.md](docs/models/language-model.md) |
| Vision-language models (LLaVA, Qwen2-VL, ...) | `VisionLanguageModel("repo/id", ...)` | [docs/models/vision-language-model.md](docs/models/vision-language-model.md) |
| Diffusion pipelines | `DiffusionModel("repo/id", ...)` | [docs/models/diffusion-model.md](docs/models/diffusion-model.md) |
| High-throughput / production / TP | `VLLM("repo/id", mode="sync"\|"async", ...)` | [docs/models/vllm.md](docs/models/vllm.md) |

Decision tree at [docs/models/index.md](docs/models/index.md).

---

## Concepts (mental models)

Read at least the first two if the user is asking "why is my code blocking / out of order / not seeing values":

- [docs/concepts/deferred-execution.md](docs/concepts/deferred-execution.md) — code is captured, compiled, run in a worker thread; `.output` blocks until the model fires
- [docs/concepts/threading-and-mediators.md](docs/concepts/threading-and-mediators.md) — each invoke = a `Mediator` (worker thread); event protocol VALUE/SWAP/SKIP/BARRIER/END/EXCEPTION
- [docs/concepts/interleaver-and-hooks.md](docs/concepts/interleaver-and-hooks.md) — lazy one-shot hooks (NEW in this branch)
- [docs/concepts/envoy-and-eproperty.md](docs/concepts/envoy-and-eproperty.md) — `Envoy` wraps a module; `eproperty` is the descriptor for hookable values
- [docs/concepts/batching-and-invokers.md](docs/concepts/batching-and-invokers.md) — invoke threads, empty invokes, when you need a barrier
- [docs/concepts/source-tracing.md](docs/concepts/source-tracing.md) — how `.source` rewrites a module's forward AST

---

## Reference

- [docs/reference/api-quick-reference.md](docs/reference/api-quick-reference.md) — every public method/property in one table
- [docs/reference/config.md](docs/reference/config.md) — every `CONFIG.*` setting
- [docs/reference/glossary.md](docs/reference/glossary.md) — Mediator, Invoker, Tracer, Envoy, eproperty, etc.
- [docs/reference/external-resources.md](docs/reference/external-resources.md) — nnsight.net, NDIF service, Discord, paper, walkthrough notebook
- [docs/reference/version-history.md](docs/reference/version-history.md) — release notes index

---

## For developers / contributors (the 15%)

- [docs/developing/index.md](docs/developing/index.md) — top of the developer tree
- [docs/developing/architecture-overview.md](docs/developing/architecture-overview.md) — how everything fits
- [docs/developing/contributing.md](docs/developing/contributing.md) — branch conventions, PR workflow, validation suite
- [docs/developing/testing.md](docs/developing/testing.md) — how to run tests
- [docs/developing/lazy-hook-system.md](docs/developing/lazy-hook-system.md) — the new lazy-hook architecture
- [docs/developing/eproperty-deep-dive.md](docs/developing/eproperty-deep-dive.md) — extension API in depth
- [docs/developing/source-accessor-internals.md](docs/developing/source-accessor-internals.md) — `.source` machinery
- [docs/developing/serialization.md](docs/developing/serialization.md) — source-based pickling for remote
- [docs/developing/vllm-integration.md](docs/developing/vllm-integration.md) — vLLM internals
- [docs/developing/agent-evals.md](docs/developing/agent-evals.md) — how the team measures agent ability with nnsight

Long-form architecture reference: [NNsight.md](NNsight.md) (kept for human reading; developing/ docs supplement it with up-to-date branch info).

---

## Inline gotcha cheat-sheet

Internalize these BEFORE writing nnsight code. Each links to the full doc.

| Rule | One-liner |
|---|---|
| Always `.save()` what you want past the trace | Values are filtered out on trace exit unless `.save()` (or `nnsight.save(x)`) was called. → [save](docs/gotchas/save.md) |
| Module access order matters within an invoke | Access modules in **forward-pass order** within a single invoke or you'll deadlock. To access "out of order" use a separate invoke. → [order-and-deadlocks](docs/gotchas/order-and-deadlocks.md) |
| `tracer.iter[:]` swallows trailing code | Code after `for step in tracer.iter[:]:` never executes. Use a separate empty invoke or bounded `iter[:N]`. → [iteration](docs/gotchas/iteration.md) |
| `model.trace()` needs input or invokes | Bare `model.trace()` with neither errors. Provide input to `.trace(...)` or use `tracer.invoke(...)`. → [order-and-deadlocks](docs/gotchas/order-and-deadlocks.md) |
| Tuple outputs need careful handling | Many transformer blocks return tuples — use `[0]` indexing for in-place or replace the whole tuple. → [modification](docs/gotchas/modification.md) |
| In-place `[:] =` ≠ replacement `=` | They have different semantics — pick deliberately. → [modification](docs/gotchas/modification.md) |
| Clone before in-place modify if you need the "before" | A saved variable is a reference; in-place modification mutates the saved value too. → [modification](docs/gotchas/modification.md) |
| Cross-invoke same-module access requires a barrier | Two invokes both touching the same module need `tracer.barrier(2)` to share values. → [cross-invoke](docs/gotchas/cross-invoke.md) |
| Trace-time values are real tensors | Inside a trace, `.output` returns the actual tensor (not a proxy). `.shape`, ops, `print()` all work normally. → [types-and-values](docs/gotchas/types-and-values.md) |
| `.scan(...)` still requires `.save()` | Scan is a tracing context too — same exit-filter rules. → [save](docs/gotchas/save.md) |
| Backward is a separate session | Get `.output` BEFORE `with tensor.backward():`; access `.grad` (on tensors, not modules) ONLY inside it. → [backward](docs/gotchas/backward.md) |
| Remote: `.save()` is the transmission mechanism | Local lists `.append()`-ed outside a remote trace stay empty; create the list inside the trace. → [remote](docs/gotchas/remote.md) |
| `LanguageModel` on a pre-loaded HF model needs `tokenizer=` | Otherwise you get `AttributeError: Tokenizer not found`. → [tokenizer-not-found](docs/errors/tokenizer-not-found.md) |
| Don't call `.source` on a module from inside another `.source` | Access the submodule directly: `model.transformer.h[0].attn.some_submodule.source`. → [integrations](docs/gotchas/integrations.md) |

Full inventory at [docs/gotchas/index.md](docs/gotchas/index.md).

---

## Skills system / Claude Code

If the user asks how to use nnsight in Claude Code or via Skills, point them to the [README.md](README.md) section on agent setup. The Skills marketplace and Context7 integration are documented there. nnsight ships with **DeepWiki** integration for AI-assisted Q&A — see the README.

---

## When in doubt

- The user's task isn't covered by anything above? Search the folder tree: `docs/usage/`, `docs/patterns/`, `docs/remote/`, `docs/models/`, `docs/concepts/`. Each folder has an `index.md` with a one-liner per doc.
- Source of truth is the source code at `src/nnsight/`. Docs cite file:line for every behavioral claim — verify against source if a doc and source disagree.
- Long-form architecture is in [NNsight.md](NNsight.md). Release notes for the current major version are in [0.6.0.md](0.6.0.md).
- External: official docs at https://nnsight.net, forum at https://discuss.ndif.us, GitHub at https://github.com/ndif-team/nnsight.

---

## Versions

These docs target nnsight 0.6+ on the `refactor/transform` branch (lazy hook execution, `eproperty` extension API, `SourceAccessor` cached on module, `Mediator` hook-tracking). Do **not** suggest deprecated v0.4 patterns: `nnsight.cond`, `nnsight.list`, `session.iter`, `nnsight.local`, the `nnsight.trace` decorator. Standard Python `if`/`for` work inside trace contexts since v0.5.

---

## Open questions log

Writer agents that produced these docs flagged judgment calls / unresolved questions in [docs/questions.md](docs/questions.md). Read it if you're confused about a behavior — your confusion may already be on the list.
