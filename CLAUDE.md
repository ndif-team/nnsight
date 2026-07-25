# nnsight — Agent Guide

This file routes you to the right documentation under `docs/` for whatever the user is asking about. The actual content lives in `docs/`. **Read the relevant doc page before writing code** — these docs are tight, recipe-style, and kept in sync with the current implementation.

If you're new to nnsight, read [docs/concepts/index.md](docs/concepts/index.md) once. Otherwise jump directly to the doc that matches the user's task.

---

## How to use this file

1. Find the user's intent in **"By task"** below and follow the link.
2. If the request maps to a model class (TransformersModel, VLLM, ...), check **"By model class"**.
3. If something is broken, check **"Errors"** or **"Gotchas"**.
4. The **"Inline gotcha cheat-sheet"** at the bottom catches the most common agent mistakes — internalize them before writing any nnsight code.

---

## By task

### "I want to read activations / modify them on a single forward pass"
- [docs/usage/trace.md](docs/usage/trace.md) — `model.trace(input)`
- [docs/usage/access-and-modify.md](docs/usage/access-and-modify.md) — `.output`, `.input`, `.inputs`, in-place vs replacement
- [docs/usage/save.md](docs/usage/save.md) — keep values past the trace exit (**now raises if called outside a trace**)

### "I want multi-token / autoregressive generation"
- [docs/usage/generate.md](docs/usage/generate.md) — `model.generate(input, max_new_tokens=N)` → **token ids** on `tracer.result`
- [docs/usage/pipe.md](docs/usage/pipe.md) — `model.pipe(input)` → the task pipeline's **records** (decoded text, labels)
- [docs/usage/iter-all-next.md](docs/usage/iter-all-next.md) — `for _ in tracer.iter[...]:`, `tracer.all()`

### "I want to run multiple prompts at once"
- [docs/usage/invoke-and-batching.md](docs/usage/invoke-and-batching.md) — `tracer.invoke(...)`, batched lists, empty invokes
- [docs/usage/barrier.md](docs/usage/barrier.md) — `tracer.barrier(n)` for cross-invoke value sharing
- [docs/patterns/multi-prompt-comparison.md](docs/patterns/multi-prompt-comparison.md)

### "I want to bundle several traces / share values between them"
- [docs/usage/session.md](docs/usage/session.md) — `with model.session():` (values flow across traces without `.save()`)

### "I want to look inside a module's forward (intermediate operations)"
- [docs/usage/source.md](docs/usage/source.md) — `model.<path>.source.<op_name>.output / .input`
- [docs/concepts/source-tracing.md](docs/concepts/source-tracing.md) — how `.source` rewrites the forward AST

### "I want to cache activations from many modules"
- [docs/usage/cache.md](docs/usage/cache.md) — `tracer.cache(modules=..., include_inputs=...)`

### "I need gradients / backward pass"
- [docs/usage/backward-and-grad.md](docs/usage/backward-and-grad.md) — `with tensor.backward():`
- [docs/patterns/gradient-based-attribution.md](docs/patterns/gradient-based-attribution.md)
- [docs/patterns/attribution-patching.md](docs/patterns/attribution-patching.md)

### "I want to run remotely on NDIF"
- [docs/remote/ndif-overview.md](docs/remote/ndif-overview.md) — what NDIF is, job lifecycle
- [docs/remote/api-key-and-config.md](docs/remote/api-key-and-config.md) — set up your API key
- [docs/remote/remote-trace.md](docs/remote/remote-trace.md) — `model.trace(..., remote=True)`
- [docs/remote/remote-session.md](docs/remote/remote-session.md) — bundle multiple traces into one job
- [docs/remote/non-blocking-jobs.md](docs/remote/non-blocking-jobs.md) — submit and poll
- [docs/remote/remote-async.md](docs/remote/remote-async.md) — `AsyncRemoteBackend`: `await` / `async for` a job
- [docs/remote/register-local-modules.md](docs/remote/register-local-modules.md) — ship local code to NDIF

### "I want to verify shapes / inspect dimensions without running the model"
- [docs/usage/scan.md](docs/usage/scan.md) — `model.scan(...)`

### "I want to make persistent edits to a model"
- [docs/usage/edit.md](docs/usage/edit.md) — `model.edit()` / `model.edit(inplace=True)`

### "I want to skip a module / stop early"
- [docs/usage/skip.md](docs/usage/skip.md) — `module.skip(replacement)`
- [docs/usage/stop-and-early-exit.md](docs/usage/stop-and-early-exit.md) — `tracer.stop()`

### "I want to run a research pattern (logit lens / patching / steering / SAE...)"
- [docs/patterns/index.md](docs/patterns/index.md) — full cookbook
- Most-asked-for: [logit-lens](docs/patterns/logit-lens.md), [activation-patching](docs/patterns/activation-patching.md), [ablation](docs/patterns/ablation.md), [steering](docs/patterns/steering.md), [attention-patterns](docs/patterns/attention-patterns.md), [sae-and-auxiliary-modules](docs/patterns/sae-and-auxiliary-modules.md), [per-head-attention](docs/patterns/per-head-attention.md)

### "I'm extending nnsight (custom model / runtime / value)"
- [docs/usage/extending.md](docs/usage/extending.md) — subclass `NNsight`/`Envoy`, `_batch_size`/`_batch`, attach modules
- [docs/concepts/envoy.md](docs/concepts/envoy.md) — mental model; `eproperty` is the descriptor behind hookable values (`.output`/`.input`/custom)
- [docs/developing/extending-envoy.md](docs/developing/extending-envoy.md) — custom hookable values via `interleaver.handle` + a property

### "Something is broken / I got an error"
- [docs/errors/index.md](docs/errors/index.md) — exception → cause → fix table
- [docs/gotchas/index.md](docs/gotchas/index.md) — most common ways things go wrong
- [docs/errors/debug-mode.md](docs/errors/debug-mode.md) — tracebacks and `CONFIG.APP.DEBUG`

---

## By model class

| Need | Use | Doc |
|---|---|---|
| Any `torch.nn.Module` | `NNsight(module)` | [docs/models/nnsight-base.md](docs/models/nnsight-base.md) |
| **HuggingFace model (text/vision/multimodal/audio)** | `TransformersModel("repo/id", task=...)` | [docs/models/transformers-model.md](docs/models/transformers-model.md) |
| Diffusion pipelines | `DiffusionModel("repo/id", ...)` | [docs/models/diffusion-model.md](docs/models/diffusion-model.md) |
| High-throughput / production / TP | `VLLM("repo/id", mode="sync"\|"async")` | [docs/models/vllm.md](docs/models/vllm.md) |
| Causal LM (**deprecated** alias) | `LanguageModel(...)` → use `TransformersModel(task="text-generation")` | [docs/models/language-model.md](docs/models/language-model.md) |
| Vision-language (**deprecated** alias) | `VisionLanguageModel(...)` → use `TransformersModel(task="image-text-to-text")` | [docs/models/vision-language-model.md](docs/models/vision-language-model.md) |

`TransformersModel` is the primary HuggingFace class. `LanguageModel`/`VisionLanguageModel` still work but **warn on construction**. Decision tree at [docs/models/index.md](docs/models/index.md).

---

## Concepts (mental models)

Read at least the first two if the user is asking "why is my code blocking / out of order / not seeing values":

- [docs/concepts/deferred-execution.md](docs/concepts/deferred-execution.md) — the block is captured, compiled, and run interleaved with the model; `.output` blocks until the model fires
- [docs/concepts/threading-and-mediators.md](docs/concepts/threading-and-mediators.md) — each invoke is a `Mediator` running in a **greenlet** (not a thread); event protocol VALUE/SWAP/SKIP/BARRIER
- [docs/concepts/interleaver-and-hooks.md](docs/concepts/interleaver-and-hooks.md) — one shared `Interleaver` installs pass-through forward hooks on every module
- [docs/concepts/envoy.md](docs/concepts/envoy.md) — `Envoy` wraps a module; `.input`/`.output` are eproperties, `.source` returns a `Source`
- [docs/concepts/batching-and-invokers.md](docs/concepts/batching-and-invokers.md) — invokes, empty invokes, batch groups, when you need a barrier
- [docs/concepts/source-tracing.md](docs/concepts/source-tracing.md) — how `.source` rewrites a module's forward AST

---

## Reference

- [docs/reference/api-quick-reference.md](docs/reference/api-quick-reference.md) — every public method/property in one table
- [docs/reference/config.md](docs/reference/config.md) — every `CONFIG.*` setting
- [docs/reference/glossary.md](docs/reference/glossary.md) — Mediator, Invoker, Tracer, Envoy, Interleaver, Batcher, greenlet, source, ...
- [docs/reference/external-resources.md](docs/reference/external-resources.md) — nnsight.net, NDIF, Discord, paper
- [docs/reference/version-history.md](docs/reference/version-history.md) — the pipeline rewrite, old→new deltas

---

## For developers / contributors

- **Base PRs on the `dev` branch, not `main`.**
- [docs/developing/index.md](docs/developing/index.md) — top of the developer tree
- [docs/developing/architecture-overview.md](docs/developing/architecture-overview.md) — how everything fits (Tracer → Backend → Interleaver → Mediator → hooks → Envoy)
- [docs/developing/tracing-pipeline.md](docs/developing/tracing-pipeline.md) — capture → parse → build → compile → execute
- [docs/developing/interleaver-internals.md](docs/developing/interleaver-internals.md) — greenlets, mediators, the event protocol
- [docs/developing/hook-system.md](docs/developing/hook-system.md) — the per-module forward-hook design
- [docs/developing/serialization.md](docs/developing/serialization.md) — source-based block reduction for remote
- [docs/developing/source-internals.md](docs/developing/source-internals.md) — `.source` AST instrumentation
- [docs/developing/vllm-integration.md](docs/developing/vllm-integration.md) — the vLLM runtime
- [docs/developing/testing.md](docs/developing/testing.md) — how to run the tests
- [docs/developing/contributing.md](docs/developing/contributing.md) — conventions

---

## Inline gotcha cheat-sheet (read before writing nnsight code)

- **`.save()` is required to keep a value past the trace, and now *raises* if called outside a trace.** `x = nnsight.save([])` *before* a `with model.trace(...)` block is an error — put the save inside.
- **`.save()` returns the value by its *variable name* — you must bind it.** `logits = model.logits.save()` works; a bare `model.logits.save()` marks the value but leaves no name to return it under, so it silently never appears in the results. This bites hardest on vLLM/remote/serve, where you read `output.saves["logits"]` (or the pushed-back local) by name — an unbound save just isn't there.
- **A module's `.output` is the real object it returns.** For a GPT-2 *block* that's a plain `Tensor (batch, seq, hidden)` — read/write the whole tensor, no `[0]`. An *attention* submodule returns a tuple; check `print(module.source)` or the shape rather than assuming.
- **Assign a modified tensor back, don't rely on in-place edits into a tuple-element view across a barrier** (that can segfault) — mutate a copy and set `.output`.
- **Execution is deferred and interleaved (greenlets).** Reading a location the model already ran past raises `OutOfOrderError`; reading `.output`/`.input` *outside* a trace raises "Cannot access `...` outside of interleaving". Capture forward tensors before a `backward()` block.
- **Unbounded `iter[:]` / `all()` drop everything after the loop.** To keep per-step values *and* a final result, use a bounded `iter[:N]`. `tracer.next()` no longer exists.
- **A value produced inside one `invoke` is not visible in another** without `tracer.barrier(n)`.
- **`generate` returns token ids** (`tracer.result`), greedy by default; use **`pipe`** for the pipeline's decoded records.
- **Prefer `TransformersModel` / `NNsight`.** `LanguageModel` / `VisionLanguageModel` are deprecated aliases that warn.
- **`eproperty` exists** — the descriptor behind `.output`/`.input`; define your own hookable values with it on a model subclass (see [docs/concepts/envoy.md](docs/concepts/envoy.md)).

---

*See `questions.md` at the repo root for open documentation questions and the decisions taken while porting these docs.*
