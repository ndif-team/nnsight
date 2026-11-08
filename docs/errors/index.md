---
title: Errors Index
one_liner: Every exception and warning nnsight users hit, keyed by its message text, with the page that explains cause and fix.
tags: [error, index]
related: [docs/index.md, docs/errors/symptom-index.md, docs/errors/debug-mode.md]
sources: [src/nnsight/tracing/tracer.py, src/nnsight/intervention/interleaver.py, src/nnsight/intervention/tracer.py, src/nnsight/intervention/batching.py, src/nnsight/intervention/envoy.py, src/nnsight/intervention/source.py]
---

# Errors Index

Up: [docs/index.md](../index.md) — all sections.

Find the message you saw. If nothing raised and the result is wrong or missing, or
the error is plain Python rather than an nnsight type and is not listed below, use
[symptom-index.md](symptom-index.md) instead.

Tracebacks from a trace block are cleaned so the top frames are your own code; the
exception type is preserved, so `except ValueError:` still works.
[debug-mode.md](debug-mode.md) covers what `CONFIG.APP.DEBUG` puts back.

## Execution order

| Exception | Message | Doc |
|---|---|---|
| `OutOfOrderError` | ``'<location>.i0' was requested but the model already ran past it`` — asked in the wrong order | [out-of-order-error.md](out-of-order-error.md) |
| `OutOfOrderError` | the same message, for a location the model never reached (module didn't fire, skipped parent) | [module-never-fired.md](module-never-fired.md) |
| `OutOfOrderError` | ``'<id>.grad.i0' was requested …`` — a numeric location is a `.grad` | [backward-and-grad.md](../usage/backward-and-grad.md) |
| `UserWarning` | ``'<location>' was never reached: the loop asked for a step the run did not make, so it was cut short …`` | [iteration.md](../usage/iteration.md#the-one-rule) |

The first two rows share one message. If the same invoke read a later location
before this one, it was asked in the wrong order; if nothing later was read, the
module never fired.

## Plain Python errors

| Exception | Message | Doc |
|---|---|---|
| `NameError` / `UnboundLocalError` | ``name 'X' is not defined`` (or ``cannot access local variable 'X' …``) on a name used **after** the trace block — it was not saved, so it never bound back | [save.md](../usage/save.md) |
| `NameError` | the same, **inside** the trace body, on a name an invoke saved — code between invokes runs before the invokes do | [invoke-and-batching.md](../usage/invoke-and-batching.md#code-between-invokes-runs-first) |

## Context and setup

| Exception | Message | Doc |
|---|---|---|
| `ValueError` | ``Cannot access `<location>` outside of interleaving`` | [cannot-access-outside-interleaving.md](cannot-access-outside-interleaving.md) |
| `ValueError` | ``save() was called outside a trace. …`` | [save-outside-trace.md](save-outside-trace.md) |
| `ValueError` | ``trace() needs an input, or at least one `with tracer.invoke(...)` block`` | [cannot-access-outside-interleaving.md](cannot-access-outside-interleaving.md) |
| `ValueError` | ``Cannot invoke while the model is already running.`` | [invoke-during-execution.md](invoke-during-execution.md) |
| `NotImplementedError` | ``<ModelClass> does not support batching multiple invokes`` | [batching-not-implemented.md](batching-not-implemented.md) |

## Capturing the block

| Exception | Message | Doc |
|---|---|---|
| `WithBlockNotFoundError` | ``nnsight found no `with` statement at <file>:<line> …`` / ``nnsight has no source for <file> …`` | [with-block-not-found.md](with-block-not-found.md) |
| `ValueError` | ``A traced `with` block cannot start with `try:` …`` | [with-block-not-found.md](with-block-not-found.md#a-trace-body-that-starts-with-try) |
| `ValueError` | ``The body of a traced `with` must start on its own line …`` | [with-block-not-found.md](with-block-not-found.md#a-body-on-the-with-line) |
| `SyntaxError` | ``'return' outside function`` | [with-block-not-found.md](with-block-not-found.md#return-inside-a-trace-body) |

## Values and features

| Exception | Message | Doc |
|---|---|---|
| `TypeError` | ``'tuple' object does not support item assignment`` — writing `module.output[0] = x` on a tuple output | [access-and-modify.md](../usage/access-and-modify.md#tuple-outputs) |
| `NotImplementedError` | ``This tensor does not require grad, so a backward session cannot produce gradients …`` — inside `generate`, or a frozen model | [backward-and-grad.md](../usage/backward-and-grad.md#limits) |
| `RuntimeError` | ``cannot register a hook on a tensor that doesn't require gradient`` | [backward-and-grad.md](../usage/backward-and-grad.md#limits) |
| `SourceNotAvailable` | ``'<op>' calls a submodule; call `.source` on that submodule directly …``, ``recursive `.source` is only available inside a trace``, or no Python source | [source.md](../usage/source.md#when-source-isnt-available) |
| `GuardOnDataDependentSymNode` | ``Could not guard on data-dependent expression …`` — branching on, or converting, a fake value under `scan` | [scan.md](../usage/scan.md#what-a-scan-cannot-tell-you) |
| `ValueError` | ``A barrier was never reached by every block it waits for; check the count it was created with`` | [barrier.md](../usage/barrier.md) |
| `ValueError` | ``A batched `.skip()` has to cover every row …`` | [skip.md](../usage/skip.md) |
| `IndexError` | ``kernel site index <i> is out of range for an axis of size <n>``; ``kernel site indices <tuple> are out of range …`` | [kernel.md](../usage/kernel.md#sites) |
| `TypeError` | ``kernel site index <name>=<value> is not an int, slice, list or tensor``; ``… is a bool; pass an int, slice, list or tensor``; ``… is a <dtype> tensor; pass integer indices`` | [kernel.md](../usage/kernel.md#sites) |
| `ValueError` | ``this kernel has no site '<site>'; available: lse, scores, probs`` (or ``chunk_states, states``, ``expert_input, expert_activation, expert_output``); ``kernel site '<site>' takes heads, query…`` / ``takes chunks, heads`` / ``takes tokens, heads`` / ``takes tokens, slot, expert`` | [kernel.md](../usage/kernel.md#sites) |
| `ValueError` | raised when a kernel call's sites are unavailable: ``kernel sites do not cover fla's packed (cu_seqlens) layout``; ``<kernel> exports the final state only; …``; ``<kernel> was called with use_gate_in_kernel …``; ``<kernel> ran chunk_gated_delta_rule_fwd_h <n> times for this call; the fla backend that ran keeps its chunk states inside its own kernel``; ``<kernel> produced <n> chunk states for <c> chunks …`` / ``bound the recurrent state <n> times …`` (when the kernel's forward changed shape); ``<kernel> exports the state after its token only; tokens=… inside it cannot be derived``; ``<kernel> bound <name> <n> times with shapes …`` / ``bound no inverse permutation over its <n> pairs`` / ``bound an inverse permutation that does not sort its <n> pairs by expert``; ``<kernel> ran under tensor or expert parallelism; …``; ``<kernel> applies a post-expert norm; …``; ``'<op>' is drilled with .source in this trace; its kernel sites are not served on that call`` | [kernel.md](../usage/kernel.md#limits) |
| `TypeError` / `IndexError` / `ValueError` | ``kernel site '<site>' takes one expert index; got …``; ``kernel site expert index <e> is out of range for <n> experts``; ``kernel site '<site>' takes slot= or expert=, not both``; ``kernel site '<site>': <n> token rows do not divide into the trace's <b> batch rows, …`` | [kernel.md](../usage/kernel.md#per-expert-rows-of-a-grouped-mixture-of-experts-kernel) |
| `OutOfOrderError` | ``'<path>.kernel' was requested while '<path>' is already running without an instrumented forward …``; ``'<path>.kernel.i0' was requested but the model already ran past it; a `.kernel` site is served only when the module's forward reaches a registered fused kernel`` | [kernel.md](../usage/kernel.md#ordering) |
| `UserWarning` | ``kernel site '<site>' materializes <x> GiB in <dtype>; pass heads=, query= or key= to request a slice`` (``pass tokens=, heads=`` on `states`; ``pass tokens=, slot= or expert=`` on MoE sites) | [kernel.md](../usage/kernel.md#sites) |
| `AttributeError` | ``module 'nnsight' has no attribute 'list'`` (or `apply`, `session`, …); ``'Tensor' object has no attribute 'value'`` | [migrating-to-0.8.md](../reference/migrating-to-0.8.md) |
| `NNsightDeprecationWarning` | ``<name> is deprecated; use <replacement> instead.`` | [migrating-to-0.8.md](../reference/migrating-to-0.8.md#deprecated--warn-under-nnsightdeprecationwarning) |

## Remote

`RemoteBackend` raises `RemoteError` on a failed submission or a server-side `ERROR`
status. A worker error from a deferring driver (vLLM) is re-raised at the client as
a `RuntimeError` carrying the original type, message and traceback — see
[debug-mode.md](debug-mode.md#errors-from-a-deferring-driver). For failures that
only happen remotely, see [remote-trace.md](../remote/remote-trace.md#gotchas) and
[env-comparison.md](../remote/env-comparison.md).
