---
title: Errors Index
one_liner: Map of the exceptions nnsight raises to their cause-and-fix docs.
tags: [error, index]
related: [docs/concepts/threading-and-mediators.md, docs/concepts/interleaver-and-hooks.md, docs/usage/trace.md]
sources: [src/nnsight/tracing/tracer.py, src/nnsight/intervention/interleaver.py, src/nnsight/intervention/tracer.py, src/nnsight/intervention/batching.py, src/nnsight/intervention/envoy.py, src/nnsight/tracing/util.py]
---

# Errors Index

The exceptions nnsight raises, with the real class, the real message text, and the
page that explains the cause and fix.

When code inside a `with model.trace(...):` block raises, nnsight cleans the
traceback (`clean_traceback`, `src/nnsight/tracing/util.py:139`) so the top frames
point at your own source rather than nnsight's plumbing. The exception type is
preserved — `except ValueError:` / `except IndexError:` still work. See
[debug-mode.md](debug-mode.md) for what `CONFIG.APP.DEBUG` does (and no longer does).

## Execution-order errors

Raised when an intervention asks for a value the model never delivered, or asks in
the wrong order. There is one class for both: `OutOfOrderError`
(`src/nnsight/intervention/interleaver.py:83`).

| Exception | Message | Doc |
|---|---|---|
| `OutOfOrderError` | ``'<location>.i0' was requested but the model already ran past it`` | [out-of-order-error.md](out-of-order-error.md) |
| `OutOfOrderError` (dangling worker) | ``'<location>.i0' was requested but the model already ran past it`` — raised at the end of the run for a worker still waiting on a location that never fired | [value-was-not-provided.md](value-was-not-provided.md) |
| `UserWarning` (not an exception) | ``'<location>' was never reached: the model ran fewer iterations than the loop requested. Values from reached iterations are kept.`` — an `iter` loop that outran the model | [value-was-not-provided.md](value-was-not-provided.md) |

> `MissedProviderError` (and its old `OutOfOrderError` subclass split) no longer
> exists. Both the eager "asked out of order" case and the late "model finished, a
> worker is still waiting" case now raise the single `OutOfOrderError`.

## Setup / context errors

Raised when an Envoy value is accessed outside a live trace, `save()` is called
outside a trace, a trace has nothing to run, invokes are nested, or batching is
unsupported.

| Exception | Message | Doc |
|---|---|---|
| `ValueError` | ``Cannot access `<location>` outside of interleaving`` | [cannot-access-outside-interleaving.md](cannot-access-outside-interleaving.md) |
| `ValueError` | ``save() was called outside a trace. …`` | [save-outside-trace.md](save-outside-trace.md) |
| `ValueError` | ``trace() needs an input, or at least one `with tracer.invoke(...)` block`` | [cannot-access-outside-interleaving.md](cannot-access-outside-interleaving.md) |
| `ValueError` | ``Cannot invoke while the model is already running.`` | [invoke-during-execution.md](invoke-during-execution.md) |
| `NotImplementedError` | ``<ModelClass> does not support batching multiple invokes`` | [batching-not-implemented.md](batching-not-implemented.md) |
| `WithBlockNotFoundError` | *(no message)* | [with-block-not-found.md](with-block-not-found.md) |

## Tracing / capture errors

Raised while capturing the `with` block's source.

| Exception | Message | Source / fix |
|---|---|---|
| `ValueError` | ``The body of a traced `with` must start on its own line; nnsight runs the body itself, and can only intercept it at the start of a line.`` | `src/nnsight/tracing/tracer.py:76`. Move the body off the `with` line: never write `with model.trace(x): out = ...`; put `out = ...` on the next, indented line. |
| `WithBlockNotFoundError` | *(no message)* | `src/nnsight/tracing/tracer.py:306`. The tracer wasn't used as a `with` block. See [with-block-not-found.md](with-block-not-found.md). |

## Coordination errors

Raised when trace blocks are wired together incorrectly (see
[docs/usage/](../usage/) for `barrier` / `skip`).

| Exception | Message | Source / fix |
|---|---|---|
| `ValueError` | ``A barrier was never reached by every block it waits for; check the count it was created with`` | `src/nnsight/intervention/interleaver.py:631`. `tracer.barrier(n)` was created for more blocks than actually call it. Pass the real number of blocks that hold the barrier. |
| `ValueError` | ``A batched `.skip()` has to cover every row: skip the module in every invoke, or none — a shared forward can't run for only the rows an invoke left unskipped.`` | `src/nnsight/intervention/batching.py:164`. When batching multiple invokes, either `.skip()` a module in *every* invoke or in none. |

## Remote execution

`RemoteBackend` (`src/nnsight/intervention/backends/remote.py`) raises `RemoteError`
on a failed submission or a server-side `ERROR` status. A deferred worker error from
a driver like vLLM is re-raised at the client as a `RuntimeError` carrying the
original type/message/traceback (`src/nnsight/intervention/errors.py:45`).

## Debugging

| Topic | Doc |
|---|---|
| What `CONFIG.APP.DEBUG` does, and how tracebacks are cleaned | [debug-mode.md](debug-mode.md) |
