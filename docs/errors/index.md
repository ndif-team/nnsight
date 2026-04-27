---
title: Errors Index
one_liner: Map of nnsight exceptions to their cause-and-fix docs.
tags: [error, index]
related: [docs/concepts/threading-and-mediators.md, docs/concepts/interleaver-and-hooks.md, docs/usage/trace.md, docs/remote/]
sources: [src/nnsight/intervention/interleaver.py, src/nnsight/intervention/envoy.py, src/nnsight/intervention/batching.py, src/nnsight/intervention/tracing/invoker.py, src/nnsight/intervention/tracing/base.py, src/nnsight/intervention/serialization.py, src/nnsight/modeling/language.py]
---

# Errors Index

When code inside a `with model.trace(...)` block raises, nnsight reconstructs the traceback so the top frame points to the user's source. The exception type is preserved (`isinstance(e, ValueError)` still works). To see internal nnsight frames, set `nnsight.CONFIG.APP.DEBUG = True` (see `docs/errors/debug-mode.md`).

## Execution-order errors

Most often raised when an intervention asks for a value that the model never delivered, or asks in the wrong order.

`MissedProviderError` is the **primary** post-`refactor/transform` failure mode for "I asked for a value the model never produced." `OutOfOrderError` is its subclass — same root cause, but detected eagerly when nnsight already knows the provider has fired and been consumed.

| Exception | Symptom snippet | Doc |
|---|---|---|
| `Mediator.MissedProviderError` | ``Execution complete but `<requester>` was not provided. Did you call an Envoy out of order?`` | [missed-provider-error.md](missed-provider-error.md) |
| `Mediator.OutOfOrderError` (subclass of `MissedProviderError`) | ``Value was missed for <requester>. Did you call an Envoy out of order?`` | [out-of-order-error.md](out-of-order-error.md) |
| `ValueError` (raised via `MissedProviderError`) | ``Execution complete but `<requester>` was not provided. Did you call an Envoy out of order?`` | [value-was-not-provided.md](value-was-not-provided.md) |

## Setup / context errors

Raised when an Envoy property is accessed outside a live trace, when invokes are nested incorrectly, or when batching is unsupported.

| Exception | Symptom snippet | Doc |
|---|---|---|
| `ValueError` | ``Cannot access `<path>.output` outside of interleaving.`` / ``Cannot set `<path>.output` outside of interleaving.`` | [model-did-not-execute.md](model-did-not-execute.md) |
| `ValueError` | ``Cannot invoke during an active model execution / interleaving.`` | [invoke-during-execution.md](invoke-during-execution.md) |
| `NotImplementedError` | ``Batching is not implemented for this model.`` | [batching-not-implemented.md](batching-not-implemented.md) |
| `WithBlockNotFoundError` | ``With block not found at line <N>`` | [with-block-not-found.md](with-block-not-found.md) |

## Serialization errors (remote execution)

Raised by `nnsight.intervention.serialization.make_function` when a function captured for remote execution can't be reconstructed on the receiving side. These almost always surface only on remote (NDIF) traces — local execution doesn't reserialize traced functions. See [`docs/remote/`](../remote/) for the remote-execution context and how submitted functions are pickled / re-compiled.

| Exception | Symptom | Source |
|---|---|---|
| `ValueError` | ``Failed to compile source for function '<name>'. This may indicate corrupted serialized data or version incompatibility.`` | `src/nnsight/intervention/serialization.py:552` (closure path), `:568` (no-closure path) |
| `ValueError` | ``Could not find function '<name>' in compiled source`` | `src/nnsight/intervention/serialization.py:583` |
| `pickle.PicklingError` | ``Cannot serialize function '<name>': source code unavailable. Attach source manually via func.__source__ = '...'.`` | `src/nnsight/intervention/serialization.py:789` |

These are caused by:
- A Python version mismatch between client and server that breaks a syntax form.
- Serialized payload corruption in transit.
- Functions defined in a way that `inspect.getsource` can't recover (lambdas in a `-c` string, `exec()`'d functions).

The fix is usually to ensure the function comes from a real `.py` file (or has `func.__source__ = "..."` attached).

## Debugging

| Topic | Doc |
|---|---|
| Enabling full tracebacks (internal frames) | [debug-mode.md](debug-mode.md) |
