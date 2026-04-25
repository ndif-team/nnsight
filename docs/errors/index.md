---
title: Errors Index
one_liner: Map of nnsight exceptions to their cause-and-fix docs.
tags: [error, index]
related: [docs/concepts/threading-and-mediators.md, docs/concepts/interleaver-and-hooks.md, docs/usage/trace.md]
sources: [src/nnsight/intervention/interleaver.py, src/nnsight/intervention/envoy.py, src/nnsight/intervention/batching.py, src/nnsight/intervention/tracing/invoker.py, src/nnsight/modeling/language.py]
---

# Errors Index

When code inside a `with model.trace(...)` block raises, nnsight reconstructs the traceback so the top frame points to the user's source. The exception type is preserved (`isinstance(e, ValueError)` still works). To see internal nnsight frames, set `nnsight.CONFIG.APP.DEBUG = True` (see `docs/errors/debug-mode.md`).

## Execution-order errors

Most often raised when an intervention asks for a value that the model never delivered, or asks in the wrong order.

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
| `AttributeError` | ``Tokenizer not found. If you passed a pre-loaded model to `LanguageModel`, you need to provide a tokenizer when initializing.`` | [tokenizer-not-found.md](tokenizer-not-found.md) |

## Debugging

| Topic | Doc |
|---|---|
| Enabling full tracebacks (internal frames) | [debug-mode.md](debug-mode.md) |
