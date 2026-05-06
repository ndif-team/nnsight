"""Boundary exception types for ``execute_request``.

Raised at the execute-request boundary so callers (FastAPI route, NDIF
Ray actor, in-process schedulers) can map them to whatever status they
expose to their own clients (HTTP status, BackendResponseModel, …).

These are distinct from the per-mediator deferred-error envelope in
``intervention/errors.py`` (``DeferredError`` /
``surface_server_errors``). Those flow through ``RequestOutput.saves``
and surface at the trace boundary on the client side; the exceptions
here surface synchronously to the caller of ``execute_request`` itself.
"""

from __future__ import annotations


class NNsightVLLMError(Exception):
    """Base class for all ``execute_request`` boundary errors."""


class TraceCompilationError(NNsightVLLMError):
    """Compiling the user's intervention function or serializing the
    mediators failed before any engine work was scheduled.

    Indicates a malformed request (bad pickle, invalid trace structure,
    unresolved envoy paths, …). The FastAPI route maps this to HTTP 400.
    """


class EngineNotDispatchedError(NNsightVLLMError):
    """``execute_request`` was called against a ``VLLM`` whose engine is
    not yet dispatched.

    Auto-dispatch from a request handler is a TOCTOU hazard (two
    concurrent first-requests can both observe ``dispatched=False`` and
    race into ``model.dispatch()``); the standalone server pre-dispatches
    in its CLI, and in-process callers are expected to do the same. The
    FastAPI route maps this to HTTP 503.
    """


class GenerationError(NNsightVLLMError):
    """``engine.generate`` or ``collective_rpc("collect_nnsight", …)``
    raised on a per-invoke task.

    Distinct from per-mediator deferred user-code errors, which surface
    via the ``errors`` envelope on the result and re-raise at the client's
    trace boundary. ``GenerationError`` indicates the engine itself
    (or the worker RPC) failed — the request can't be completed at all.
    The FastAPI route maps this to HTTP 500.
    """
