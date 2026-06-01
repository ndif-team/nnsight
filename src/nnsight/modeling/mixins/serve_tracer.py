"""Client-side tracer for ``model.trace(..., serve=url)``.

Holds serve-specific state (the in-flight HTTP future for non-blocking
requests) on a dedicated subclass so the base tracing layer stays free
of serve concerns. Shared by both ``VLLM.trace(serve=...)`` and
``LanguageModel.trace(serve=...)`` since both pair with the same
:class:`LocalServeBackend` protocol.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Dict, Optional

from .remoteable import RemoteInterleavingTracer


class ServeInterleavingTracer(RemoteInterleavingTracer):
    """Tracer for client-side calls to nnsight-serve / nnsight-vllm-serve.

    Selected via ``tracer_cls=`` from :meth:`VLLM.trace` and
    :meth:`LanguageModel.trace` when ``serve=`` is set. Owns the
    non-blocking request handle and exposes :meth:`collect` so callers
    can retrieve saves once the server responds.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._serve_future: Optional[Future] = None

    def collect(self, timeout: float = None) -> Dict:
        """Block until a non-blocking serve request completes and return saves.

        Only valid after ``model.trace(..., serve=url, blocking=False)``.

        Args:
            timeout: Maximum seconds to wait. ``None`` means wait forever.

        Returns:
            Dict mapping saved variable names to their values.

        Raises:
            RuntimeError: If no serve request is in flight (e.g.
                ``blocking=True`` was used).
            Exception: Any deferred server-side error surfaced via
                ``surface_server_errors``.

        Example::

            with model.trace("prompt", serve=url, blocking=False) as t:
                out = model.logits.output.save()
            saves = t.collect()
            out = saves["out"]
        """
        if self._serve_future is None:
            raise RuntimeError(
                "Tracer has no pending serve request. "
                "Did you use blocking=False?"
            )
        # The non-blocking serve path stores the raw future from
        # ``LocalServeBackend._send`` which returns ``(saves, errors)``.
        # Surface deferred server-side errors here so the contract
        # matches the blocking path: errors raise at the caller's
        # await boundary rather than being silently dropped.
        saves, errors = self._serve_future.result(timeout=timeout)
        if errors:
            from ...intervention.errors import surface_server_errors
            surface_server_errors(errors, context="[nnsight-serve]")
        return saves
