import pickle
import uuid
from typing import TYPE_CHECKING, Any

import zstandard as _zstd

_ZSTD_DECOMPRESSOR = _zstd.ZstdDecompressor()

from ...intervention.backends.base import Backend
from ...intervention.tracing.util import wrap_exception
from .lazy_remote_tensor import merge_saved

if TYPE_CHECKING:
    from .vllm import VLLM
else:
    VLLM = Any


class AsyncVLLMBackend(Backend):
    """Backend for async vLLM generation that returns an async generator.

    Usage pattern:
    - ``__call__(tracer)``: Called from ``__exit__``. Compiles the traced
      code, sets up mediators, serializes them into sampling params, and
      immediately submits the request to the async engine via ``.generate()``.
    - ``__call__()``: Called by user via ``tracer.backend()``. Returns an
      async generator that streams ``RequestOutput`` from the already-submitted
      request.
    """

    def __init__(self, model: "VLLM"):
        self.model = model
        self._generator = None
        self._request_id = None

    def __call__(self, tracer):
        """Compile traced code, set up mediators, serialize, and submit.

        Uses ``tracer._setup_interleaver()`` directly instead of going
        through ``tracer.execute()`` / ``model.interleave()``, since the
        async path only needs to serialize mediators — not run the model.

        Submits the request to the async engine immediately so vLLM can
        start processing it via dynamic batching before the user awaits.
        """
        fn = Backend.__call__(self, tracer)

        try:
            # Set up mediators and collect batched args (shared with sync path).
            args, kwargs = tracer._setup_interleaver(fn)

            if not self.model.dispatched:
                self.model.dispatch()

            # Serialize mediators into sampling params.
            prompts, params, lora_requests = self.model._serialize_mediators(
                *args, **kwargs
            )

            # Submit the request to the engine immediately.
            self._request_id = str(uuid.uuid4())
            self._generator = self.model.vllm_entrypoint.generate(
                prompts[0], params[0], self._request_id, lora_request=lora_requests[0]
            )

            tracer.mediators.clear()
        except Exception as e:
            raise wrap_exception(e, tracer.info) from None

    def __await__(self):
        return self._generator.__await__()

    async def __aiter__(self):
        # Saves are collected ONLY on the finished output (one per request).
        # Intermediate (non-finished) outputs yield without `output.saves` populated.
        # A per-yield streaming-saves mode existed briefly during development and
        # may return as an opt-in option in the future, but the current behavior
        # is finished-only.
        async for output in self._generator:
            if output.finished:
                finished = [output.request_id]
                results = await self.model.vllm_entrypoint.collective_rpc(
                    "collect_nnsight",
                    args=([output.request_id], finished),
                )
                # Worker returns ``{base_id: {var_name: value}}``. Without
                # PP only TP-rank-0 returns data; with PP > 1 every PP
                # stage's TP-rank-0 contributes — each ships only the slots
                # it owns (others are NOT_ON_THIS_RANK sentinels), so
                # same-named saves are merged position-wise into one complete
                # result (``merge_saved``).
                merged: dict = {}
                for r in results:
                    if r is None:
                        continue
                    rank_saves = pickle.loads(_ZSTD_DECOMPRESSOR.decompress(r))
                    for base_id, per_req in rank_saves.items():
                        dst = merged.setdefault(base_id, {})
                        for name, value in per_req.items():
                            dst[name] = (
                                merge_saved(dst[name], value) if name in dst else value
                            )
                per_req = merged.get(output.request_id)
                if per_req:
                    # Surface server-side deferred exceptions before exposing
                    # saves — otherwise the caller sees UnboundLocalError on
                    # saves the mediator never produced, with no hint of the
                    # real cause. Mirrors ``vllm.py:__call__`` and
                    # ``intervention/backends/local_serve.py:146``.
                    exc_map = per_req.pop("__nnsight_exceptions__", None)
                    if exc_map:
                        from ...intervention.errors import surface_server_errors
                        surface_server_errors(
                            list(exc_map.values()),
                            context="[vLLM async]",
                        )
                    output.saves = per_req
            yield output
