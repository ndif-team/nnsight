import pickle
import uuid
from typing import TYPE_CHECKING, Any

import zstandard as _zstd

_ZSTD_DECOMPRESSOR = _zstd.ZstdDecompressor()

from ...intervention.backends.base import Backend
from ...intervention.tracing.globals import Globals
from ...intervention.tracing.util import wrap_exception

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
            Globals.enter()

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
        finally:
            Globals.exit()

    def __await__(self):
        return self._generator.__await__()

    async def __aiter__(self):
        async for output in self._generator:
            if output.finished:
                finished = [output.request_id]
                results = await self.model.vllm_entrypoint.collective_rpc(
                    "collect_nnsight",
                    args=([output.request_id], finished),
                )
                saves_bytes = next((r for r in results if r is not None), None)
                if saves_bytes:
                    output.saves = pickle.loads(_ZSTD_DECOMPRESSOR.decompress(saves_bytes))
            yield output
