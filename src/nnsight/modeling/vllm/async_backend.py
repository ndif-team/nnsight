import pickle
import uuid
from typing import TYPE_CHECKING, Any, Optional

from ...intervention.backends.base import Backend
from ...intervention.tracing.globals import Globals
from ...intervention.tracing.util import wrap_exception

if TYPE_CHECKING:
    from .async_tracer import AsyncInterleavingTracer
    from .vllm import VLLM
else:
    AsyncInterleavingTracer = Any
    VLLM = Any


class AsyncVLLMBackend(Backend):
    """Backend for async vLLM generation that returns an async generator.

    Dual-call pattern:
    - ``__call__(tracer)``: Called from ``__exit__``. Compiles and executes
      the traced function which stores prepared data on the tracer.
    - ``__call__()``: Called by user via ``tracer.backend()``. Returns an async
      generator that streams ``RequestOutput`` objects from ``AsyncLLM``.
    """

    def __init__(self, model: "VLLM"):
        self.model = model
        self._prompts = None
        self._params = None
        self._kwargs = None

    def __call__(self, tracer: Optional["AsyncInterleavingTracer"] = None):
        if tracer is not None:
            # Compile step: call base Backend.__call__ to get compiled function
            fn = super().__call__(tracer)

            # Execute step: run the compiled function which sets up mediators
            # and prepares generation data on the tracer.
            try:
                Globals.enter()
                tracer.execute(fn)
            except Exception as e:
                raise wrap_exception(e, tracer.info) from None
            finally:
                Globals.exit()

            # Grab prepared data from the tracer (not the model).
            if tracer.prepared is not None:
                self._prompts, self._params, self._kwargs = tracer.prepared

            return

        # No tracer: return async generator for streaming
        return self._stream()

    async def _stream(self):
        """Async generator that submits to AsyncLLM and streams results.

        On every output, collects current saves from the worker via
        ``collect_nnsight``.  When the request finishes, the mediator
        is also finalized and cleaned up.
        """
        if self._prompts is None:
            raise RuntimeError(
                "No prepared data. Ensure model.trace() context has exited "
                "before calling tracer.backend()."
            )

        request_id = str(uuid.uuid4())
        prompt = self._prompts[0]
        param = self._params[0]

        async for output in self.model.vllm_entrypoint.generate(
            prompt, param, request_id
        ):
            finished = [output.request_id] if output.finished else None
            results = await self.model.vllm_entrypoint.collective_rpc(
                "collect_nnsight",
                args=([output.request_id], finished),
            )
            saves_bytes = next((r for r in results if r is not None), None)
            if saves_bytes:
                saves = pickle.loads(saves_bytes)
                output.saves = saves
            yield output
