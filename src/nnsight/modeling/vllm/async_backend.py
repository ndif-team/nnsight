import pickle
import uuid
from typing import TYPE_CHECKING, Any

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
      code, sets up mediators via ``_setup_interleaver()``, serializes them
      into sampling params, and stores the prepared data on this backend.
    - ``__call__()``: Called by user via ``tracer.backend()``. Returns an
      async generator that streams ``RequestOutput`` objects from ``AsyncLLM``.
    """

    def __init__(self, model: "VLLM"):
        self.model = model
        self._prompts = None
        self._params = None
        self._kwargs = None
        self._lora_requests = None

    def _compile_and_execute(self, tracer):
        """Compile traced code, set up mediators, and serialize them.

        Uses ``tracer._setup_interleaver()`` directly instead of going
        through ``tracer.execute()`` / ``model.interleave()``, since the
        async path only needs to serialize mediators — not run the model.
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
            self._prompts = prompts
            self._params = params
            self._lora_requests = lora_requests
            self._kwargs = kwargs

            tracer.mediators.clear()
        except Exception as e:
            raise wrap_exception(e, tracer.info) from None
        finally:
            Globals.exit()

    def __call__(self, tracer=None):
        if tracer is not None:
            self._compile_and_execute(tracer)
            return

        # No tracer: return async generator for streaming.
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
        lora_request = self._lora_requests[0]

        async for output in self.model.vllm_entrypoint.generate(
            prompt, param, request_id, lora_request=lora_request
        ):
            finished = [output.request_id] if output.finished else None
            results = await self.model.vllm_entrypoint.collective_rpc(
                "collect_nnsight",
                args=([output.request_id], finished),
            )
            all_saves = {}
            for r in results:
                if r is not None:
                    all_saves.update(pickle.loads(r))
            if all_saves:
                output.saves = all_saves
            yield output
