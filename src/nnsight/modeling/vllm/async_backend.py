import pickle
import uuid
from typing import TYPE_CHECKING, Any

from ...intervention.backends.base import Backend
from ...intervention.tracing.util import wrap_exception
from .execute import _compile_invokes

if TYPE_CHECKING:
    from .vllm import VLLM
else:
    VLLM = Any


class AsyncVLLMBackend(Backend):
    """Backend for async vLLM generation that returns an async generator.

    Usage pattern:
    - ``__call__(tracer)``: Called from ``__exit__``. Compiles the traced
      code, sets up mediators via ``_run_user_fn()``, serializes them
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

        Local-trace path: dispatches the engine if needed, then delegates
        the compile block to :func:`_compile_invokes` (shared with the
        serve handler's ``execute_request``). Auto-dispatch is safe here
        because this runs on the user's call thread, not a request
        boundary — there is no TOCTOU hazard the way there is in the
        FastAPI handler.
        """
        fn = Backend.__call__(self, tracer)

        if not self.model.dispatched:
            self.model.dispatch()

        try:
            prompts, params, lora_requests, kwargs = _compile_invokes(
                self.model, tracer, fn,
            )
        except Exception as e:
            raise wrap_exception(e, tracer.info) from None

        self._prompts = prompts
        self._params = params
        self._lora_requests = lora_requests
        self._kwargs = kwargs

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
            saves_bytes = next((r for r in results if r is not None), None)
            if saves_bytes:
                saves = pickle.loads(saves_bytes)
                output.saves = saves
            yield output
