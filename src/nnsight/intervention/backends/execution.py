from typing import TYPE_CHECKING, Any

from ..tracing.util import wrap_exception
from .base import Backend


if TYPE_CHECKING:
    from ..tracing.tracer import Tracer
else:
    Tracer = Any


class ExecutionBackend(Backend):

    def __call__(self, tracer: Tracer):

        fn = super().__call__(tracer)

        try:
            # ``Object.save`` mount is installed in
            # ``InterleavingTracer._setup_interleaver`` — the chokepoint
            # that every executor (this backend, AsyncVLLMBackend, vLLM
            # serve ``server.py``, LocalSimulationBackend) routes through
            # before running the outer trace body.
            return tracer.execute(fn)
        except Exception as e:

            raise wrap_exception(e, tracer.info) from None
