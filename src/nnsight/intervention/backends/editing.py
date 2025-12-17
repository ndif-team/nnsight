from .base import Backend
from typing import TYPE_CHECKING, Any
from ..interleaver import Mediator

if TYPE_CHECKING:
    from ..tracing.tracer import InterleavingTracer
else:
    InterleavingTracer = Any


class EditingBackend(Backend):

    def __call__(self, tracer: InterleavingTracer):

        invoker = tracer.invoke()
        invoker.info = tracer.info.copy()

        fn = super().__call__(invoker)

        mediator = Mediator(fn, invoker.info)

        tracer.model._default_mediators = tracer.model._default_mediators + [mediator]
