from .base import Backend
from typing import TYPE_CHECKING, Any
from ..interleaver import Mediator, AsyncMediator

if TYPE_CHECKING:
    from ..tracing.tracer import InterleavingTracer
else:
    InterleavingTracer = Any


class EditingBackend(Backend):

    def __call__(self, tracer: InterleavingTracer):

        invoker = tracer.invoke()
        invoker.info = tracer.info.copy()

        fn = super().__call__(invoker)

        mediator_type = AsyncMediator if tracer.asynchronous else Mediator

        mediator = mediator_type(
            fn, invoker.info, batch_group=len(tracer.model._default_mediators)
        )

        tracer.model._default_mediators = tracer.model._default_mediators + [mediator]

        async def async_call():
            pass

        if tracer.asynchronous:
            return async_call()
