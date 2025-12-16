from typing import Callable, TYPE_CHECKING, Any

from ..interleaver import Mediator
from .base import Tracer
from .util import try_catch


if TYPE_CHECKING:
    from .tracer import InterleavingTracer
else:
    InterleavingTracer = Any


class Invoker(Tracer):
    """
    Extends the Tracer class to invoke intervention functions.

    This class captures code blocks and compiles them into intervention functions
    that can be executed by the Interleaver.
    """

    def __init__(self, tracer: InterleavingTracer, *args, **kwargs):
        """
        Initialize an Invoker with a reference to the parent tracer.

        Args:
            tracer: The parent InterleavingTracer instance
            *args: Additional arguments to pass to the traced function
            **kwargs: Additional keyword arguments to pass to the traced function
        """

        if tracer is not None and tracer.model.interleaving:
            raise ValueError(
                "Cannot invoke during an active model execution / interleaving."
            )

        self.tracer = tracer

        super().__init__(*args, **kwargs)

    def compile(self):
        """
        Compile the captured code block into an intervention function.

        The function is wrapped with try-catch logic to handle exceptions
        and signal completion to the mediator.

        Returns:
            A callable intervention function
        """

        self.info.source = [
            f"def __nnsight_tracer_{id(self)}__(__nnsight_mediator__, __nnsight_tracing_info__):\n",
            "    __nnsight_mediator__.pull()\n",
            *try_catch(
                self.info.source,
                exception_source=["__nnsight_mediator__.exception(exception)\n"],
                else_source=["__nnsight_mediator__.end()\n"],
            ),
        ]

        self.info.start_line -= 2

    def execute(self, fn: Callable):
        """
        Execute the compiled intervention function.

        Creates a new Mediator for the intervention function and adds it to the
        parent tracer's mediators list.

        Args:
            fn: The compiled intervention function
        """

        inputs, batch_group = self.tracer.batcher.batch(
            self.tracer.model, *self.args, **self.kwargs
        )

        self.inputs = inputs

        mediator = Mediator(fn, self.info, batch_group=batch_group)

        self.tracer.mediators.append(mediator)
