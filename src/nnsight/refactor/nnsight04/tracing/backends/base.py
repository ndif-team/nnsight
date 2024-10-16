from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..contexts import Context


class Backend:

    def __call__(self, context: "Context") -> None:

        context.graph.reset()
        context.graph.execute()


class ChildBackend(Backend):

    def __call__(self, context: "Context") -> None:

        context.add(
            context.graph.stack[-1], context.graph, *context.args, **context.kwargs
        )
