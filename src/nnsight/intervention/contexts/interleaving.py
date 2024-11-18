import weakref
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)


from ...tracing.backends import Backend
from ...tracing.graph import GraphType
from ..graph import (
    InterventionGraph,
    InterventionNode,
    InterventionNodeType,
    ValidatingInterventionNode,
)

from . import Invoker
from . import InterventionTracer
if TYPE_CHECKING:
    from .. import NNsight


class InterleavingTracer(InterventionTracer):

    def __init__(
        self,
        model: "NNsight",
        *invoker_args,
        method: Optional[str] = None,
        backend: Optional[Backend] = None,
        parent: Optional[GraphType] = None,
        scan: bool = False,
        invoker_kwargs: Dict[str, Any] = {},
        validate: bool = False,
        **kwargs,
    ) -> None:

        super().__init__(
            graph_class=InterventionGraph,
            model=model,
            node_class=ValidatingInterventionNode if validate else InterventionNode,
            proxy_class=model.proxy_class,
            backend=backend,
            parent=parent,
            graph=model._default_graph,
        )

        self._model = model

        self._model._envoy._set_tracer(weakref.proxy(self))

        self.invoker: Optional[Invoker] = None
        self.invoker_args = invoker_args
        self.invoker_kwargs = invoker_kwargs
        self.invoker_kwargs["scan"] = scan

        self.args = [method]
        self.kwargs = kwargs

    def invoke(self, *inputs: Any, **kwargs) -> Invoker:
        """Create an Invoker context for a given input.

        Raises:
            Exception: If an Invoker context is already open

        Returns:
            Invoker: Invoker.
        """

        if self.invoker is not None:

            raise Exception("Can't create an invoker context with one already open!")

        return Invoker(self, *inputs, **kwargs)

    def __enter__(self):

        tracer = super().__enter__()

        if self.invoker_args:

            invoker = self.invoke(*self.invoker_args, **self.invoker_kwargs)
            invoker.__enter__()

        return tracer

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    
        if self.invoker is not None:

            self.invoker.__exit__(None, None, None)

        self._model._envoy._reset()

        super().__exit__(exc_type, exc_val, exc_tb)

    @classmethod
    def _batch(
        cls, model: "NNsight", invoker_inputs: Tuple[Tuple[Tuple[Any], Dict[str, Any]]]
    ) -> Tuple[Tuple[Tuple[Any], Dict[str, Any]], List[Tuple[int, int]]]:

        batch_groups = []
        batch_start = 0
        batched_input = None

        for args, kwargs in invoker_inputs:
            (args, kwargs), batch_size = model._prepare_input(*args, **kwargs)

            batch_groups.append((batch_start, batch_size))

            batched_input = model._batch(batched_input, *args, **kwargs)

            batch_start += batch_size

        if batched_input is None:

            batched_input = (((0, -1),), dict())

        return batched_input, batch_groups

    @property
    def _invoker_group(self):

        return len(self.args) - 2
    
    @classmethod
    def execute(cls, node: InterventionNodeType):

        graph, method, *invoker_inputs = node.args
        
        graph: InterventionGraph
        model = graph.model

        invoker_inputs, kwargs = node.prepare_inputs((invoker_inputs, node.kwargs))

        (invoker_args, invoker_kwargs), batch_groups = cls._batch(model, invoker_inputs)

        graph.compile()

        graph.reset()

        graph.execute()

        model.interleave(
            *invoker_args,
            fn=method,
            intervention_graph=graph,
            batch_groups=batch_groups,
            **kwargs,
            **invoker_kwargs,
        )


