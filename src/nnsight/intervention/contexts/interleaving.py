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
from ..interleaver import Interleaver

from . import Invoker
from . import InterventionTracer
if TYPE_CHECKING:
    from .. import NNsight


class InterleavingTracer(InterventionTracer):
    """This is the Tracer type that actually interleaves an `InterventionGraph` with a PyTorch model upon execute.

    Attributes:
        _model (NNsight): NNsight model.
        invoker (Invoker): Current open invoker so we can prevent opening two at the same time.
        args (Tuple[...]): Positional arguments. First is which method to interleave with and subsequent args are invoker inputs.
        kwargs (Dict[str,Any]): Keyword arguments passed to the method to interleave. These are "global" keyword arguments for our chosen methof
            while kwargs for a given invoker are used for preprocessing the invoker input.
    """

    def __init__(
        self,
        model: "NNsight",
        method: Optional[str] = None,
        backend: Optional[Backend] = None,
        parent: Optional[GraphType] = None,
        validate: bool = False,
        debug: Optional[bool] = None,
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
            debug=debug,
        )

        self._model = model

        # Tell all Envoy's about the current Tracer so they can use it to add InterventionProtocol Nodes.
        self._model._envoy._set_tracer(weakref.proxy(self))

        self.invoker: Optional[Invoker] = None

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

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    
        if self.invoker is not None:

            self.invoker.__exit__(None, None, None)

        self._model._envoy._reset()

        super().__exit__(exc_type, exc_val, exc_tb)

    @classmethod
    def _batch(
        cls, model: "NNsight", invoker_inputs: Tuple[Tuple[Tuple[Any], Dict[str, Any]]]
    ) -> Tuple[Tuple[Tuple[Any], Dict[str, Any]], List[Tuple[int, int]]]:
        """Batches together each set of inputs from each Invoker by iteratively calling the models ._prepare_input and ._batch methods.

        Args:
            model (NNsight): Model which defines its own logic for preparing and batching input
            invoker_inputs (Tuple[Tuple[Tuple[Any], Dict[str, Any]]]): Tuple of invoker inputs.

        Returns:
            Tuple[Tuple[Tuple[Any], Dict[str, Any]], List[Tuple[int, int]]]: One single batched input.
            List[Tuple[int, int]]: Batch groups
        """

        batch_groups = []
        batch_start = 0
        batched_input = None

        for args, kwargs in invoker_inputs:

            (args, kwargs), batch_size = model._prepare_input(*args, **kwargs)

            batch_groups.append((batch_start, batch_size))

            batched_input = model._batch(batched_input, *args, **kwargs)

            batch_start += batch_size

        if batched_input is None:

            return (tuple(), dict()), ((0, -1),)

        return batched_input, batch_groups

    @property
    def _invoker_group(self):

        return len(self.args) - 2
    
    @classmethod
    def execute(cls, node: InterventionNodeType):

        graph, method, *invoker_inputs = node.args
        
        
        
        graph: InterventionGraph
        model = graph.model

        # There may be Nodes in the inputs. Convert them to their value
        invoker_inputs, kwargs = node.prepare_inputs((invoker_inputs, node.kwargs))

        # Batch each invoker input into one input
        (invoker_args, invoker_kwargs), batch_groups = cls._batch(model, invoker_inputs)

        # Compile Intervention Graph
        graph.compile()

        graph.reset()
        
        interleaver = Interleaver(graph, batch_groups=batch_groups)
        
        graph.model.interleave(interleaver, *invoker_args, fn=method,**kwargs, **invoker_kwargs)
        
        graph.cleanup()
