from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from ...tracing.contexts import Context
from ...tracing.graph import SubGraph
from ..protocols import ApplyModuleProtocol, GradProtocol, InterventionProtocol
from . import InterventionNode, InterventionNodeType, InterventionProxyType

if TYPE_CHECKING:
    from .. import NNsight


class InterventionGraph(SubGraph[InterventionNode, InterventionProxyType]):

    def __init__(
        self,
        *args,
        model: Optional["NNsight"] = None,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)
        
        self.model = model

        self.interventions: Dict[str, List[InterventionNode]] = defaultdict(list)
        self.grad_subgraph: Set[int] = set()

        self.compiled = False
        self.call_counter: Dict[int, int] = defaultdict(int)
        self.deferred:Dict[int, List[int]] = defaultdict(list)

    def reset(self) -> None:
        self.call_counter = defaultdict(int)
        return super().reset()

    def set(self, model: "NNsight"):

        self.model = model

    def context_dependency(
        self,
        context_node: InterventionNode,
        intervention_subgraphs: List[SubGraph],
    ) -> None:

        context_graph: SubGraph = context_node.args[0]

        start = context_graph.subset[0]
        end = context_graph.subset[-1]

        for intervention_subgraph in intervention_subgraphs:

            # continue if the subgraph does not overlap with the context's graph
            if intervention_subgraph.subset[-1] < start or end < intervention_subgraph.subset[0]:
                continue

            for intervention_index in intervention_subgraph.subset:

                # if there's an overlapping node, make the context depend on the intervention node in the subgraph
                if start <= intervention_index and intervention_index <= end:

                    # the first node in the subgraph is an InterventionProtocol node
                    intervention_node = intervention_subgraph[0]

                    context_node._dependencies.add(intervention_node.index)
                    intervention_node._listeners.add(context_node.index)
                    # TODO: maybe we don't need this
                    intervention_subgraph.subset.append(context_node.index)

                    break

    def compile(self) -> Optional[Dict[str, List[InterventionNode]]]:

        if self.compiled:
            return self.interventions

        intervention_subgraphs: List[SubGraph] = []

        start = self[0].index

        # is the first node corresponding to an executable graph?
        # occurs when a Conditional or Iterator context is explicitly entered by a user
        if isinstance(self[0].target, type) and issubclass(
            self[0].target, Context
        ):
            graph = self[0].args[0]

            # handle emtpy if statments or for loops
            if len(graph) > 0:
                start = graph[0].index

        end = self[-1].index + 1

        context_start: int = None
        defer_start: int = None
        context_node: InterventionNode = None

        # looping over all the nodes created within this graph's context
        for index in range(start, end):

            node: InterventionNodeType = self.nodes[index]

            # is this node part of an inner context's subgraph?
            if context_node is None and node.graph is not self:
                
                context_node = self.nodes[node.graph[-1].index + 1]

                context_start = self.subset.index(context_node.index)
                
                defer_start = node.index

                self.context_dependency(context_node, intervention_subgraphs)

            if node.target is InterventionProtocol:
                
                # build intervention subgraph
                subgraph = SubGraph(self, subset=sorted(list(node.subgraph())))

                module_path, *_ = node.args

                self.interventions[module_path].append(node)

                intervention_subgraphs.append(subgraph)

                # if the InterventionProtocol is defined within a sub-context
                if context_node is not None:
  
                    # make the current context node dependent on this intervention node
                    context_node._dependencies.add(node.index)
                    node._listeners.add(context_node.index)
                    # TODO: maybe we don't need this
                    self.subset.append(node.index)

                    graph: SubGraph = node.graph

                    graph.subset.remove(node.index)

                    node.kwargs["start"] = context_start
                    node.kwargs["defer_start"] = defer_start

                    node.graph = self

                else:

                    node.kwargs["start"] = self.subset.index(subgraph.subset[0])
                    node.kwargs["defer_start"] = node.kwargs["start"]

            elif node.target is GradProtocol:

                subgraph = SubGraph(self, subset=sorted(list(node.subgraph())))

                intervention_subgraphs.append(subgraph)

                self.grad_subgraph.update(subgraph.subset[1:])

                if context_node is not None:

                    context_node._dependencies.add(node.index)
                    node._listeners.add(context_node.index)
                    subgraph.subset.append(context_node.index)

                    graph: SubGraph = node.graph

                    graph.subset.remove(node.index)

                    node.kwargs["start"] = context_start

                    node.graph = self

                else:

                    node.kwargs["start"] = self.subset.index(subgraph.subset[1])

            elif node.target is ApplyModuleProtocol:

                node.graph = self

            elif context_node is not None and context_node is node:
                context_node = None

        self.compiled = True

    def execute(self, start: int = 0, grad: bool = False, defer:bool=False, defer_start:int=0) -> None:
                        
        exception = None
                
        if defer_start in self.deferred:
                    
            for index in self.deferred[defer_start]:
                
                self.nodes[index].reset()
                
            del self.deferred[defer_start]
            
        if defer:

            self.defer_stack.append(defer_start)

        for node in self[start:]:

            if node.executed:
                continue
            elif node.index != self[start].index and node.target is InterventionProtocol:
                break
            elif node.fulfilled:
                try:
                    node.execute()
                    if defer and node.target is not InterventionProtocol:
                        self.deferred[defer_start].append(node.index)
                except Exception as e:
                    exception = (node.index, e)
                    break
            elif not grad and node.index in self.grad_subgraph:
                continue
            else:
                break
            
        if defer:
            self.defer_stack.pop()

        if exception is not None:
            defer_stack = self.defer_stack
            self.defer_stack = []
            self.clean(exception[0])
            self.defer_stack = defer_stack
            raise exception[1]

    def count(
        self, index: int, iteration: Union[int, List[int], slice]
    ) -> bool:
        """Increments the count of times a given Intervention Node has tried to be executed and returns if the Node is ready and if it needs to be deferred.

        Args:
            index (int): Index of intervention node to return count for.
            iteration (Union[int, List[int], slice]): What iteration(s) this Node should be executed for.

        Returns:
            bool: If this Node should be executed on this iteration.
            bool: If this Node and recursive listeners should have updating their remaining listeners (and therefore their destruction) deferred.
        """

        ready = False
        defer = False

        count = self.call_counter[index]

        if isinstance(iteration, int):
            ready = count == iteration
        elif isinstance(iteration, list):
            iteration.sort()

            ready = count in iteration
            defer = count != iteration[-1]

        elif isinstance(iteration, slice):

            start = iteration.start or 0
            stop = iteration.stop

            ready = count >= start and (stop is None or count < stop)

            defer = stop is None or count < stop - 1

        # if defer:
        #     self.deferred.add(index)
        # else:
        #     self.deferred.discard(index)

        self.call_counter[index] += 1

        return ready, defer
    
    # def clean(self):
        
    #     for deferred in self.def

    # @classmethod
    # def shift(cls, mgraph: MultiGraph) -> MultiGraph:

    #     InterventionProtocol.compile(mgraph)

    #     intervention_subgraphs = InterventionProtocol.get_interventions(mgraph).values()

    #     graph_id_to_invoker_groups = defaultdict(set)
    #     graph_id_to_intervention_node = defaultdict(list)

    #     for subgraph in intervention_subgraphs:
    #         for (start, end) in subgraph:

    #             node = mgraph[start]

    #             invoker_group = node.args[1]

    #             offset = 0

    #             for graph in mgraph.id_to_graphs.values():
    #                 offset  += len(graph)
    #                 if start < offset:
    #                     graph_id_to_invoker_groups[graph.id].add(invoker_group)
    #                     graph_id_to_intervention_node[graph.id].append(node)
    #                     break

    #     global_offset = 0

    #     for graph_id, invoker_groups in graph_id_to_invoker_groups.items():

    #         min_group = min(invoker_groups)
    #         max_group = max(invoker_groups)

    #         offset = global_offset - min_group

    #         for node in graph_id_to_intervention_node[graph_id]:

    #             node.args[1] += offset

    #         global_offset += max_group + 1

    #     return mgraph
