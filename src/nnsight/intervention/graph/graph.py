import copy
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

from typing_extensions import Self

from ...tracing.contexts import Context
from ...tracing.graph import SubGraph
from ...util import NNsightError
from ..protocols import ApplyModuleProtocol, GradProtocol, InterventionProtocol
from . import InterventionNode, InterventionNodeType, InterventionProxyType

if TYPE_CHECKING:
    from .. import NNsight
    from ...tracing.graph.graph import GraphType, NodeType


class InterventionGraph(SubGraph[InterventionNode, InterventionProxyType]):
    """The `InterventionGraph` is the special `SubGraph` type that handles the complex intervention operations a user wants to make during interleaving.
    We need to `.compile()` it before execution to determine how to execute interventions appropriately.

    Attributes:
        model (NNsight): NNsight model.
        interventions
        grad_subgraph
        compiled
        call_counter
        deferred
    """

    def __init__(
        self,
        *args,
        model: Optional["NNsight"] = None,
        **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        self.model = model

        self.interventions: Dict[str, List[int]] = defaultdict(list)
        self.grad_subgraph: Set[int] = set()

        self.compiled = False
        self.call_counter: Dict[int, int] = defaultdict(int)
        self.deferred: Dict[int, List[int]] = defaultdict(list)

    def __getstate__(self) -> Dict:

        return {
            "subset": self.subset,
            "nodes": self.nodes,
            "interventions": self.interventions,
            "compiled": self.compiled,
            "call_counter": self.call_counter,
            "deferred": self.deferred,
            "grad_subgraph": self.grad_subgraph,
            "defer_stack": self.defer_stack,
        }

    def __setstate__(self, state: Dict) -> None:

        self.__dict__.update(state)

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

        if len(self) == 0:
            self.compiled = True
            return
        
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

                self.interventions[module_path].append(node.index)

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

    def execute(
        self,
        start: int = 0,
        grad: bool = False,
        defer: bool = False,
        defer_start: int = 0,
    ) -> None:

        err: Tuple[int, NNsightError] = None

        if defer_start in self.deferred:

            for index in self.deferred[defer_start]:

                self.nodes[index].reset()

            del self.deferred[defer_start]

        if defer:

            self.defer_stack.append(defer_start)

        for node in self[start:]:

            if node.executed:
                continue
            elif (
                node.index != self[start].index and node.target is InterventionProtocol
            ):
                break
            elif node.fulfilled:
                try:
                    node.execute()
                    if defer and node.target is not InterventionProtocol:
                        self.deferred[defer_start].append(node.index)
                except NNsightError as e:
                    err = (node.index, e)
                    break
            elif not grad and node.index in self.grad_subgraph:
                continue
            else:
                break

        if defer:
            self.defer_stack.pop()

        if err is not None:
            defer_stack = self.defer_stack
            self.defer_stack = []
            self.clean(err[0])
            self.defer_stack = defer_stack
            raise err[1]

    def count(self, index: int, iteration: Union[int, List[int], slice]) -> bool:
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

    def clean(self, start: Optional[int] = None):

        if start is None:
            start = self[0].index

        end = self[-1].index + 1

        # Loop over ALL nodes within the span of this graph.
        for index in range(start, end):

            node = self.nodes[index]

            if node.executed:
                break

            node.update_dependencies()

    def cleanup(self) -> None:
        """Because some modules may be executed more than once, and to accommodate memory management just like a loop,
        intervention graph sections defer updating the remaining listeners of Nodes if this is not the last time this section will be executed.
        If we never knew it was the last time, there may still be deferred sections after execution.
        These will be leftover in graph.deferred, and therefore we need to update their dependencies.
        """

        # For every intervention graph section (indicated by where it started)
        for start in self.deferred:

            # Loop through all nodes that got their dependencies deferred.
            for index in range(start, self.deferred[start][-1] + 1):

                node = self.nodes[index]

                # Update each of its dependencies
                for dependency in node.dependencies:
                    # Only if it was before start
                    # (not within this section, but before)
                    if dependency.index < start:
                        dependency.remaining_listeners -= 1

                        if dependency.redundant:
                            dependency.destroy()

    def copy(
        self,
        new_graph: Self = None,
        parent: Optional["GraphType"] = None,
        memo: Optional[Dict[int, "NodeType"]] = None,
    ) -> Self:

        if memo is None:
            memo = {}

        new_graph = super().copy(new_graph, parent=parent, memo=memo)

        new_graph.compiled = self.compiled

        for key, value in self.call_counter.items():
            new_graph.call_counter[memo[key]] = value

        if new_graph.compiled:

            for module_path, list_of_nodes in self.interventions.items():

                new_graph.interventions[module_path] = [
                    new_graph.nodes[memo[index]].index for index in list_of_nodes
                ]

            for key, values in self.deferred.items():

                new_graph.deferred[memo[key]] = [memo[index] for index in values]

            new_graph.grad_subgraph = [memo[index] for index in self.grad_subgraph]

        return new_graph

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
