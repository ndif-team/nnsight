from __future__ import annotations
from typing import (Callable, Dict, Generic, Iterator, List, Optional,
                    Tuple, Type, TypeVar, Union, overload)

from typing_extensions import Self

from ... import util
from ...util import NNsightError
from .. import protocols
from . import Node, NodeType, Proxy, ProxyType


class Graph(Generic[NodeType, ProxyType]):
    """The `Graph` class represents a computation graph composed of individual `Node`s (operations).
    It contains logic to both trace/build the computation graph, as well as how to execute it.
    Sections of the graph can be divided into `SubGraphs`, but there will always be one root `Graph`.
    The final `Node` of the graph (graph[-1]) should be the root `Node` which when executed, downstream executes the entire `Graph`.

    Attributes:
        node_class (Type[NodeType]): Class used to create `Node`s. Can be changed to add additional functionality to `Node's. Defaults to `Node`.
        proxy_class (Type[ProxyType]): Class used to create `Proxy`s for 'Node's. Can be changed to add additional functionality to `Proxy's. Defaults to `Proxy`.
        nodes (List[Node]): Ordered list of all `Node`s. Used to access `Nodes` via their index.
        stack (List[Graph]): List of `Graph`s as a stack. Used to move `Node`s onto the most recent graph, as opposed to the `Graph` used to create the `Node`.
            Managed outside the `Graph` class by the `Context` objects.
        defer_stack (List[int]): List of `Node` indexes as a stack. Used to prevent destruction/memory cleanup of `Node`s whose index is less than the most recent index on the stack.
            This happens when you have `Node`s that will be executed more than once. In a loop for example, you only want to destroy a `Node`s dependencies on the final iteration.
            Also managed outside the `Graph` object.
        alive (bool): If the `Graph` is "alive". Alive meaning its still open for tracing (adding new `Node`s). Set to False before executing the `Graph`.
    """

    def __init__(
        self,
        node_class: Type[NodeType] = Node,
        proxy_class: Type[ProxyType] = Proxy,
        debug: bool = False,
    ) -> None:

        self.node_class = node_class
        self.proxy_class = proxy_class
        self.debug = debug

        self._alive = [True]

        self.nodes: List[Node] = []
        self.stack: List[Graph] = []
        self.defer_stack: List[int] = []

    @property
    def alive(self) -> bool:
        return self._alive[0]

    @alive.setter
    def alive(self, value: bool):

        self._alive[0] = value

    def reset(self) -> None:
        """Resets the `Graph` to prepare for execution.
        Simply resets all `Node`s in the `Graph`.
        """

        for node in self:
            node.reset()

    def execute(self) -> None:
        """Executes all `Node`s (operations) in this `Graph`.

        Raises:
            exception: If there is an exception during executing a `Node`. If so, we need to clean up the dependencies of `Node`s yet to be executed.
        """

        err: Tuple[int, NNsightError] = None

        for node in self:
            try:
                node.execute()
            except NNsightError as e:
                err = (node.index, e)
                break

        if err is not None:
            defer_stack = self.defer_stack.copy()
            self.defer_stack.clear()
            self.clean(err[0])
            self.defer_stack.extend(defer_stack)
            raise err[1]

    def clean(self, start: Optional[int] = None):
        """Cleans up dependencies of `Node`s so their values are appropriately memory managed.
        Cleans all `Node`s from start to end regardless if they are on this `Graph`.

        Args:
            start (Optional[int], optional): `Node` index to start cleaning up from. Defaults to None.
        """
        
        if len(self) == 0:
            return

        if start is None:
            start = self[0].index

        end = self[-1].index + 1

        # Loop over ALL nodes within the span of this graph.
        for index in range(start, end):
            
            node = self.nodes[index]

            node.update_dependencies()

    def create(
        self,
        target: Union[Callable, protocols.Protocol],
        *args,
        redirect: bool = True,
        **kwargs,
    ) -> ProxyType:
        """Creates a new `Node` using this `Graph`'s node_class and returns a `Proxy` for it with this `Graph`'s proxy_class.

        Args:
            target (Union[Callable, protocols.Protocol]): Target for the new `Node`.
            redirect (bool, optional): If to move the newly created `Node` to the most recent `Graph` on the Graph.stack. Defaults to True.

        Returns:
            ProxyType: `Proxy` for newly created `Node`.
        """

        # Redirection.
        graph = self.stack[-1] if redirect and self.stack else self

        return self.proxy_class(self.node_class(target, *args, graph=graph, **kwargs))

    def add(self, node: NodeType) -> None:
        """Adds a `Node` to this `Graph`.
        Sets the `Node`'s .index attribute so it knows its own index within the entire computation graph.

        Args:
            node (NodeType): `Node` to add.
        """

        # Tag the Node with its own index.
        node.index = len(self.nodes)

        # Add Node.
        self.nodes.append(node)

    def copy(self, new_graph: Optional[Graph[NodeType, ProxyType]] = None) -> Graph:
        """Creates a shallow copy of the root `Graph` object.

        Args:
            new_graph (Optional[Graph[NodeType, ProxyType]], optional): `Graph` to copy into. Defaults to None and creates a new `Graph`.

        Returns:
            Graph: New `Graph`.
        """

        if new_graph is None:
            new_graph = Graph(node_class=self.node_class, proxy_class=self.proxy_class, debug=self.debug)
        
        node = self[-1]

        def process(arg: Union[Node, SubGraph]):

            if isinstance(arg, SubGraph):
                return arg.copy(parent=new_graph)

            if arg.done:
                return arg.value

        new_graph.create(
            node.target,
            *util.apply(node.args, process, (Node, SubGraph)),
            **util.apply(node.kwargs, process, (Node, SubGraph)),
        )

        return new_graph

    ### Magic Methods ######################################

    def __str__(self) -> str:
        result = f"{self.__class__.__name__}:\n"

        for node in self:
            result += f"  {str(node)}\n"

        return result

    @overload
    def __getitem__(self, key: int) -> Node: ...

    @overload
    def __getitem__(self, key: Union[slice, List[int]]) -> List[Node]: ...

    def __getitem__(self, key: Union[int, Union[slice, List[int]]]) -> Union[Node, List[Node]]:
        return self.nodes[key]

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)


class SubGraph(Graph[NodeType, ProxyType]):
    """Represents a slice of the greater computation graph. It has a reference to the same underlying list of nodes and simply maintains a subset of node indexes.

    Attributes:
        subset (List[int]): Node indexes for `Node`s contained within this subgraph.
    """

    def __init__(
        self,
        parent: GraphType,
        subset: Optional[List[int]] = None,
    ):
        """Init

        Args:
            parent (GraphType): Graph to inherit attributes from.
            subset (Optional[List[int]], optional): Subset to start from when loading a pre-defined `SubGraph`
        """

        self.__dict__.update(parent.__dict__)

        self.subset: List[int] = [] if subset is None else subset

    def __getstate__(self):

        return {
            "nodes":self.nodes,
            "subset":self.subset,
            "defer_stack": self.defer_stack,
        }
    
    def __setstate__(self, state: Dict) -> None:

        self.__dict__.update(state)

    def add(self, node: NodeType) -> None:

        super().add(node)

        # Also add the index to this SubGraph's subset upon adding.
        self.subset.append(self.nodes[-1].index)

    @overload
    def __getitem__(self, key: int) -> Node: ...

    @overload
    def __getitem__(self, key: Union[slice, List[int]]) -> List[Node]: ...

    def __getitem__(self, key: Union[int, Union[slice, List[int]]]) -> Union[Node, List[Node]]:        

        index = self.subset[key]

        # We iterate over indexes and get their Nodes.
        node = (
            [self.nodes[idx] for idx in index]
            if isinstance(index, list)
            else self.nodes[index]
        )

        return node

    def __iter__(self) -> Iterator[Node]:
        return self.Iterator(self)

    def __len__(self) -> int:
        return len(self.subset)

    class Iterator(Iterator):

        def __init__(self, subgraph: SubGraph[GraphType]) -> None:

            self.subgraph = subgraph
            self.start = 0
            self.end = len(self.subgraph)

        def __next__(self) -> NodeType:

            if self.start < self.end:
                value = self.subgraph[self.start]
                self.start += 1
                return value

            raise StopIteration

    def copy(
        self,
        new_graph: Optional[SubGraph[NodeType, ProxyType]] = None,
        parent: Optional[Graph[NodeType, ProxyType]] = None,
        memo: Optional[Dict[int, NodeType]] = None,
    ) -> Self:
        """Creates a shallow copy of this SubGraph.

        Args:
            new_graph (Optional[SubGraph[NodeType, ProxyType]], optional): SubGraph to copy into. Defaults to None and creates a new SubGraph of the same type.
            parent (Optional[Graph[NodeType, ProxyType]], optional): Parent graph. Defaults to None and will create a root `Graph` as the parent.

        Returns:
            Self: New graph.
        """
        if parent is None:
            parent = Graph(node_class=self.node_class, proxy_class=self.proxy_class)

        if new_graph is None:
            new_graph = type(self)(parent)

        if memo is None:
            memo = {}

        def process(arg: Union[Node, SubGraph]):

            if isinstance(arg, SubGraph):
                return arg.copy(parent=new_graph, memo=memo)

            if arg.done:
                return arg.value

            return new_graph.nodes[memo[arg.index]]

        for node in self:

            new_node = new_graph.create(
                node.target,
                *util.apply(node.args, process, (Node, SubGraph)),
                **util.apply(node.kwargs, process, (Node, SubGraph)),
            ).node

            memo[node.index] = new_node.index

        return new_graph


# class MultiGraph(Graph):


#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(proxy_class, validate)


GraphType = TypeVar("GraphType", bound=SubGraph)
