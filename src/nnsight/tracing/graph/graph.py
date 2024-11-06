from __future__ import annotations

from typing import (Callable, Generic, Iterator, List, Optional, Type, TypeVar,
                    Union, overload)

from ... import util
from .. import protocols
from . import Node, NodeType, Proxy, ProxyType


class Graph(Generic[NodeType, ProxyType]):

    def __init__(
        self,
        node_class: Type[NodeType] = Node,
        proxy_class: Type[ProxyType] = Proxy,
    ) -> None:

        self.node_class = node_class
        self.proxy_class = proxy_class

        self.alive = True

        self.nodes: List[Node] = []
        self.stack: List[Graph] = []
        self.defer_stack: List[int] = []

    def reset(self) -> None:

        for node in self:
            node.reset()

    def execute(self) -> None:

        self.alive = False

        exception = None

        for node in self:
            try:
                node.execute()
            except Exception as e:
                exception = (node.index, e)
                break

        if exception is not None:
            defer_stack = self.defer_stack
            self.defer_stack = []
            self.clean(exception[0])
            self.defer_stack = defer_stack
            raise exception[1]

    def clean(self, start: Optional[int] = None):

        if start is None:
            start = self[0].index

        end = self[-1].index + 1

        for index in range(start, end):

            node = self.nodes[index]
            
            node.update_dependencies()

    def create(
        self,
        target: Union[Callable, protocols.Protocol],
        *args,
        **kwargs,
    ) -> ProxyType:

        graph = self.stack[-1] if self.stack else self

        return self.proxy_class(
            self.node_class(target, *args, graph=graph, **kwargs)
        )

    def add(self, node: NodeType) -> None:

        # Tag the Node with its own index.
        node.index = len(self.nodes)

        # Add Node.
        self.nodes.append(node)

    def copy(self, new_graph: Graph[NodeType, ProxyType]):

        memo = {}

        def from_memo(arg: Union[Node, SubGraph]):
            
            if isinstance(arg, SubGraph):
                return arg.copy(SubGraph(new_graph))
            
            if arg.done:
                return arg.value

            return new_graph.nodes[memo[arg.index]]

        for node in self:

            new_node = new_graph.create(
                node.target,
                *util.apply(node.args, from_memo, (Node, SubGraph)),
                **util.apply(node.kwargs, from_memo, (Node, SubGraph)),
            ).node

            memo[node.index] = new_node.index

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

    def __init__(
        self,
        parent: GraphType,
        subset: Optional[List[int]] = None,
    ):

        self.__dict__.update(parent.__dict__)

        self.subset: List[int] = [] if subset is None else subset

    def add(self, node: NodeType) -> None:

        super().add(node)

        self.subset.append(self.nodes[-1].index)
        
    @overload
    def __getitem__(self, key: int) -> Node: ...

    @overload
    def __getitem__(self, key: Union[slice, List[int]]) -> List[Node]: ...

    def __getitem__(self, key: Union[int, Union[slice, List[int]]]) -> Union[Node, List[Node]]:
        
        index = self.subset[key]
        
        node = self.nodes[index] if isinstance(index, int) else [self.nodes[idx] for idx in index]

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


# class MultiGraph(Graph):


#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(proxy_class, validate)


GraphType = TypeVar("GraphType", bound=SubGraph)
