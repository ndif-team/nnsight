from __future__ import annotations

import inspect
from typing import (Any, Callable, Dict, Iterator, List, Optional, Set, Type,
                    Union, overload)
import weakref

from ... import util
from .. import protocols
from . import Node, Proxy


class Graph:

    def __init__(
        self,
        proxy_class: Type[Proxy] = Proxy,
        validate: bool = False,
    ) -> None:

        self.proxy_class = proxy_class
        self.validate = validate

        self.alive = True

        self.nodes: List[Node] = []
        self.stack:List[Graph] = []

        self.attachments = {}

    def reset(self) -> None:

        for node in self:
            node.reset()

    def execute(self, subgraph: Optional[Set[int]] = None) -> None:
        """Executes operations of `Graph`.

        Executes all `Node`s sequentially if `Graph.sequential`. Otherwise execute only root `Node`s sequentially.
        """

        self.alive = False

        flag = False

        if subgraph is None:
            start, end = 0, len(self) - 1
        else:
            listed = list(subgraph)
            start, end = listed[0], listed[-1]

        for index in range(start, end + 1):

            node = self.nodes[index]

            if node.executed:
                continue
            elif flag:
                if node.fulfilled and index in subgraph:
                    node.execute()
            elif node.fulfilled:
                node.execute()
            elif subgraph is None:
                break
            elif index not in subgraph:
                flag = True

            if end == index:
                break

    def create(
        self,
        target: Union[Callable, protocols.Protocol],
        *args,
        trace_value: Any = inspect._empty,
        **kwargs,
    ) -> Proxy:
        
        graph = self
        
         # There might be a more recent Graph open.
        if self.stack and self is not self.stack[-1]:
            graph = self.stack[-1]
        
        return self.proxy_class(
            Node(target, *args,  graph=graph, trace_value=trace_value, **kwargs)
        )

    def add(self, node: Node) -> None:
        
        # Tag the Node with its own index.
        node.index = len(self.nodes)
        
        # Add Node.
        self.nodes.append(node)

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

    def __getitem__(self, key: Any) -> Union[Node, List[Node]]:
        return self.nodes[key]

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def __getstate__(self) -> Dict:
        return {"id": self.id, "nodes": self.nodes, "name_idx": self.name_idx}

    def __setstate__(self, state: Dict) -> None:

        self.__dict__.update(state)


class SubGraph(Graph):

    def __init__(self, parent: Graph, subset: Optional[Set[int]] = None, **kwargs):

        super().__init__(**kwargs)

        self.nodes = parent.nodes
        self.stack = parent.stack
        
        self.subset: List[int] = []

    def add(self, node: Node) -> None:
        
        super().add(node)
        
        self.subset.append(self.nodes[-1].index)

    def reset(self) -> None:
        for index in self.subset:
            self.nodes[index].reset()

    def execute(self) -> None:
        return super().execute(subgraph=self.subset)
    
    def __getitem__(self, key: Any) -> Union[Node, List[Node]]:
        return self.nodes[self.subset[key]]

    def __iter__(self) -> Iterator[Node]:
        return iter([self.nodes[index] for index in self.subset])

    def __len__(self) -> int:
        return len(self.subset)