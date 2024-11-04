from __future__ import annotations

import tempfile
from collections import defaultdict
from collections.abc import Iterable
from typing import (Any, Callable, Dict, Generic, Iterator, List, Optional,
                    Set, Type, TypeVar, Union, overload)

import torch
from PIL import Image as PILImage

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
    
    ######## Graph Visualization ###########################
    
    def visualize(self,
        title: str = "graph",
        path: str = ".",
        display: bool = True,
        save: bool = False,
        recursive: bool = False,
        group: bool = False,
    ):
        
        try:

            import pygraphviz as pgv

        except Exception as e:

            raise type(e)(
                "Visualization of the Graph requires `pygraphviz` which requires `graphviz` to be installed on your machine."
            ) from e
        
        if group and not recursive:
            print("Warning: set `recursive=True` to visualize all subgraphs and make use of the 'group' functionality.")
            group = False
        
        from IPython.display import Image
        from IPython.display import display as IDisplay

        graph: pgv.AGraph = pgv.AGraph(strict=True, directed=True)

        graph.graph_attr.update(
            label=title, fontsize="20", labelloc="t", labeljust="c"
        )

        def style_node(node):
            """
            
            """

            if isinstance(node.target, type) and issubclass(node.target, protocols.Protocol):
                return node.target.style()
            else:
                return {
                    "node": {"color": "black", "shape": "ellipse"},
                    "label": (node.target if isinstance(node.target, str) else node.target.__name__),
                    "arg": defaultdict(lambda: {"color": "gray", "shape": "box"}),
                    "arg_kname": defaultdict(lambda: None),
                    "edge": defaultdict(lambda: {"style": "solid"}),
                }
        
        subgraphs = {}
        subgraph_names: Dict[str, int] = defaultdict(lambda: 0)
        def get_subgraph(node):
            nonlocal subgraphs
            if group:
                if id(node.graph) != id(self):
                    if not id(node.graph) in subgraphs.keys():
                        subgraph = graph.subgraph(name=f"cluster_{id(node.graph)}")
                        subgraph.graph_attr['penwidth'] = 0.25
                        subgraphs[id(node.graph)] = subgraph

                    return subgraphs[id(node.graph)]
                else:
                    return graph
            else:
                return graph
        
        if recursive:
            nodes = [node for node in self.nodes if id(node.graph) >= id(self)]
        else:
            nodes = self

        visualized_nodes = set()
        for node in nodes:
            styles = style_node(node)

            subgraph = get_subgraph(node)

            subgraph.add_node(node.index, label=styles["label"], **styles["node"])
            visualized_nodes.add(node.index)

            for idx, arg in enumerate(node.args):
                if isinstance(arg, SubGraph):
                    name = f"{node.index}_{arg}_{idx}"
                    label = f"Subgraph"
                    
                    subgraph.add_node(name, label=label, **{"color": "purple", "shape": "box"})
                    
                    if recursive:
                        for sub_node in arg:
                            root_node: bool = True
                            for dep_idx in sub_node._dependencies:
                                root_node = root_node and (dep_idx not in arg.subset)
                                
                            if root_node:
                                graph.add_edge(node.index, sub_node.index, **{"style": "dashed", "color": styles["node"]["color"]})

                    if group:
                        subgraph_label = styles['label']
                        subgraphs[id(arg)].graph_attr['label'] = f"{subgraph_label}_{subgraph_names[subgraph_label]}"
                        subgraph_names[subgraph_label] += 1
                    
                elif isinstance(arg, Node):
                    name = arg.index
                    label = node.index

                    if arg.index not in visualized_nodes:
                        arg_label = (node.target if isinstance(node.target, str) else node.target.__name__)

                        subgraph.add_node(arg.index, label=arg_label, **{"color": "brown", "shape": "box"})

                        visualized_nodes.add(arg.index)
                else:
                    name = str(node.index)
                    if isinstance(arg, torch.Tensor):
                        name += f"_Tensor_{idx}"
                        label = "Tensor"
                    elif isinstance(arg, str):
                        name += f"_{arg}_{idx}"
                        label = f'"{arg}"'
                    else:
                        name += f"_{arg}_{idx}"
                        label = str(arg)

                    if not styles["arg_kname"][idx] is None:
                            label = f"{styles['arg_kname'][idx]}={label}"
                    else:
                        label = label

                    subgraph.add_node(name, label=label, **{"color": "gray", "shape": "box"}) # same

                    if isinstance(arg, Iterable):
                        for idx, element in enumerate(arg):
                            if isinstance(element, Node):
                                if element.index not in visualized_nodes:
                                    """ styles = (
                                        element.target.style() 
                                        if isinstance(element.target, type) and issubclass(element.target, protocols.Protocol) 
                                        else default_style(element)
                                    )

                                    styles["node"]["color"] = "brown"
                                    styles["node"]["shape"] = "box" """

                                    element_label = (element.target if isinstance(element.target, str) else element.target.__name__)
                                    subgraph.add_node(element.index, label=element_label, color="brown", shape="box")
                                    visualized_nodes.add(element.index)
                                graph.add_edge(element.index, name, style="dashed", color="gray", label=f"{idx}", fontsize=10)
                    
                subgraph.add_edge(name, node.index, **styles["edge"][idx])

        def display_graph(file_name):
            in_notebook = True

            # Credit: Till Hoffmann - https://stackoverflow.com/a/22424821
            try:
                from IPython import get_ipython

                if "IPKernelApp" not in get_ipython().config:
                    in_notebook = False
            except ImportError:
                in_notebook = False
            except AttributeError:
                in_notebook = False

            if in_notebook:
                IDisplay(Image(filename=file_name))
            else:
                img = PILImage.open(file_name)
                img.show()
                img.close()

        if not save:
            with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
                graph.draw(temp_file.name, prog="dot")
                if display:
                    display_graph(temp_file.name)
        else:
            graph.draw(f"{path}/{title}.png", prog="dot")
            if display:
                display_graph(f"{path}/{title}.png")

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
