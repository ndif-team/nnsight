"""Module for utility functions and classes used throughout the package."""

import importlib
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from contextlib import AbstractContextManager
from typing import (TYPE_CHECKING, Any, Callable, Collection, Dict, List,
                    Optional, Type, TypeVar)

import torch
from PIL import Image as PILImage
from typing_extensions import Self

from .tracing import protocols
from .tracing.graph import Node, SubGraph

if TYPE_CHECKING:
    from .tracing.graph import Graph

# TODO Have an Exception you can raise to stop apply early

T = TypeVar("T")
C = TypeVar("T", bound=Collection[T])


def apply(
    data: C, fn: Callable[[T], Any], cls: Type[T], inplace: bool = False
) -> C:
    """Applies some function to all members of a collection of a give type (or types)

    Args:
        data (Any): Collection of data to apply function to.
        fn (Callable): Function to apply.
        cls (type): Type or Types to apply function to.
        inplace (bool): If to apply the fn inplace. (For lists and dicts)

    Returns:
        Any: Same kind of collection as data, after then fn has been applied to members of given type.
    """

    if isinstance(data, cls):
        return fn(data)

    data_type = type(data)

    if data_type == list:
        if inplace:
            for idx, _data in enumerate(data):
                data[idx] = apply(_data, fn, cls, inplace=inplace)
            return data
        return [apply(_data, fn, cls, inplace=inplace) for _data in data]

    elif data_type == tuple:
        return tuple([apply(_data, fn, cls, inplace=inplace) for _data in data])

    elif data_type == dict:
        if inplace:
            for key, value in data.items():
                data[key] = apply(value, fn, cls, inplace=inplace)
            return data
        return {
            key: apply(value, fn, cls, inplace=inplace)
            for key, value in data.items()
        }

    elif data_type == slice:
        return slice(
            apply(data.start, fn, cls, inplace=inplace),
            apply(data.stop, fn, cls, inplace=inplace),
            apply(data.step, fn, cls, inplace=inplace),
        )

    return data

def fetch_attr(object: object, target: str) -> Any:
    """Retrieves an attribute from an object hierarchy given an attribute path. Levels are separated by '.' e.x (transformer.h.1)

    Args:
        object (object): Root object to get attribute from.
        target (str): Attribute path as '.' separated string.

    Returns:
        Any: Fetched attribute.
    """
    if target == "":
        return object

    target_atoms = target.split(".")

    for atom in target_atoms:

        if not atom:
            continue

        object = getattr(object, atom)

    return object

def to_import_path(type: type) -> str:

    return f"{type.__module__}.{type.__name__}"


def from_import_path(import_path: str) -> type:

    *import_path, classname = import_path.split(".")
    import_path = ".".join(import_path)

    return getattr(importlib.import_module(import_path), classname)


def viz_graph(
        graph: "Graph",
        title: str = "graph",
        path: str = ".",
        display: bool = True,
        save: bool = False,
        recursive: bool = False,
        group: bool = False,
    ) -> None:
        
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

    graph_viz: pgv.AGraph = pgv.AGraph(strict=True, directed=True)

    graph_viz.graph_attr.update(
        label=title, fontsize="20", labelloc="t", labeljust="c"
    )

    def style_node(node: Node) -> Dict:
        """Gets the style of the node based on it's target.
        If the target is a Protocol, then it gets the style directly from the protocol class.

        Args:
            - node (Node): node.

        Returns:
            - Dict: dictionary style.
        
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
    
    subgraphs: Dict[int, pgv.AGraph] = {}
    subgraph_names_count: Dict[str, int] = defaultdict(lambda: 0)
    def get_subgraph(node: Node) -> pgv.AGraph:
        """ Returns the Graph Visualization Object where this node should be rendered.

        Args:
            - node (Node: )
        

        Returns:
            - pgv.AGraph: Graph Visualization Object.
        """

        nonlocal subgraphs
        if group:
            if id(node.graph) != id(graph):
                if not id(node.graph) in subgraphs.keys():
                    subgraph = graph_viz.subgraph(name=f"cluster_{id(node.graph)}")
                    subgraph.graph_attr['penwidth'] = 0.25
                    subgraphs[id(node.graph)] = subgraph

                return subgraphs[id(node.graph)]
            else:
                return graph_viz
        else:
            return graph_viz
    
    if recursive:
        nodes = [node for node in graph.nodes if id(node.graph) >= id(graph)]
    else:
        nodes = graph

    visualized_nodes = set()
    for node in nodes:

        styles: Dict = style_node(node)

        subgraph: pgv.AGraph = get_subgraph(node)

        subgraph.add_node(node.index, label=styles["label"], **styles["node"])
        visualized_nodes.add(node.index)

        for idx, arg in enumerate(node.args):
            if isinstance(arg, SubGraph):
                name: str = f"{node.index}_{arg}_{idx}"
                label: str = f"Subgraph"
                
                subgraph.add_node(name, label=label, **{"color": "purple", "shape": "box"})
                
                if recursive:
                    for sub_node in arg:
                        root_node: bool = True
                        for dep_idx in sub_node._dependencies:
                            root_node = root_node and (dep_idx not in arg.subset)
                            
                        if root_node:
                            graph_viz.add_edge(node.index, sub_node.index, **{"style": "dashed", "color": styles["node"]["color"]})

                if group:
                    subgraph_label: str = styles['label']
                    subgraphs[id(arg)].graph_attr['label'] = f"{subgraph_label}_{subgraph_names_count[subgraph_label]}"
                    subgraph_names_count[subgraph_label] += 1
                
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

                subgraph.add_node(name, label=label, **{"color": "gray", "shape": "box"})

                if isinstance(arg, Iterable):
                    for idx, element in enumerate(arg):
                        if isinstance(element, Node):
                            if element.index not in visualized_nodes:

                                element_label = (element.target if isinstance(element.target, str) else element.target.__name__)
                                subgraph.add_node(element.index, label=element_label, color="brown", shape="box")
                                visualized_nodes.add(element.index)
                                
                            graph_viz.add_edge(element.index, name, style="dashed", color="gray", label=f"{idx}", fontsize=10)
                
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
            graph_viz.draw(temp_file.name, prog="dot")
            if display:
                display_graph(temp_file.name)
    else:
        graph_viz.draw(f"{path}/{title}.png", prog="dot")
        if display:
            display_graph(f"{path}/{title}.png")


class Patch:
    """Class representing a replacement of an attribute on a module.

    Attributes:
        obj (Any): Object to replace.
        replacement (Any): Object that replaces.
        parent (Any): Module or class to replace attribute.
    """

    def __init__(self, parent: Any, replacement: Any, key: str) -> None:
        self.parent = parent
        self.replacement = replacement
        self.key = key
        self.orig = getattr(self.parent, key)

    def patch(self) -> None:
        """Carries out the replacement of an object in a module/class."""
        setattr(self.parent, self.key, self.replacement)

    def restore(self) -> None:
        """Carries out the restoration of the original object on the objects module/class."""

        setattr(self.parent, self.key, self.orig)

class Patcher(AbstractContextManager):
    """Context manager that patches from a list of Patches on __enter__ and restores the patch on __exit__.

    Attributes:
        patches (List[Patch]):
    """

    def __init__(self, patches: Optional[List[Patch]] = None) -> None:
        self.patches = patches or []
        
        self.entered = False

    def add(self, patch: Patch) -> None:
        """Adds a Patch to the patches. Also calls `.patch()` on the Patch.

        Args:
            patch (Patch): Patch to add.
        """
        self.patches.append(patch)

        if self.entered:
            patch.patch()

    def __enter__(self) -> Self:
        """Enters the patching context. Calls `.patch()` on all patches.

        Returns:
            Patcher: Patcher
        """
        
        self.entered = True
        
        for patch in self.patches:
            patch.patch()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Calls `.restore()` on all patches."""
        self.entered = False
        for patch in self.patches:
            patch.restore()


class WrapperModule(torch.nn.Module):
    """Simple torch module which passes it's input through. Useful for hooking.
    If there is only one argument, returns the first element.
    """

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            args = args[0]

        return args

