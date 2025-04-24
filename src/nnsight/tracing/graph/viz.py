import tempfile
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Dict


import torch
from PIL import Image as PILImage

from ..protocols import Protocol
from . import Node, SubGraph

if TYPE_CHECKING:
    from . import Graph


def viz_graph(
    graph: "Graph",
    title: str = "graph",
    display: bool = True,
    save: bool = False,
    path: str = ".",
    recursive: bool = False,
    group: bool = False,
    ) -> None:
    """
    Utility funciton to visualize the NNsight Graph structures built during tracing.

    Args:
        - graph (Graph): NNsight Graph to be visualized.
        - title (str): Title given to the visualization. Default: "graph".
        - display (bool): Displays the rendered graph visualization. Default: True.
        - save (bool): Saves the rendered graph to a file with the title as the name. Default: False.
        - path (str): Path to store the saved visualization. Default: ".".
        - recursive (bool): Recursively visualize all the inner Subgraphs of a given Graph. Default: False.
        - group (bool): Visually group all the nodes belonging to the same Subgraph together. Default: False.
    """

    from ..contexts.globals import GlobalTracingContext

    with GlobalTracingContext.exit_global_tracing_context():
        
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

            if isinstance(node.target, type) and issubclass(node.target, Protocol):
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
                        arg_label = (arg.target if isinstance(arg.target, str) else arg.target.__name__)

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
