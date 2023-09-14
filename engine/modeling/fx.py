from __future__ import annotations

from typing import Any, Callable, Dict, List, Union

from pydantic import BaseModel

from .. import util
from ..fx.Graph import Graph
from ..fx.Node import Node


class NodeModel(BaseModel):
    class Reference(BaseModel):
        name: str

    name: str
    target: Union[Callable, str]
    args: List[Any]
    kwargs: Dict[str, Any]

    @staticmethod
    def from_node(node: Node):
        def _reference(node: Node):
            return NodeModel.Reference(name=node.name)

        args = util.apply(node.args, _reference, Node)
        kwargs = util.apply(node.kwargs, _reference, Node)

        return NodeModel(name=node.name, target=node.target, args=args, kwargs=kwargs)

    @staticmethod
    def to_node(graph: Graph, nodes: Dict[str, NodeModel], node_model: NodeModel):
        def _dereference(reference: NodeModel.Reference):
            return NodeModel.to_node(graph, nodes, nodes[reference.name])

        # Arguments might be interventions themselves so recurse.
        args = util.apply(node_model.args, _dereference, NodeModel.Reference)
        kwargs = util.apply(node_model.kwargs, _dereference, NodeModel.Reference)

        # Processing of args may have already created an Intervention for this node so just return it.
        if node_model.name in graph.nodes:
            return graph.nodes[node_model.name]

        graph.add(
            graph=graph,
            value=None,
            target=node_model.target,
            args=args,
            kwargs=kwargs,
            name=node_model.name,
        )
