from __future__ import annotations

import pickle
from datetime import datetime
from typing import Dict, List, Type, Union, Any

from pydantic import BaseModel, ConfigDict, field_serializer

from ..tracing.Graph import Graph
from .tracing import NodeModel


class RequestModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), arbitrary_types_allowed=True)

    args: List
    kwargs: Dict
    model_name: str
    batched_input: Union[List[Any], bytes]
    intervention_graph: Union[Graph, bytes, Dict[str, NodeModel]]
    generation: bool
    # Edits
    # altered

    id: str = None
    received: datetime = None
    blocking: bool = False

    @field_serializer("intervention_graph")
    def intervention_graph_serialize(self, value: Union[str, Graph], _info) -> str:
        if isinstance(value, Graph):
            nodes = dict()

            for node in value.nodes.values():
                node = NodeModel.from_node(node)
                nodes[node.name] = node

            value = nodes

        return pickle.dumps(value)
    
    @field_serializer("batched_input")
    def serialize(self, value, _info) -> str:
        return pickle.dumps(value)

    def graph(self):
        graph = Graph(None)

        for node in self.intervention_graph.values():
            NodeModel.to_node(graph, self.intervention_graph, node)

        return graph
