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
    batched_input: Union[Any, bytes]
    intervention_graph: Union[Graph, bytes]
    generation: bool
    # Edits
    # altered

    id: str = None
    received: datetime = None
    blocking: bool = False

    @field_serializer("intervention_graph")
    def intervention_graph_serialize(self, value: Graph, _info) -> bytes:  
        value.compile(None)

        for node in value.nodes.values():

            node.proxy_value = None

        return pickle.dumps(value)
    
    @field_serializer("batched_input")
    def serialize(self, value, _info) -> bytes:
        return pickle.dumps(value)
