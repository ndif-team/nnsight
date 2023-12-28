import pickle
from datetime import datetime
from typing import Any, Dict, List, Union

from pydantic import BaseModel, field_serializer


class RequestModel(BaseModel):
    args: List
    kwargs: Dict
    repo_id: str
    batched_input: Union[bytes, Any]
    intervention_graph: Union[bytes, Any]
    generation: bool

    id: str = None
    session_id: str = None
    received: datetime = None
    blocking: bool = False

    @field_serializer("intervention_graph")
    def intervention_graph_serialize(self, value, _info) -> bytes:
        value.compile(None)

        for node in value.nodes.values():
            node.proxy_value = None

        return pickle.dumps(value)

    @field_serializer("batched_input")
    def serialize(self, value, _info) -> bytes:
        return pickle.dumps(value)
