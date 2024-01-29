from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Union

from pydantic import BaseModel, ConfigDict

from ..tracing.Graph import Graph
from .format import types
from .format.types import *


class RequestModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    args: List[types.ValueTypes]
    kwargs: Dict[str, types.ValueTypes]
    repo_id: str
    batched_input: types.ValueTypes
    intervention_graph: Union[Dict[str, Union[types.NodeType, types.NodeModel]], Graph]
    generation: bool

    id: str = None
    session_id: str = None
    received: datetime = None
    include_output: bool = False

    def compile(self) -> RequestModel:
        graph = Graph(None, validate=False)

        for node in self.intervention_graph.values():
            node.compile(graph, self.intervention_graph)

        self.intervention_graph = graph

        self.batched_input = self.batched_input.compile(None, None)

        self.args = [arg.compile(None, None) for arg in self.args]
        self.kwargs = {
            key: value.compile(None, None) for key, value in self.kwargs.items()
        }

        return self
