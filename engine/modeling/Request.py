from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Callable, Union, Any

from pydantic import BaseModel, ConfigDict
import torch.fx
from .. import util


class InterventionModel(BaseModel):
    class Reference(BaseModel):
        name: str

    name: str
    operation: str
    target: Union[Callable, str]
    args: List[Any]
    kwargs: Dict[str, Any]
    dependencies: List[InterventionModel.Reference]
    listeners: List[InterventionModel.Reference]

    @staticmethod
    def from_graph(graph: torch.fx.graph.Graph):
        interventions = dict()

        for node in graph.nodes:
            intervention = InterventionModel.from_node(node)
            interventions[intervention.name] = intervention

        return interventions

    @staticmethod
    def from_node(node: torch.fx.node.Node):
        def _reference(node: torch.fx.node.Node):
            return InterventionModel.Reference(name=node.name)

        args = util.apply(node.args, _reference, torch.fx.Node)
        kwargs = util.apply(node.kwargs, _reference, torch.fx.Node)
        dependencies = util.apply(
            list(node._input_nodes.keys()), _reference, torch.fx.Node
        )
        listeners = util.apply(list(node.users.keys()), _reference, torch.fx.Node)

        return InterventionModel(
            name=node.name,
            operation=node.op,
            target=node.target,
            args=args,
            kwargs=kwargs,
            dependencies=dependencies,
            listeners=listeners,
        )


class RequestModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), arbitrary_types_allowed=True)

    args: List
    kwargs: Dict
    model_name: str
    prompts: List[str]
    interventions: Dict[str, InterventionModel]

    id: str = None
    recieved: datetime = None
    blocking: bool = False
