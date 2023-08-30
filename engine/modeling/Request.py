from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel, ConfigDict
from .fx import NodeModel


class RequestModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), arbitrary_types_allowed=True)

    args: List
    kwargs: Dict
    model_name: str
    prompts: List[str]
    intervention_graph: Dict[str,NodeModel]
    #Edits

    id: str = None
    recieved: datetime = None
    blocking: bool = False
