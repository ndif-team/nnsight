from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel, ConfigDict

from ..Intervention import InterventionTree


class RequestModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), arbitrary_types_allowed=True)

    args: List
    kwargs: Dict
    model_name: str
    prompts: List[str]
    interventions_tree: InterventionTree

    id: str = None
    recieved: datetime = None
    blocking: bool = False
