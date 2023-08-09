from datetime import datetime
from typing import Dict, List

from pydantic import BaseModel, ConfigDict


class PromiseModel(BaseModel):
    command: str
    id: str
    args: List


class RequestModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    args: List
    kwargs: Dict
    model_name: str
    execution_graphs: List[List[str]]
    promises: Dict[str, PromiseModel]
    prompts: List[str]

    id: str = None
    recieved: datetime = None
    blocking: bool = False
