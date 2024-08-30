from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch
from pydantic import BaseModel

from .. import util
from ..tracing.Graph import Graph


class ResultModel(BaseModel):
    id: str
    value: Any = None

    @classmethod
    def from_graph(cls, graph: Graph) -> Dict[str, Any]:

        saves = {
            name: util.apply(node.value, lambda x: x.detach().cpu(), torch.Tensor)
            for name, node in graph.nodes.items()
            if node.done()
        }

        return saves
 

class ResponseModel(BaseModel):
    class JobStatus(Enum):
        RECEIVED = "RECEIVED"
        APPROVED = "APPROVED"
        RUNNING = "RUNNING"
        COMPLETED = "COMPLETED"
        LOG = "LOG"
        ERROR = "ERROR"

    id: str
    status: JobStatus
    description: str

    received: datetime = None
    session_id: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.id} - {self.status.name}: {self.description}"

    def log(self, logger: logging.Logger) -> ResponseModel:
        if self.status == ResponseModel.JobStatus.ERROR:
            logger.error(str(self))
        else:
            logger.info(str(self))

        return self
