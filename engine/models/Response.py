from enum import Enum
from typing import Any, Dict, Union

import torch
from pydantic import BaseModel
from transformers.generation.utils import GenerateOutput


class JobStatus(Enum):
    RECIEVED = "RECIEVED"
    APPROVED = "APPROVED"
    SUBMITTED = "SUBMITTED"
    COMPLETED = "COMPLETED"

    ERROR = "ERROR"


class ResponseModel(BaseModel):
    id: str
    status: JobStatus
    description: str

    output: Any = None
    copies: Dict[str, Any] = None
    blocking: bool = False
