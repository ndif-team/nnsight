from enum import Enum
from typing import Any, Dict

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

    def __str__(self) -> str:
        return f"{self.id} - {self.status.name}: {self.description}"
