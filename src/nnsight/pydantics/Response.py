from __future__ import annotations

import logging
import pickle
from datetime import datetime
from enum import Enum
from typing import Any, Union

from pydantic import BaseModel, field_validator


class ResponseModel(BaseModel):
    class JobStatus(Enum):
        RECEIVED = "RECEIVED"
        APPROVED = "APPROVED"
        SUBMITTED = "SUBMITTED"
        COMPLETED = "COMPLETED"
        ERROR = "ERROR"

    id: str
    status: JobStatus
    description: str

    received: datetime = None
    saves: Union[bytes, Any] = None
    output: Union[bytes, Any] = None
    session_id: str = None
    blocking: bool = False

    def __str__(self) -> str:
        return f"{self.id} - {self.status.name}: {self.description}"

    def log(self, logger: logging.Logger) -> ResponseModel:
        if self.status == ResponseModel.JobStatus.ERROR:
            logger.error(str(self))
        else:
            logger.info(str(self))

        return self

    @field_validator("output", "saves")
    @classmethod
    def unpickle(cls, value):
        return pickle.loads(value)
