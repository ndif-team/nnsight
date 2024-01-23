from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Union, Optional

from pydantic import BaseModel


class ResultModel(BaseModel):
    id: str
    output: Any = None
    saves: Dict[str, Any] = None


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
    session_id: Optional[str] = None

    result: Optional[Union[bytes, ResultModel]] = None

    def __str__(self) -> str:
        return f"{self.id} - {self.status.name}: {self.description}"

    def log(self, logger: logging.Logger) -> ResponseModel:
        if self.status == ResponseModel.JobStatus.ERROR:
            logger.error(str(self))
        else:
            logger.info(str(self))

        return self
