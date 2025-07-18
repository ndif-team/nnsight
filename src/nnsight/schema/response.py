from __future__ import annotations

import io
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch
from pydantic import BaseModel, ConfigDict

RESULT = Dict[str, Any]

class ResponseModel(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    class JobStatus(Enum):
        RECEIVED = "RECEIVED"
        QUEUED = "QUEUED"
        DISPATCHED = "DISPATCHED"
        RUNNING = "RUNNING"
        COMPLETED = "COMPLETED"
        LOG = "LOG"
        STREAM = "STREAM"
        ERROR = "ERROR"
        NNSIGHT_ERROR = "NNSIGHT_ERROR"

    id: str
    status: ResponseModel.JobStatus

    description: Optional[str] = ""
    data: Optional[Union[RESULT, Any]] = None
    session_id: Optional[str] = None

    def __str__(self) -> str:
        return f"[{self.id}] {self.status.name.ljust(10)} : {self.description}"

    def log(self, logger: logging.Logger) -> ResponseModel:
       
        if self.status == ResponseModel.JobStatus.STREAM:
            pass
        else:
            logger.info(str(self))

        return self

    def pickle(self) -> bytes:
        """Pickles self and returns bytes.

        Returns:
            bytes: Pickled ResponseModel
        """

        with io.BytesIO() as file:

            torch.save(self.model_dump(exclude_unset=True), file)

            file.seek(0)

            return file.read()

    @classmethod
    def unpickle(cls, data: bytes) -> ResponseModel:
        """Loads a ResponseModel from pickled bytes.

        Args:
            data (bytes): Pickled ResponseModel.

        Returns:
            ResponseModel: Response.
        """

        with io.BytesIO(data) as file:
            return ResponseModel(
                **torch.load(file, map_location="cpu", weights_only=False)
            )
            
ResponseModel.model_rebuild()