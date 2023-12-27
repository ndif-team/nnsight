from __future__ import annotations

import logging
import pickle
from datetime import datetime
from enum import Enum
from typing import Any, Dict

import requests
from pydantic import BaseModel


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

    output: Any = None
    received: datetime = None
    saves: Dict[str, Any] = None
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

    def update_backend(self, client) -> ResponseModel:
        responses_collection = client["ndif_database"]["responses"]

        from bson.objectid import ObjectId

        responses_collection.replace_one(
            {"_id": ObjectId(self.id)}, {"bytes": pickle.dumps(self)}, upsert=True
        )

        return self

    def blocking_response(self, api_url: str) -> ResponseModel:
        if self.blocking:
            requests.get(f"{api_url}/blocking_response/{self.id}")

        return self
