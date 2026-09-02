"""The status message a remote job sends back to the client.

Each update NDIF pushes for a job — whether streamed over a blocking websocket or
saved to the object store and polled — arrives as a [`ResponseModel`][nnsight.schema.response.ResponseModel]. Its
`Status` names where the job is in its lifecycle; on ``COMPLETED`` the
returned values ride in [`data`][nnsight.schema.response.ResponseModel.data]
and what the run cost the server rides in
[`meta_data`][nnsight.schema.response.ResponseModel.meta_data].
"""

from __future__ import annotations

import io
from enum import Enum
from typing import Any, Dict, Optional

import torch
from pydantic import BaseModel, ConfigDict

RESULT = Dict[str, Any]


class Status(str, Enum):
    """Where a remote job is in its lifecycle (or a transient log message)."""

    RECEIVED = "RECEIVED"
    QUEUED = "QUEUED"
    PROVISIONING = "PROVISIONING"
    DEPLOYING = "DEPLOYING"
    DISPATCHED = "DISPATCHED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    LOG = "LOG"  # a transient server message, not a lifecycle stage


class ResponseModel(BaseModel):
    """One status update for a remote job, optionally carrying its result.

    Streamed over the websocket for a blocking job, or saved to the object store
    and fetched by the client for a non-blocking one. [`data`][nnsight.schema.response.ResponseModel.data] holds the
    saved values only on a ``COMPLETED`` response, and [`meta_data`][nnsight.schema.response.ResponseModel.meta_data]
    what the run cost on the server -- also COMPLETED-only.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    id: str
    status: Status
    description: str = ""
    data: Optional[Any] = None
    meta_data: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return f"[{self.id}] {self.status.value.ljust(12)} {self.description}"

    def pickle(self) -> bytes:
        """Serialize to bytes via ``torch.save`` (carries tensors in ``data``)."""
        with io.BytesIO() as file:
            torch.save(self.model_dump(exclude_unset=True), file)
            return file.getvalue()

    @classmethod
    def unpickle(cls, data: bytes) -> ResponseModel:
        """Rebuild a [`ResponseModel`][nnsight.schema.response.ResponseModel] from [`pickle`][nnsight.schema.response.ResponseModel.pickle] bytes (onto CPU)."""
        with io.BytesIO(data) as file:
            return cls(**torch.load(file, map_location="cpu", weights_only=False))
