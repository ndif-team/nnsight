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
import warnings
from enum import Enum
from typing import Any, Dict, Optional

import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

RESULT = Dict[str, Any]


class MetaData(BaseModel):
    """What a remote job cost the server, reported on the response that ends it.

    The measurements NDIF takes around a run, shaped for the client. Present on
    ``COMPLETED`` and on a failure alike -- a job that timed out or ran out of
    memory is exactly when its cost is worth reading.

    Attributes:
        runtime: Wall-clock seconds the block ran on the server.
        max_memory_usage: Peak bytes on the worst-pressured single device.
        max_mem_by_gpu: Bytes the request drove *above the resident weights*,
            per device. Not the card's total usage -- the weights are the
            server's, not the request's, and you cannot shrink them.
        max_mem_pct_by_gpu: ``max_mem_by_gpu`` against the headroom the request
            actually had (its share of the card, less what the weights already
            hold), as a percentage. 100 means it filled everything left for it.
        alloc_shortfall_by_gpu: Of the allocation the server refused, the bytes that would
            not fit -- how much to free for the block to run -- per card that
            ran out. Not the size of the refused allocation itself: asking for
            2 GB with 1.9 GB free and asking for it with nothing free are the
            same request and completely different problems. Set **only** on an
            out-of-memory failure, so ``None`` is itself the answer to "did this
            run out of memory". Approximate: the allocator frees cached blocks
            and retries before giving up.

    Every field is optional, and unknown ones are kept (``extra="allow"``): a
    server may report a measurement this client has never heard of, and it stays
    readable as an attribute rather than being dropped. So check a field before
    trusting it -- an older server may send none of them.

    GPU maps are keyed by device id **as a string**. A response reaches the
    client as JSON or as ``torch.save`` bytes, and only JSON stringifies dict
    keys; declaring ``str`` here means a server that sends integers fails loudly
    at the boundary instead of handing the two encodings different shapes.
    """

    model_config = ConfigDict(extra="allow")

    runtime: Optional[float] = None
    max_memory_usage: Optional[int] = None
    max_mem_by_gpu: Dict[str, int] = Field(default_factory=dict)
    max_mem_pct_by_gpu: Dict[str, float] = Field(default_factory=dict)
    alloc_shortfall_by_gpu: Optional[Dict[str, int]] = None


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
    what the run cost on the server -- on ``COMPLETED`` and on a failure alike,
    since a job that ran out of memory or timed out is exactly when its cost is
    worth reading.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    id: str
    status: Status
    description: str = ""
    data: Optional[Any] = None
    meta_data: Optional[MetaData] = None

    @field_validator("meta_data", mode="before")
    @classmethod
    def _drop_an_unreadable_report(cls, value: Any) -> Any:
        """Never let the cost report fail the response carrying it.

        ``meta_data`` is diagnostic; ``data`` is the job. A server that sends a
        malformed report would otherwise raise here and turn a run that finished
        perfectly well into a client-side crash, which is a far worse outcome
        than not knowing what it cost. So an unreadable report is dropped.

        Dropped *loudly*. Every field is optional and unknown ones are kept, so
        the only way to land here is a known field with the wrong type -- a
        server bug or a version mismatch, which is worth hearing about. Silence
        would also make this indistinguishable from an older server that reports
        nothing at all, and those want different responses from whoever hits it.
        """
        if value is None or isinstance(value, MetaData):
            return value
        try:
            return MetaData.model_validate(value)
        except ValidationError as error:
            warnings.warn(
                f"Discarded an unreadable cost report from the server; the "
                f"job itself is unaffected. {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

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
