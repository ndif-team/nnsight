from __future__ import annotations

import io
import zstandard as zstd
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, ConfigDict
from ..intervention.serialization import save, load

if TYPE_CHECKING:
    from ..intervention.tracing.tracer import Tracer
else:
    Tracer = Any


class RequestModel(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    interventions: Callable
    tracer: Tracer

    def serialize(self, compress: bool = False) -> bytes:

        self.interventions.__source__ = "".join(self.tracer.info.source)

        data = save(self)

        if compress:

            data = zstd.ZstdCompressor(level=6).compress(data)

        return data

    @staticmethod
    def deserialize(
        request: bytes, persistent_objects: dict = None, compress: bool = False
    ) -> RequestModel:

        if compress:

            request = zstd.ZstdDecompressor().decompress(request)

        with io.BytesIO(request) as data:

            data.seek(0)

            request: RequestModel = load(data.read(), persistent_objects)

        return request


RequestModel.model_rebuild()
