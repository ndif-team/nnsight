from __future__ import annotations

import json
import zlib
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Union

import msgspec
from pydantic import BaseModel, ConfigDict, TypeAdapter, field_serializer

from .. import NNsight
from .format.types import *

if TYPE_CHECKING:
    from ..contexts.backends.RemoteBackend import RemoteMixin

OBJECT_TYPES = Union[SessionType, TracerType, SessionModel, TracerModel]


class RequestModel(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    object: str | bytes | OBJECT_TYPES
    msgspec: bool = False
    zlib: bool = False

    model_key: str

    id: str = None
    received: datetime = None

    session_id: Optional[str] = None

    @field_serializer("object")
    def serialize_object(
        self, object: Union[SessionType, TracerType, SessionModel, TracerModel]
    ) -> str:

        if isinstance(object, (str, bytes)):
            return object

        self.msgspec = True
        self.zlib = True

        data = msgspec.json.encode(object)
        data = zlib.compress(data)

        return data

    def deserialize(self, model: NNsight) -> "RemoteMixin":

        handler = DeserializeHandler(model=model)

        object = self.object

        if self.zlib:
            object = zlib.decompress(object)
        if self.msgspec:
            object = msgspec.json.decode(object)

        object: OBJECT_TYPES = TypeAdapter(
            OBJECT_TYPES, config=RequestModel.model_config
        ).validate_python(object)

        return object.deserialize(handler)


class StreamValueModel(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    value: str | bytes | ValueTypes

    msgspec: bool = False
    zlib: bool = False

    @field_serializer("value")
    def serialize_object(self, value: Union[ValueTypes]) -> str:

        if isinstance(value, (str, bytes)):
            return value

        self.msgspec = True
        self.zlib = True

        data = msgspec.json.encode(value)
        data = zlib.compress(data)

        return data

    def deserialize(self, model: NNsight):

        handler = DeserializeHandler(model=model)

        value = self.value

        if self.zlib:
            value = zlib.decompress(value)
        if self.msgspec:
            value = msgspec.json.decode(value)

        return try_deserialize(value, handler)
