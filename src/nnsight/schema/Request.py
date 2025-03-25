from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Union

from pydantic import BaseModel, ConfigDict, TypeAdapter, field_serializer

from .. import NNsight
from .format.types import *

if TYPE_CHECKING:
    from ..contexts.backends.RemoteBackend import RemoteMixin

OBJECT_TYPES = Union[SessionType, TracerType, SessionModel, TracerModel]


class RequestModel(BaseModel):

    model_config = ConfigDict(
        arbitrary_types_allowed=True, protected_namespaces=()
    )

    object: Union[str, OBJECT_TYPES]
    model_key: str

    id: str = None
    received: datetime = None

    session_id: Optional[str] = None

    @field_serializer("object")
    def serialize_object(
        self, object: Union[SessionType, TracerType, SessionModel, TracerModel]
    ) -> str:

        if isinstance(object, str):
            return object

        return object.model_dump_json()

    def deserialize(self, model: NNsight) -> "RemoteMixin":

        handler = DeserializeHandler(model=model)

        object = TypeAdapter(
            OBJECT_TYPES, config=RequestModel.model_config
        ).validate_python(json.loads(self.object))

        return object.deserialize(handler)
