from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Union

from pydantic import BaseModel, ConfigDict

from .. import NNsight
from .format.types import *

if TYPE_CHECKING:
    from ..contexts.backends.RemoteBackend import RemoteMixin


class RequestModel(BaseModel):
    
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    object: Union[SessionType, TracerType, SessionModel, TracerModel]
    model_key: str

    id: str = None
    received: datetime = None

    session_id: Optional[str] = None

    def deserialize(self, model: NNsight) -> "RemoteMixin":
        
        handler = DeserializeHandler(model=model)

        return self.object.deserialize(handler)
