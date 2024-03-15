from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Union

from pydantic import BaseModel, ConfigDict

from .. import NNsight
from .format.objects import *
from .format.types import *

if TYPE_CHECKING:
    from ..contexts.backends.LocalBackend import LocalMixin
    from ..contexts.backends.RemoteBackend import RemoteMixin


class RequestModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    object: ObjectTypes
    repo_id: str

    id: str = None
    session_id: str = None
    received: datetime = None

    def compile(self, model: NNsight) -> Union["LocalMixin", "RemoteMixin"]:

        obj = self.object.compile(model)

        return obj
