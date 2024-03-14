from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Union

from pydantic import BaseModel, ConfigDict

from .. import NNsight
from ..tracing.Graph import Graph
from .format import types
from .format.objects import *
from .format.types import *


class RequestModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    object: ObjectTypes
    repo_id: str

    id: str = None
    session_id: str = None
    received: datetime = None

    def compile(self, model: NNsight):

        obj = self.object.compile(model)

        return obj
