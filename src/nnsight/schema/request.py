from __future__ import annotations

import copy
import json
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Union

from pydantic import BaseModel, ConfigDict

from .format.types import (
    MEMO,
    DeserializeHandler,
    Graph,
    GraphModel,
    GraphType,
    ValueTypes,
)

if TYPE_CHECKING:
    from .. import NNsight


class RequestModel(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    graph: Union[GraphModel, GraphType]
    memo: Dict[int, ValueTypes]

    def __init__(self, *args, memo: Dict = None, **kwargs):

        super().__init__(*args, memo=memo or dict(), **kwargs)

        if memo is None:

            self.memo = {**MEMO}

            MEMO.clear()

    def deserialize(self):

        handler = DeserializeHandler(self.memo)

        return self.graph.deserialize(handler)

class StreamValueModel(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    value: ValueTypes

    def deserialize(self, model: "NNsight"):

        handler = DeserializeHandler(model=model)

        return try_deserialize(self.value, handler)
