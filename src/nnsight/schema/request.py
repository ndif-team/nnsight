from __future__ import annotations

import copy
import io
import json
import zlib
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Union

import msgspec
import torch
from pydantic import BaseModel, ConfigDict

from .format.types import (MEMO, DeserializeHandler, Graph, GraphModel,
                           GraphType, ValueTypes, try_deserialize)

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
    
    @staticmethod
    def serialize(graph: Graph, format:str, _zlib:bool) -> bytes:
        
        if format == "json":

            data = RequestModel(graph=graph)

            json = data.model_dump(mode="json")

            data = msgspec.json.encode(json)

        elif format == "pt":

            data = io.BytesIO()

            torch.save(graph, data)

            data.seek(0)

            data = data.read()
            
        if _zlib:

            data = zlib.compress(data)
            
        return data

    @staticmethod
    def deserialize(model: "NNsight", graph:bytes, format:str, _zlib:bool) -> Graph:
        
        if _zlib:

            graph = zlib.decompress(graph)

        if format == "json":

            nnsight_request = msgspec.json.decode(graph)

            request = RequestModel(**nnsight_request)
            
            handler = DeserializeHandler(request.memo, model)

            graph = request.graph.deserialize(handler)

        elif format == "pt":

            data = io.BytesIO(graph)

            data.seek(0)

            graph = torch.load(data, map_location="cpu", weights_only=False)

        return graph

class StreamValueModel(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    values: Dict[int, ValueTypes]
    memo: Dict[int, ValueTypes]
    
    def __init__(self, *args, memo: Dict = None, **kwargs):

        super().__init__(*args, memo=memo or dict(), **kwargs)

        if memo is None:

            self.memo = {**MEMO}

            MEMO.clear()

    @staticmethod
    def serialize(values: Dict[int, Any], format:str, _zlib:bool) -> bytes:
        
        if format == "json":

            data = StreamValueModel(values=values)

            json = data.model_dump(mode="json")

            data = msgspec.json.encode(json)

        elif format == "pt":

            data = io.BytesIO()

            torch.save(values, data)

            data.seek(0)

            data = data.read()
            
        if _zlib:

            data = zlib.compress(data)
            
        return data

    @staticmethod
    def deserialize(values:bytes, format:str, _zlib:bool) -> Dict[int, Any]:
        
        if _zlib:

            values = zlib.decompress(values)

        if format == "json":

            nnsight_request = msgspec.json.decode(values)

            request = StreamValueModel(**nnsight_request)
            
            handler = DeserializeHandler(request.memo, None)

            values = {index: try_deserialize(value, handler) for index, value in request.values.items()}

        elif format == "pt":

            data = io.BytesIO(values)

            data.seek(0)

            values = torch.load(data, map_location="cpu", weights_only=False)

        return values
