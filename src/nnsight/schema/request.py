from __future__ import annotations

import io
import zlib
from typing import TYPE_CHECKING, Any, Dict, List, Union
from typing import Callable
import torch
import dill
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from .. import NNsight
    from ..intervention.tracing.tracer import Tracer
else:
    Tracer = Any

class RequestModel(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    interventions: Callable
    tracer: Tracer

    
    def serialize(self, _zlib:bool) -> bytes:
                
        with io.BytesIO() as data:
        
            dill.dump(self, data, recurse=True)

            data.seek(0)

            data = data.read()
                
        if _zlib:

            data = zlib.compress(data)
                
        return data

    @staticmethod
    def deserialize(model: "NNsight", request:bytes,  _zlib:bool) -> RequestModel:
        
        if _zlib:

            request = zlib.decompress(request)

        with io.BytesIO(request) as data:

            data.seek(0)

            request:RequestModel = dill.load(data)
        
        request.tracer.__setmodel__(model)

        return request

RequestModel.update_forward_refs()