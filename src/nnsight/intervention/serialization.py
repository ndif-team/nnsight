import io
import pickle
from builtins import open
from types import FrameType
from typing import Any, Optional, Union
from ..util import Patcher, Patch
import cloudpickle

from .envoy import Envoy


class CustomCloudPickler(cloudpickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, FrameType):
            return "FRAME"

        return None

original_setstate = Envoy.__setstate__
    
   

class CustomCloudUnpickler(pickle.Unpickler):
    def __init__(self, file, root: Envoy, frame: FrameType):
        super().__init__(file)
        self.root = root
        self.frame = frame
        
    def load(self):
        
        def inject(_self, state):
            
            original_setstate(_self, state)
            
            envoy = self.root.get(_self.path.removeprefix("model"))
            
            _self._module = envoy._module
            _self._interleaver = envoy._interleaver
                                                
            for key, value in envoy.__dict__.items():
                if key not in _self.__dict__:
                    _self.__dict__[key] = value
        
        with Patcher([Patch(Envoy, inject, '__setstate__')]):
            return super().load()

    def persistent_load(self, pid):


        if pid == "FRAME":
            return self.frame

        raise pickle.UnpicklingError(f"Unknown persistent id: {pid}")


def save(obj: Any, path: Optional[str] = None):


    if path is None:
        file = io.BytesIO()
        CustomCloudPickler(file, protocol=4).dump(obj)
        file.seek(0)
        return file.read()

    with open(path, "wb") as file:
        CustomCloudPickler(file).dump(obj)


def load(data: Union[str, bytes], model: Envoy, frame: Optional[FrameType] = None):

    if isinstance(data, bytes):
        return CustomCloudUnpickler(io.BytesIO(data), model, frame).load()

    with open(data, "rb") as file:
        return CustomCloudUnpickler(file, model, frame).load()
