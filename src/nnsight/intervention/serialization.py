import io
import pickle
from builtins import open
from types import FrameType
from typing import TYPE_CHECKING, Any, Optional, Union

import cloudpickle

from .envoy import Envoy


class CustomCloudPickler(cloudpickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, FrameType):
            return "FRAME"

        return None


class CustomCloudUnpickler(pickle.Unpickler):
    def __init__(self, file, root: Envoy, frame: FrameType):
        super().__init__(file)
        self.root = root
        self.frame = frame
        
    def find_class(self, module, name):
        
        cls = super().find_class(module, name)
        
        if isinstance(cls, type) and issubclass(cls, Envoy):
            
            class EnvoyProxy(cls):
                def __setstate__(_self, state):
                    cls.__setstate__(_self, state) 
                    
                    envoy = self.root.get(_self.path.removeprefix("model"))

                    _self._module = envoy._module
                    _self._interleaver = envoy._interleaver
                    
                    for key, value in envoy.__dict__.items():
                        if key not in _self.__dict__:
                            _self.__dict__[key] = value
            
            return EnvoyProxy
        return cls

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
