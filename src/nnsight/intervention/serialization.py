import pickle
from types import FrameType
from typing import TYPE_CHECKING, Any, Optional

import dill

if TYPE_CHECKING:
    from .envoy import Envoy
else:
    Envoy = Any



class CustomDillPickler(dill.Pickler):
    def persistent_id(self, obj):

        from .envoy import Envoy

        if isinstance(obj, Envoy):
            return f"ENVOY:{obj.path}"
        
        if isinstance(obj, FrameType):
            return "FRAME"

        return None


class CustomDillUnpickler(dill.Unpickler):
    def __init__(self, file, root: Envoy, frame: FrameType):
        super().__init__(file)
        self.root = root
        self.frame = frame

    def persistent_load(self, pid):

        if pid.startswith("ENVOY:"):
            path = pid.removeprefix("ENVOY:model")
            return self.root.get(path)

        if pid == "FRAME":
            return self.frame

        raise pickle.UnpicklingError(f"Unknown persistent id: {pid}")


def save(obj: Any, path: str):

    with open(path, "wb") as file:
        CustomDillPickler(file).dump(obj)


def load(path: str, model: Envoy, frame: Optional[FrameType] = None):

    with open(path, "rb") as file:
        return CustomDillUnpickler(file, model, frame).load()
