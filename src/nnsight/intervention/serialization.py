import io
import pickle
from builtins import open
from types import FrameType
from typing import TYPE_CHECKING, Any, Optional, Union

import cloudpickle

from .envoy import Envoy


class CustomCloudPickler(cloudpickle.Pickler):
    def persistent_id(self, obj):

        if isinstance(obj, Envoy):
            return f"ENVOY:{obj.path}"

        if isinstance(obj, FrameType):
            return "FRAME"

        return None


class CustomCloudUnpickler(pickle.Unpickler):
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
