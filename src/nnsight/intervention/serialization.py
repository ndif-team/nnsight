from collections import defaultdict
import io
import pickle
import warnings
from builtins import open
from types import FrameType
from typing import Any, Optional, Tuple, Union, TYPE_CHECKING
from ..util import Patcher, Patch
import cloudpickle

from .envoy import Envoy
from .serialization_source import (
    serialize_source_based,
    deserialize_source_based,
    SourceSerializationError,
    can_serialize_source_based,
)

if TYPE_CHECKING:
    from .tracing.base import Tracer


class CustomCloudPickler(cloudpickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, FrameType):
            return f"FRAME{id(obj)}"

        return None

original_setstate = Envoy.__setstate__
   

class CustomCloudUnpickler(pickle.Unpickler):
    def __init__(self, file, root: Envoy, frame: FrameType):
        super().__init__(file)
        self.root = root
        self.frame = frame
        
        self.proxy_frames = defaultdict(dict)
        
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


        if pid.startswith("FRAME"):
            return self.proxy_frames[pid]

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


def save_for_remote(tracer: "Tracer") -> Tuple[bytes, str]:
    """
    Serialize tracer for remote execution with automatic format selection.

    Attempts source-based serialization first. Falls back to cloudpickle
    with a deprecation warning if any variables can't be serialized.

    Args:
        tracer: The tracer object to serialize

    Returns:
        Tuple of (serialized_bytes, format_string)
        format_string is either "source" or "cloudpickle"
    """
    # Try source-based serialization first
    can_serialize, error = can_serialize_source_based(tracer)

    if can_serialize:
        return serialize_source_based(tracer), "source"

    # Fall back to cloudpickle with deprecation warning
    warnings.warn(
        f"Falling back to cloudpickle serialization:\n"
        f"  {error}\n\n"
        f"This requires matching Python versions between client and server.\n"
        f"To use version-agnostic serialization:\n"
        f"  - Use JSON-serializable variables (int, float, str, list, dict)\n"
        f"  - Mark functions and classes with @nnsight.remote\n\n"
        f"Cloudpickle fallback will be removed in nnsight 2.0.",
        DeprecationWarning,
        stacklevel=4
    )
    return save(tracer), "cloudpickle"


def load_for_remote(data: bytes, format: str, model: Envoy, frame: Optional[FrameType] = None):
    """
    Deserialize remote payload based on format.

    Args:
        data: Serialized bytes
        format: Either "source" or "cloudpickle"
        model: The model Envoy for context
        frame: Optional frame for cloudpickle deserialization

    Returns:
        Deserialized object or namespace dict (for source format)
    """
    if format == "source":
        return deserialize_source_based(data, model)
    else:
        return load(data, model, frame)
