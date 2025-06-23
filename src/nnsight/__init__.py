from functools import wraps
import os, yaml
import warnings
from typing import Optional
from .schema.config import ConfigModel

PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PATH, "config.yaml"), "r") as file:
    CONFIG = ConfigModel(**yaml.safe_load(file))
    
from importlib.metadata import PackageNotFoundError, version
  
try:
    __version__ = version("nnsight")
except PackageNotFoundError:
    __version__ = "unknown version"

from IPython import get_ipython

try:
    __IPYTHON__ = get_ipython() is not None
except NameError:
    __IPYTHON__ = False
    
base_deprecation_message = "is deprecated as of v0.5.0 and will be removed in a future version."
    
def deprecated(message:Optional[str]=None, error:bool=False):
    
    def decorator(func):
        
        @wraps(func)
        def inner(*args, **kwargs):
            
            deprecation_message = (
                f"{func.__module__}.{func.__name__} {base_deprecation_message}"
                + (f"\n{message}" if message is not None else "")
            )
            
            if error:
                raise DeprecationWarning(deprecation_message)
            else:
                warnings.warn(deprecation_message)
                
            return func(*args, **kwargs)
        
        return inner
    
    return decorator




from .intervention.envoy import Envoy
from .modeling.base import NNsight
from .modeling.language import LanguageModel
from .intervention.tracing.base import Tracer


    
def session(*args, **kwargs):
    return Tracer(*args, **kwargs)

from .util import Patcher, Patch

DEFAULT_PATCHER = Patcher()

# Tensor creation operations
from torch._subclasses.fake_tensor import FakeTensor


def fake_bool(self):
    return True


DEFAULT_PATCHER.add(Patch(FakeTensor, fake_bool, "__bool__"))

from torch.amp.autocast_mode import autocast

def wrap_autocast(func):
    
    @wraps(func)
    def inner(self, device_type:str, *args, **kwargs):
        
        if device_type == "meta":
            device_type = "cpu"
            
        return func(self, device_type, *args, **kwargs)
        
    return inner


DEFAULT_PATCHER.add(Patch(autocast, wrap_autocast(autocast.__init__), "__init__"))


DEFAULT_PATCHER.__enter__()


## TODO: Legacy

@deprecated()
def apply(fn, *args, **kwargs):
    
    return fn(*args, **kwargs)

@deprecated()
def log(message:str):
    
    print(message)
  
@deprecated(error=True)
def local(*args, **kwargs):
    pass

@deprecated(error=True)
def cond(*args, **kwargs):
    pass

@deprecated(error=True)
def iter(*args, **kwargs):
    pass

from .intervention.interleaver import EarlyStopException

@deprecated()
def stop():
    raise EarlyStopException()

@deprecated()
def trace(fn):
    
    return fn


bool = deprecated(message="Use the standard `bool()` instead.")(bool)
bytes = deprecated(message="Use the standard `bytes()` instead.")(bytes)
complex = deprecated(message="Use the standard `complex()` instead.")(complex)
dict = deprecated(message="Use the standard `dict()` instead.")(dict)
float = deprecated(message="Use the standard `float()` instead.")(float)
int = deprecated(message="Use the standard `int()` instead.")(int)
list = deprecated(message="Use the standard `list()` instead.")(list)
str = deprecated(message="Use the standard `str()` instead.")(str)
tuple = deprecated(message="Use the standard `tuple()` instead.")(tuple)
bytearray = deprecated(message="Use the standard `bytearray()` instead.")(bytearray)







