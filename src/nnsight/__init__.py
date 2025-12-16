# This section ensures that the source code for both the main script and the file where `nnsight` is imported
# is cached in Python's `linecache` module. This is important for robust stack trace and debugging support,
# especially in interactive or dynamic environments where files may change after import.
# 
# 1. We first attempt to cache the main script (the file passed to the Python interpreter) by calling
#    `linecache.getlines` on `sys.argv[0]`. This ensures that if the script is modified or deleted after
#    execution starts, traceback and inspection tools can still access its source.
# 2. Next, we walk up the call stack to find the first frame outside of importlib (i.e., the user code that
#    imported `nnsight`), and cache its source file as well. This helps ensure that the file where `nnsight`
#    is imported is also available in `linecache`, even if it changes later.

import linecache
import sys
import os

try:
    # Cache the main script file
    linecache.getlines(os.path.abspath(sys.argv[0]))
except Exception:
    pass

import inspect

try:
    # Walk up the stack to cache the file where nnsight is imported
    frame = inspect.currentframe()
    while frame.f_back:
        frame = frame.f_back
        if 'importlib' not in frame.f_code.co_filename:
            linecache.getlines(frame.f_code.co_filename, frame.f_globals)
except Exception:
    pass

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

from .ndif import *

from IPython import get_ipython

try:
    __IPYTHON__ = get_ipython() is not None
except NameError:
    __IPYTHON__ = False
    
__INTERACTIVE__ = (sys.flags.interactive or not sys.argv[0]) and not __IPYTHON__

base_deprecation_message = "is deprecated as of v0.5.0 and will be removed in a future version."

NNS_VLLM_VERSION = "0.9.2"
    
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
from .intervention.tracing.globals import save


    
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


from torch import Tensor
from .intervention.tracing.backwards import BackwardsTracer
from .intervention.tracing.base import WithBlockNotFoundError

def wrap_backward(func):

    @wraps(func)
    def inner(tensor: Tensor, *args, **kwargs):
        
        try:

            tracer = BackwardsTracer(tensor, func, *args, **kwargs)

        except WithBlockNotFoundError:

            return func(tensor, *args, **kwargs)

        return tracer

    return inner

DEFAULT_PATCHER.add(Patch(Tensor, wrap_backward(Tensor.backward), "backward"))

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

if __INTERACTIVE__:
    from code import InteractiveConsole
    import readline
    
    # We need to use our own console so when we trace, we can get the source code from the buffer of our custom console.
    class NNsightConsole(InteractiveConsole):
        pass
    
    # We want the new console to have the locals of the interactive frame the user is already in.
    frame = inspect.currentframe()

    while frame.f_back:
        frame = frame.f_back

    # This was kicked off by importing nnsight, but upon entering the new console, we wont actually have the import in the new locals.
    # So we need to get the last command from the history and execute it in the new locals.
    length = readline.get_current_history_length()
    l1 = readline.get_history_item(length)
    ilocals = {**frame.f_locals}
    exec(l1, frame.f_globals, ilocals)
    
    __INTERACTIVE_CONSOLE__ = NNsightConsole(filename="<nnsight-console>", locals=ilocals)
    __INTERACTIVE_CONSOLE__.interact(banner="")

else:
    __INTERACTIVE_CONSOLE__ = None
