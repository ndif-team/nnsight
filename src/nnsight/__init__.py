# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#   _____  ___   _____  ___    ________  __     _______    __    __  ___________        ______     ___  ___     # 
#  (\"   \|"  \ (\"   \|"  \  /"       )|" \   /" _   "|  /" |  | "\("     _   ")      /    " \   (: "||_  |    #
#  |.\\   \    ||.\\   \    |(:   \___/ ||  | (: ( \___) (:  (__)  :))__/  \\__/      // ____  \  |  (__) :|    # 
#  |: \.   \\  ||: \.   \\  | \___  \   |:  |  \/ \       \/      \/    \\_ /        /  /    ) :)  \____  ||    # 
#  |.  \    \. ||.  \    \. |  __/  \\  |.  |  //  \ ___  //  __  \\    |.  |       (: (____/ //_____  _\ '|    # 
#  |    \    \ ||    \    \ | /" \   :) /\  |\(:   _(  _|(:  (  )  :)   \:  |        \        /))_  ")/" \_|\   # 
#   \___|\____\) \___|\____\)(_______/ (__\_|_)\_______)  \__|  |__/     \__|         \"_____/(_____((_______)  # 
#                                                                                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import os
from functools import wraps

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, Dict, Union

try:
    __version__ = version("nnsight")
except PackageNotFoundError:
    __version__ = "unknown version"

from IPython import get_ipython

try:
    __IPYTHON__ = get_ipython() is not None
except NameError:
    __IPYTHON__ = False

import torch
import yaml

from .schema.config import ConfigModel
from .util import Patch, Patcher

PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PATH, "config.yaml"), "r") as file:
    CONFIG = ConfigModel(**yaml.safe_load(file))


from .logger import logger, remote_logger
from .intervention import Envoy, NNsight
from .modeling.language import LanguageModel

logger.disabled = not CONFIG.APP.LOGGING
remote_logger.disabled = not CONFIG.APP.REMOTE_LOGGING

# Below do default patching:
DEFAULT_PATCHER = Patcher()


# Tensor creation operations
from torch._subclasses.fake_tensor import FakeTensor


def fake_bool(self):
    return True


DEFAULT_PATCHER.add(Patch(FakeTensor, fake_bool, "__bool__"))


def fake_tensor_new_wrapper(fn):

    @wraps(fn)
    def inner(cls, fake_mode, elem, device, constant=None):

        if isinstance(elem, FakeTensor):

            return elem

        else:

            return fn(cls, fake_mode, elem, device, constant=constant)

    return inner


DEFAULT_PATCHER.add(
    Patch(FakeTensor, fake_tensor_new_wrapper(FakeTensor.__new__), "__new__")
)

def autoamp_init_wrapper(fn):
    
    @wraps(fn)
    def inner(self, device_type, dtype=None, **kwargs):
        
        if device_type == "meta":
            dtype = torch.get_autocast_cpu_dtype()
            
        return fn(self, device_type, dtype, **kwargs)
    
    return inner

DEFAULT_PATCHER.add(
    Patch(torch.autocast, autoamp_init_wrapper(torch.autocast.__init__), "__init__")
)

DEFAULT_PATCHER.__enter__()

from .intervention.contexts import GlobalInterventionTracingContext

apply = GlobalInterventionTracingContext.GLOBAL_TRACING_CONTEXT.apply
log = GlobalInterventionTracingContext.GLOBAL_TRACING_CONTEXT.log
local = GlobalInterventionTracingContext.GLOBAL_TRACING_CONTEXT.local
cond = GlobalInterventionTracingContext.GLOBAL_TRACING_CONTEXT.cond
iter = GlobalInterventionTracingContext.GLOBAL_TRACING_CONTEXT.iter
stop = GlobalInterventionTracingContext.GLOBAL_TRACING_CONTEXT.stop

def trace(fn):
    """Helper decorator to add a function to the intervention graph via `.apply(...)`.
    This is opposed to entering the function during tracing and tracing all inner operations.

    Args:
        fn (Callable): Function to apply.

    Returns:
        Callable: Traceable function.
    """

    @wraps(fn)
    def inner(*args, **kwargs):
        
        return apply(fn, *args, **kwargs)

    return inner


bool = trace(bool)
bytes = trace(bytes)
int = trace(int)
float = trace(float)
str = trace(str)
complex = trace(complex)
bytearray = trace(bytearray)
tuple = trace(tuple)
list = trace(list)
set = trace(set)
dict = trace(dict)