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
        if "importlib" not in frame.f_code.co_filename:
            linecache.getlines(frame.f_code.co_filename, frame.f_globals)
except Exception:
    pass

from functools import wraps

import os
from .schema.config import ConfigModel

PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG = ConfigModel.load(PATH)

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nnsight")
except PackageNotFoundError:
    try:
        from ._version import version as __version__
    except ImportError:
        __version__ = "unknown version"

from .ndif import *

from IPython import get_ipython

try:
    __IPYTHON__ = get_ipython() is not None
except NameError:
    __IPYTHON__ = False

__INTERACTIVE__ = (sys.flags.interactive or not sys.argv[0]) and not __IPYTHON__

NNS_VLLM_VERSION = "0.15.1"


from .intervention.envoy import Envoy
from .modeling.base import NNsight
from .modeling.language import LanguageModel
from .modeling.vlm import VisionLanguageModel
try:
    from .modeling.diffusion import DiffusionModel
except ImportError:
    pass
from .intervention.tracing.base import Tracer
from .intervention.tracing.globals import save
from .intervention.tracing.util import ExceptionWrapper

# Custom exception hook to show clean tracebacks for NNsight exceptions
_original_excepthook = sys.excepthook


def _nnsight_excepthook(exc_type, exc_value, exc_tb):
    """Custom exception hook that prints clean tracebacks for NNsight exceptions."""
    if isinstance(exc_value, ExceptionWrapper):
        # Print the reconstructed traceback with rich syntax highlighting
        # Pass outer_tb to include user code frames from the call stack
        exc_value.print_exception(file=sys.stderr, outer_tb=exc_tb)
    else:
        # Use the original exception hook for other exceptions or in DEBUG mode
        _original_excepthook(exc_type, exc_value, exc_tb)


sys.excepthook = _nnsight_excepthook

# Also handle IPython if available
try:
    _ipython = get_ipython()
    if _ipython is not None:

        def _nnsight_ipython_exception_handler(self, etype, evalue, tb, tb_offset=None):
            """Custom IPython exception handler for NNsight exceptions."""

            if isinstance(evalue, ExceptionWrapper):
                evalue.print_exception(file=sys.stderr, outer_tb=tb)
            else:
                self.showtraceback((etype, evalue, tb), tb_offset=tb_offset)

        _ipython.set_custom_exc((ExceptionWrapper,), _nnsight_ipython_exception_handler)
except (NameError, AttributeError):
    pass


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
    def inner(self, device_type: str, *args, **kwargs):

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
            tracer.capture()

        except WithBlockNotFoundError:

            return func(tensor, *args, **kwargs)

        return tracer

    return inner


DEFAULT_PATCHER.add(Patch(Tensor, wrap_backward(Tensor.backward), "backward"))

DEFAULT_PATCHER.__enter__()

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

    __INTERACTIVE_CONSOLE__ = NNsightConsole(
        filename="<nnsight-console>", locals=ilocals
    )
    __INTERACTIVE_CONSOLE__.interact(banner="")

else:
    __INTERACTIVE_CONSOLE__ = None
