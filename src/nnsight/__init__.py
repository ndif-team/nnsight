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

from .intervention.tracing.globals import save
from .ndif import *

# Detect IPython without importing it — if IPython isn't already in sys.modules,
# we are not running under IPython and there's no need to pay its import cost.
_ipy_mod = sys.modules.get("IPython")
try:
    __IPYTHON__ = _ipy_mod is not None and _ipy_mod.get_ipython() is not None
except Exception:
    __IPYTHON__ = False

__INTERACTIVE__ = (sys.flags.interactive or not sys.argv[0]) and not __IPYTHON__


from .intervention.envoy import Envoy
from .modeling.base import NNsight

# Public names whose modules are expensive to import (transformers ~3s,
# diffusers ~1.5s) but only relevant when those classes are actually used.
# Map: public name -> submodule (relative to this package) defining it.
_LAZY_IMPORTS = {
    "LanguageModel":       ".modeling.language",
    "VisionLanguageModel": ".modeling.vlm",
    "DiffusionModel":      ".modeling.diffusion",
}

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # Static-analyzer view (Pyright/Pylance/mypy/PyCharm). Dead at runtime.
    from .modeling.language import LanguageModel
    from .modeling.vlm import VisionLanguageModel
    from .modeling.diffusion import DiffusionModel


def __getattr__(name):
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'nnsight' has no attribute {name!r}")
    import importlib
    try:
        module = importlib.import_module(target, __name__)
    except ImportError as e:
        raise AttributeError(
            f"{name} is unavailable ({e}). Install the missing optional "
            "dependency to use it."
        ) from e
    value = getattr(module, name)
    globals()[name] = value
    return value


from .intervention.tracing.base import Tracer
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
    _ipython = _ipy_mod.get_ipython() if _ipy_mod is not None else None
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

# `from nnsight import *` consults __all__ (or globals() if absent), but PEP 562
# module-level __getattr__ is not consulted by star-imports. Without naming the
# lazy entries here, `LanguageModel` / `VisionLanguageModel` / `DiffusionModel`
# would silently drop from `*`. Build __all__ from the currently-public globals
# plus the lazy names to preserve prior star-import behavior end-to-end.
__all__ = sorted(
    {name for name in globals() if not name.startswith("_")} | set(_LAZY_IMPORTS)
)

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
