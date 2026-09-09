# Import torch and transformers up front, before this module mounts `.save` onto
# `object` (bottom of the file). anyio's strict
# TypedAttributeSet.__init_subclass__ rejects any class with an unannotated
# public attribute defined *after* the mount adds `save` to every object
# ("Attribute 'save' is missing its type annotation"), so every anyio class has to
# be defined first. Warming them here means they always are, no matter what imports
# nnsight (the API worker, the Ray controller / model actor, ...).
import torch  # noqa: F401
import transformers  # noqa: F401

# transformers reaches anyio via huggingface_hub -> httpcore, but only on some
# installs, and other stacks reach it by their own route (vLLM via openai). Warm it
# directly rather than through a chain that may not pull it; it isn't a dependency
# of nnsight, so it may legitimately be absent.
try:
    import anyio  # noqa: F401
except ImportError:
    pass

import functools
import warnings
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# The installed package metadata (pyproject `version`) is the single source of
# truth; falls back to the static string when running from an unbuilt tree.
try:
    __version__ = _pkg_version("nnsight")
except PackageNotFoundError:
    __version__ = "0.8.0"


class NNsightDeprecationWarning(FutureWarning):
    """The category every nnsight deprecation is warned under.

    A ``FutureWarning``, not a ``DeprecationWarning``: Python's default filters
    are ``default::DeprecationWarning:__main__`` followed by
    ``ignore::DeprecationWarning``, so a deprecated call reached from anywhere but
    the top level of the running script — a package, a helper module, a library
    wrapping nnsight — warns to nobody. ``FutureWarning`` is the category the
    language reserves for deprecations addressed to a library's *users* rather
    than to its developers, and no default filter hides it, so the warning shows
    in a script, in an imported module and in a notebook alike.

    Its own class so a caller can silence nnsight's deprecations and nothing
    else::

        warnings.filterwarnings("ignore", category=nnsight.NNsightDeprecationWarning)

    nnsight registers no filters of its own: a library that mutates the global
    filter list overrides the ``-W`` flags and ``PYTHONWARNINGS`` its user chose.
    """


def deprecated(message: str):
    """Decorator: emit an `NNsightDeprecationWarning` when the wrapped callable is used.

    Defined before the submodule imports below so they can decorate methods with
    ``@deprecated("...")`` (e.g. the ``model.iter`` / ``model.all()`` aliases).
    ``message`` reads ``"<what is deprecated> is deprecated; use <what to write
    instead> instead."`` — every nnsight deprecation names its replacement.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(message, NNsightDeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Applies global torch patches on import (e.g. Tensor.backward, so
# `with tensor.backward():` enters a backward trace). Importing any nnsight
# submodule runs this, so the patches are always in place.
from .intervention import backward  # noqa: F401

# nnsight.save(value): mark a value to return from the outermost trace.
from .tracing.tracer import save  # noqa: F401

# nnsight.Object: tensor-like static type for values read inside a trace (hints).
from .tracing.hint import Object  # noqa: F401

# nnsight.NNsight(module): wrap an arbitrary torch.nn.Module for tracing.
from .modeling.base import NNsight  # noqa: F401

from .intervention.envoy import Envoy  # noqa: F401

# Model classes are exposed lazily so importing nnsight doesn't pull in
# transformers/diffusers model machinery until one is actually constructed
# (and an optional dep like diffusers only errors when its model is used).
_LAZY_IMPORTS = {
    "LanguageModel": ".modeling.language",
    "VisionLanguageModel": ".modeling.vlm",
    "TransformersModel": ".modeling.transformers",
    "DiffusionModel": ".modeling.diffusion",
}


def __getattr__(name):
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'nnsight' has no attribute {name!r}")
    import importlib

    try:
        module = importlib.import_module(target, __name__)
    except ImportError as error:
        raise AttributeError(
            f"{name} is unavailable ({error}). Install the missing optional "
            "dependency to use it."
        ) from error
    value = getattr(module, name)
    globals()[name] = value
    return value


# A module-level __getattr__ is not consulted by ``from nnsight import *``; name
# the lazy entries in __all__ so star-imports still see them.
__all__ = [
    name for name in globals() if not name.startswith("_")
] + list(_LAZY_IMPORTS)

# Top-level NDIF helpers: register a local module for remote pickling, query
# service/model status, compare local vs remote environments.
from .ndif import (  # noqa: F401
    compare,
    is_model_running,
    login,
    ndif_status,
    register,
    status,
    whoami,
)

# Mount `.save()` on every object (so `value.save()` works, not just
# `save(value)`) via the optional C extension. Gated by config; if the extension
# didn't build, this silently falls back to `nnsight.save(value)`.
from .schema.config import CONFIG  # noqa: E402

if CONFIG.APP.PYMOUNT:
    try:
        from ._c import mount

        mount(save, "save")
    except Exception:  # noqa: BLE001 — extension optional; save(value) still works
        pass

# Neutralize glibc backtrace() so a torch C++ error raised inside an interleaving
# greenlet surfaces as a Python exception instead of segfaulting the process.
if CONFIG.APP.DISABLE_CPP_BACKTRACE:
    from ._c import install_backtrace_guard

    install_backtrace_guard()
