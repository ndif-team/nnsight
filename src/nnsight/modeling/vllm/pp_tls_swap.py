"""Per-greenlet torch thread-local state isolation. Env-gated.

``NNSIGHT_PP_TLS_SWAP=1`` installs a greenlet trace hook on the forward
thread: on every switch or throw, capture the departing greenlet's torch
thread-local state and install the arriving greenlet's saved one. This
restores the per-thread isolation that 0.7's real threads provided, where
each intervention block had its own copy of torch's per-thread state.

The bundle is the C-level ``at::ThreadLocalState`` plus the c10 warning
handler, captured through a small extension JIT-built against the installed
torch on first use (see ``pp_tls_state.cpp``). A Python-level bundle (grad
mode, dispatcher key sets) was prototyped first and failed the canary —
``torch.ones_like(unforced_lazy)`` on a PP engine still killed the worker —
because the crashing word, the warning handler pointing into the parked
greenlet's C stack, has no Python surface.

While the swap is active, the ``__torch_function__`` materialization guard
in ``lazy_remote_tensor`` stands down: parks inside torch's dispatcher are
the case this swap makes safe.
"""

from __future__ import annotations

import os
import threading
import weakref

import greenlet

# Saved bundle per greenlet; weak keys so dead workers drop their state.
_saved: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_installed = threading.local()
_module = None


def requested() -> bool:
    return os.environ.get("NNSIGHT_PP_TLS_SWAP") == "1"


def active() -> bool:
    """Whether the swap is loaded and protecting this process.

    The guard consults this: it must stand down only once the extension
    actually loaded, never on the env var alone.
    """
    return _module is not None


def _load():
    global _module
    if _module is not None:
        return
    from torch.utils.cpp_extension import load

    _module = load(
        name="nnsight_pp_tls_state",
        sources=[os.path.join(os.path.dirname(__file__), "pp_tls_state.cpp")],
        verbose=False,
    )


def _trace(event: str, args) -> None:
    if event not in ("switch", "throw"):
        return
    origin, target = args
    _saved[origin] = _module.Bundle()
    bundle = _saved.get(target)
    if bundle is not None:
        bundle.restore()


def install() -> None:
    """Build/load the extension and install the swap on the calling thread.

    Idempotent per thread. Must run on the greenlets' thread (the forward
    thread); ``load_model`` may run on a different one.
    """
    _load()
    if getattr(_installed, "done", False):
        return
    greenlet.settrace(_trace)
    _installed.done = True
