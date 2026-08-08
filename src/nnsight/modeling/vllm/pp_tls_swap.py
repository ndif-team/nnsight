"""Per-greenlet torch thread-local state isolation. PROTOTYPE, env-gated.

``NNSIGHT_PP_TLS_SWAP=1`` installs a greenlet trace hook on the forward
thread: on every switch or throw, capture the departing greenlet's torch
thread-local bundle and install the arriving greenlet's saved one. This
emulates the per-thread isolation that 0.7's real threads provided, where
each intervention block had its own copy of torch's per-thread state.

While active, the ``__torch_function__`` materialization guard in
``lazy_remote_tensor`` stands down: parks inside torch's dispatcher are the
case this swap exists to make safe, and the canary for coverage is exactly
``torch.ones_like(unforced_lazy)`` on a PP engine.

Bundle covered: grad mode and the dispatcher's local include/exclude key
sets. NOT covered, because they have no Python surface: the C++ warning
handler (a pointer into the greenlet's C stack installed by every torch
binding on entry) and whatever else ``at::ThreadLocalState`` carries. If the
canary still crashes with the swap active, an uncovered word is confirmed and
the full fix needs a C binding to ``at::ThreadLocalState`` capture/replace.
"""

from __future__ import annotations

import os
import threading
import weakref

import greenlet
import torch

_KEYS = list(torch._C.DispatchKey.__members__.values())
# Saved bundle per greenlet; weak keys so dead workers drop their state.
_saved: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_installed = threading.local()


def active() -> bool:
    return os.environ.get("NNSIGHT_PP_TLS_SWAP") == "1"


def _capture():
    return (
        torch.is_grad_enabled(),
        torch._C._dispatch_tls_local_include_set(),
        torch._C._dispatch_tls_local_exclude_set(),
    )


def _restore(bundle) -> None:
    grad, include, exclude = bundle
    if torch.is_grad_enabled() != grad:
        torch._C._set_grad_enabled(grad)
    # ponytail: full 145-key sweep on the changed path (~15us); a delta-set
    # walk if this graduates from prototype.
    if torch._C._dispatch_tls_local_include_set() != include:
        for key in _KEYS:
            torch._C._dispatch_tls_set_dispatch_key_included(key, include.has(key))
    if torch._C._dispatch_tls_local_exclude_set() != exclude:
        for key in _KEYS:
            torch._C._dispatch_tls_set_dispatch_key_excluded(key, exclude.has(key))


def _trace(event: str, args) -> None:
    if event not in ("switch", "throw"):
        return
    origin, target = args
    _saved[origin] = _capture()
    bundle = _saved.get(target)
    if bundle is not None:
        _restore(bundle)


def install() -> None:
    """Install the swap on the calling thread (idempotent per thread)."""
    if getattr(_installed, "done", False):
        return
    greenlet.settrace(_trace)
    _installed.done = True
