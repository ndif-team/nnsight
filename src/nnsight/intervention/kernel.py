"""Serve intermediate values from fused kernels during a trace: registry, shim and running-module stack.

Fused kernels read their inputs from global memory, compute intermediate values
in registers or shared memory, and write only the final output back. For example,
FlashAttention computes attention probabilities on the fly without writing the
full probability matrix to memory, and chunked state-space kernels only form
recurrent states at chunk boundaries. Standard interpretability methods often
switch the model to an eager implementation to read these intermediates, which
changes the floating-point arithmetic. In bf16 on Qwen3-8B (56-token prompt; see
docs/usage/kernel.md), eager attention probabilities differ from the fused
kernel's probabilities by up to 19% at the top key, and the output logits shift
by a KL divergence of 1e-3.

Fused kernels already save compact summary statistics for their backward pass:
attention kernels save the per-row logsumexp, and chunked scans save the state at
each chunk boundary. During the backward pass, the kernel recomputes the needed
intermediates from the inputs and those saved statistics. This module applies the
same idea during a trace: it runs the exact kernel the model is configured to use,
retains the statistics that the high-level wrapper normally discards, and
reconstructs only the slice requested by the user. Each reconstructible quantity
(such as ``probs`` or ``lse``) is exposed as a kernel *site*.

How it works with the interleaver's location and
[`Interleaver.handle`][nnsight.intervention.interleaver.Interleaver.handle] mechanism:

* A global registry maps each fused callable to a *serve function* and to the
  places from which the model looks up the callable: a module attribute, a
  closure cell, or a dictionary entry (see
  [`register`][nnsight.intervention.kernel.register]). While at least one trace
  is open, [`install`][nnsight.intervention.kernel.install] replaces the
  callable at each of those places with a *shim*, and
  [`uninstall`][nnsight.intervention.kernel.uninstall] restores the original
  callable when the last open trace exits (refcounted across
  [`Interleaver.__enter__`][nnsight.intervention.interleaver.Interleaver.__enter__]
  and
  [`Interleaver.__exit__`][nnsight.intervention.interleaver.Interleaver.__exit__]).
  When the shim is called and a worker is waiting on the kernel location of the
  innermost module currently executing on this thread, the shim invokes the
  serve function. The serve function returns the kernel's normal output along
  with a [`Sites`][nnsight.intervention.sites.Sites] reader holding the
  captured inputs and statistics, and the shim delivers that reader to
  ``"{module}.kernel"``. When no worker is waiting on that module's kernel
  location, the shim calls the underlying kernel directly. The reader computes a
  site only when a worker requests it.
* `Controller.__call__` (in intervention/source.py) pushes the running module
  onto a per-thread stack for the duration of the module's body
  ([`push`][nnsight.intervention.kernel.push] and
  [`pop`][nnsight.intervention.kernel.pop]), which identifies the module that
  owns the kernel call. Plain helper functions between a module's forward pass
  and the kernel, such as transformers' attention interfaces, are not modules
  and do not push onto the stack, so the shim receives the kernel call with its
  preprocessed arguments (after RoPE, KV-cache updates, and layout
  transpositions).
* On the user side, [`Envoy.kernel`][nnsight.intervention.envoy.Envoy.kernel]
  returns a [`Kernel`][nnsight.intervention.sites.Kernel] accessor whose
  attributes correspond to site names. Calling a site parks the worker on
  ``"{module}.kernel"``, receives the `Sites` reader when the kernel runs,
  computes the requested slice on the worker thread, and narrows the result to
  the worker's batch rows. Because slicing happens on the worker thread, invalid
  site names or out-of-bounds indices raise directly on the user's line inside
  the trace block.

.. code-block:: python

    with model.trace(prompt):
        p = model.model.layers[6].self_attn.kernel.probs(heads=4, query=-1).save()

    p.provenance   # Provenance(kind='exact reconstruction', kernel='aten._scaled_dot_product_flash_attention', ...)

Supported kernels are listed in catalogs of
[`Spec`][nnsight.intervention.kernel.Spec] rows, one catalog per backend, and
[`register_catalog`][nnsight.intervention.kernel.register_catalog] registers the
functions each row locates. Each row specifies a table label, a site family, a
locator yielding ``(function, holders)`` pairs, and a serve function:

* intervention/attention.py: PyTorch's SDPA, registered at import.
* Model-library catalogs live alongside the model classes that use them under
  modeling/.

The site interface (`Sites`, `Provenance`, the `Kernel` accessor, and the
shared slicing helpers) is in intervention/sites.py.

Because a kernel executes inside its module's forward pass, the interleaver tracks
the occurrence count of ``"{module}.kernel"`` using ``"{module}.output"`` (see
`Mediator.occurrence`). That counter advances on every call to the module
regardless of whether a site was requested on that call, so a request scoped to
step ``n`` via ``tracer.iter`` always matches step ``n``.

Limitations:
* Sites are read-only, and returned tensors are detached from autograd.
* Because sites are served when the kernel executes, a site must be requested
  before reading any value produced later in the forward pass.
* The shim is reached through the storage locations listed at registration. A
  module that bound the kernel function by value at import time (``from
  torch.nn.functional import scaled_dot_product_attention``) and calls that local
  binding does not reach the shim unless that module is passed to
  [`register`][nnsight.intervention.kernel.register] as a holder. A function
  instrumented by ``.source`` runs with a copy of its module's globals dict
  (`function_like` in intervention/source.py); [`adopt`][nnsight.intervention.kernel.adopt]
  registers every kernel found in that copy as a [`Global`][nnsight.intervention.kernel.Global]
  holder for the lifetime of the copy, and [`register`][nnsight.intervention.kernel.register]
  scans all live copies when a new kernel is registered, so a kernel called as a
  bare global inside an instrumented function also reaches the shim.
* Inside a ``torch.compile`` region, the module stack is not updated and the shim
  calls the kernel directly, so a site requested on a compiled module fails with
  ``OutOfOrderError`` ("requested but the model already ran past it"). If the
  kernel call itself runs eagerly inside a compiled module (for example, inside a
  helper decorated with ``torch._dynamo.disable``), the call is attributed to the
  innermost enclosing module that is running eagerly.
"""

from __future__ import annotations

import functools
import sys
import threading
import warnings
import weakref
from types import FunctionType
from typing import Any, Callable, Iterable, NamedTuple

import torch
from greenlet import getcurrent, greenlet

from .interleaver import Interleaver
from .sites import SUFFIX, Unavailable

# ---------------------------------------------------------------------------
# Registry, shim, and the per-thread stack of running modules
# ---------------------------------------------------------------------------


class Attr:
    """Storage location that holds a kernel function on an attribute, such as a module global or a closure cell (``Attr(cell, "cell_contents")``)."""

    def __init__(self, obj: Any, key: str) -> None:
        self.obj, self.key = obj, key

    def get(self) -> Any:
        return getattr(self.obj, self.key)

    def set(self, value: Any) -> None:
        setattr(self.obj, self.key, value)

    @property
    def alive(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.obj!r}, {self.key!r})"


class Item(Attr):
    """Storage location that holds a kernel function as a dictionary entry. Restoring writes the original value back without deleting the key."""

    def get(self) -> Any:
        return self.obj[self.key]

    def set(self, value: Any) -> None:
        self.obj[self.key] = value


class Global(Attr):
    """Storage location for a global variable inside a function built by `function_like` (intervention/source.py).

    The function is referenced weakly so the holder stays alive only as long as the
    function and its copied globals dict remain referenced (for example, while an
    instrumented forward is cached on a module, while a drilled function lives for
    a single trace, or while `record` runs a single kernel call). Once the function
    is garbage-collected, `get` returns ``None`` and `Entry.add` and `_restore`
    skip the expired holder.
    """

    def __init__(self, fn: FunctionType, key: str) -> None:
        self.ref = weakref.ref(fn)
        self.key = key
        self.name = fn.__qualname__

    @property
    def obj(self) -> dict[str, Any] | None:
        fn = self.ref()
        return None if fn is None else fn.__globals__

    @property
    def alive(self) -> bool:
        return self.ref() is not None

    def get(self) -> Any:
        namespace = self.obj
        return None if namespace is None else namespace.get(self.key)

    def set(self, value: Any) -> None:
        namespace = self.obj
        if namespace is not None:
            namespace[self.key] = value

    def __repr__(self) -> str:
        return f"Global({self.name!r}, {self.key!r})"


class Entry:
    """Registry record pairing a kernel function with its serve function, its shim, and the holders where the model looks it up."""

    def __init__(self, fn: Callable, serve: Callable) -> None:
        self.fn = fn
        self.serve = serve
        self.holders: list[Attr] = []
        #: Holders that `install` overwrote with `shim`, to be restored on `uninstall`.
        self.patched: list[Attr] = []
        self.shim = shim(self)

    def add(self, holder: Attr) -> bool:
        """Append ``holder`` if that storage location is not already present; return ``True`` when added."""
        self.holders = [known for known in self.holders if known.alive]
        key = (type(holder), id(holder.obj), holder.key)
        if any(key == (type(known), id(known.obj), known.key) for known in self.holders):
            return False
        self.holders.append(holder)
        return True


#: Maps each registered fused callable to its `Entry`.
_KERNELS: dict[Callable, Entry] = {}

#: Guards `_KERNELS` and `_open`.
_lock = threading.Lock()

#: Count of currently open interleavers; shims remain installed while this is greater than zero.
_open = 0

#: Maps each registered kernel's shim to the kernel function it wraps (see `unshimmed`).
_shims: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

#: Live functions built by `function_like` over copied module globals (see `adopt`).
_copies: weakref.WeakSet = weakref.WeakSet()


class Frame(NamedTuple):
    """Active module execution frame pushed by `Controller.__call__` while the module's body runs."""

    interleaver: Interleaver
    path: str
    #: Greenlet that entered the module. Kernel calls made from a different
    #: greenlet while the module is on the stack belong to a worker block and
    #: bypass the serve function.
    greenlet: greenlet


class _Frames(threading.local):
    def __init__(self) -> None:
        self.stack: list[Frame] = []


_frames = _Frames()


def push(interleaver: Interleaver, path: str) -> Frame | None:
    """Push an active-module frame for ``path`` under ``interleaver`` onto this thread's stack; return the frame for `pop`.

    When TorchDynamo is tracing, ``push`` records nothing and returns ``None``.
    Dynamo defers Python list mutations until graph exit, so a shim invoked from
    compiled code would observe an out-of-date stack. Compiled regions therefore
    run without frame bookkeeping and call the underlying kernel directly.
    """
    if torch.compiler.is_compiling():
        return None
    frame = Frame(interleaver, path, getcurrent())
    _frames.stack.append(frame)
    return frame


def pop(frame: Frame | None) -> None:
    """Pop ``frame`` from this thread's active-module stack; do nothing when ``frame`` is ``None``."""
    if frame is not None:
        _frames.stack.pop()


def requested(interleaver: Interleaver, module_path: str) -> bool:
    """Return whether ``interleaver`` is active and a worker is parked on ``module_path``'s kernel location.

    Calling an [`Envoy`][nnsight.intervention.envoy.Envoy] directly inside a
    trace block (``envoy(x)`` with ``hook=False``) sets ``interleaving`` to
    ``False`` for the duration of the call, so kernels executed inside that call
    do not serve sites.
    """
    return interleaver.interleaving and f"{module_path}{SUFFIX}" in interleaver.parked


def unshimmed(fn: Callable) -> Callable:
    """Return the underlying kernel function if ``fn`` is a kernel shim, or ``fn`` unchanged."""
    # Every shim is a Python function built by `shim`; the type check keeps unhashable values out of the lookup.
    return _shims[fn] if isinstance(fn, FunctionType) and fn in _shims else fn


def _serves(frame: Frame | None) -> bool:
    """Return whether a kernel call with ``frame`` at the top of the stack should serve a parked worker."""
    return frame is not None and requested(frame.interleaver, frame.path) and frame.greenlet is getcurrent()


def shim(entry: Entry) -> Callable:
    """Build the wrapper installed at ``entry``'s holders while a trace is open.

    When called, the shim checks the innermost module frame on the current
    thread. If a worker is parked on that module's ``.kernel`` location and the
    current greenlet is the one that entered the module, the shim runs
    ``entry.serve`` and delivers the resulting [`Sites`][nnsight.intervention.sites.Sites]
    reader to the location. Kernel calls made on another greenlet (such as a
    worker calling a kernel function on saved tensors while a second worker waits
    on the module's ``.kernel``) or with no parked waiter call ``entry.fn``
    directly.

    Because ``fn`` is the single free variable called by name inside the shim,
    ``.source`` peels the shim like a standard decorator (`peel_index` in
    intervention/source.py) to reach the underlying kernel function. For builtin
    kernels, which are not Python functions and have no source, the
    `unshimmed` lookup allows `decorator_chain` to step onto the
    builtin after peeling so `compiled` raises ``SourceNotAvailable``. When
    ``.source`` drills into a Python kernel and `rewrap` rebuilds this shim
    around the source-instrumented copy, ``fn`` holds that copy (`fn is not
    entry.fn`); the shim executes the instrumented copy and delivers an
    [`Unavailable`][nnsight.intervention.sites.Unavailable] reader to the
    module's kernel location. While TorchDynamo is compiling, the shim calls
    ``fn`` directly (see [`push`][nnsight.intervention.kernel.push]).
    """
    fn = entry.fn

    @functools.wraps(fn)
    def kernel_shim(*args: Any, **kwargs: Any) -> Any:
        if torch.compiler.is_compiling():
            return fn(*args, **kwargs)
        stack = _frames.stack
        frame = stack[-1] if stack else None
        if fn is not entry.fn:
            # Execute the source-instrumented copy before checking for a waiter,
            # because a worker reading inner `.source` operations only advances to
            # the `.kernel` request while this call is running.
            output = fn(*args, **kwargs)
            if _serves(frame):
                frame.interleaver.handle(
                    f"{frame.path}{SUFFIX}",
                    Unavailable(f"{fn.__name__!r} is drilled with .source in this trace; its kernel sites are not served on that call"),
                )
            return output
        if not _serves(frame):
            return fn(*args, **kwargs)
        output, sites = entry.serve(fn, args, kwargs)
        frame.interleaver.handle(f"{frame.path}{SUFFIX}", sites)
        return output

    _shims[kernel_shim] = fn  # lets `decorator_chain` step from the shim onto a builtin kernel
    return kernel_shim


def register(fn: Callable, serve: Callable, *holders: Attr) -> None:
    """Register ``serve`` as the site handler for ``fn`` at the storage locations ``holders``.

    ``serve(fn, args, kwargs)`` must return ``(output, sites)``, where ``output``
    is the unmodified return value of ``fn(*args, **kwargs)`` and ``sites`` is a
    [`Sites`][nnsight.intervention.sites.Sites] instance.

    Each entry in ``holders`` specifies a place from which the model looks up
    ``fn``, given as [`Attr`][nnsight.intervention.kernel.Attr] (a module
    attribute or closure cell) or [`Item`][nnsight.intervention.kernel.Item] (a
    dictionary entry). When ``fn`` is a Python function whose own module global
    currently holds ``fn``, that module attribute is added automatically. A
    builtin's module (``torch._C._nn`` for SDPA) is PyTorch's dispatch table,
    consulted by ``__torch_function__`` handling and by TorchDynamo, and is never
    patched. Calling
    ``register`` again for an already-registered ``fn`` adds any new holders to
    the existing entry; if traces are currently open, the new holders are patched
    immediately, and passing a shim read from an already-patched holder resolves
    to the underlying kernel function via [`unshimmed`][nnsight.intervention.kernel.unshimmed].
    """
    fn = unshimmed(fn)
    with _lock:
        entry = _KERNELS.get(fn)
        if entry is None:
            entry = _KERNELS[fn] = Entry(fn, serve)
        entry.serve = serve
        module = sys.modules.get(fn.__module__) if isinstance(fn, FunctionType) else None
        if module is not None and unshimmed(vars(module).get(fn.__name__)) is fn:
            holders = (Attr(module, fn.__name__), *holders)
        added = [holder for holder in holders if entry.add(holder)]
        if _open:
            _patch(entry, added)
        for copy in list(_copies):
            for name, value in list(copy.__globals__.items()):
                if _entry_of(value) is entry:
                    _hold(entry, copy, name)


def _entry_of(value: Any) -> Entry | None:
    """Return the `Entry` registered for ``value`` (a kernel function or its shim), or ``None``."""
    if not callable(value):
        return None
    try:
        return _KERNELS.get(unshimmed(value))
    except TypeError:  # an unhashable callable is not a registered kernel
        return None


def _hold(entry: Entry, fn: FunctionType, name: str) -> None:
    # Normalize the entry in the copied globals dict to `entry.fn` (in case the copy
    # was taken while a trace had installed the shim), then patch it if traces are open.
    # A copy that is already a holder is left as it is: it holds the shim while traces are open.
    if not entry.add(Global(fn, name)):
        return
    fn.__globals__[name] = entry.fn
    if _open:
        _patch(entry, [entry.holders[-1]])


def adopt(fn: FunctionType) -> None:
    """Track ``fn`` (built by `function_like` with a copy of its module's globals) so kernels in its globals reach their shims.

    Each registered kernel present in ``fn.__globals__`` is added to its entry as a
    [`Global`][nnsight.intervention.kernel.Global] holder, and ``fn`` is recorded in
    `_copies` so kernels registered later also add holders for ``fn``. Because both
    `_copies` and `Global` reference ``fn`` weakly, the holders expire when ``fn``
    is garbage-collected.
    """
    with _lock:
        _copies.add(fn)
        for name, value in list(fn.__globals__.items()):
            entry = _entry_of(value)
            if entry is not None:
                _hold(entry, fn, name)


def _patch(entry: Entry, holders: list[Attr]) -> None:
    # Only overwrite holders that currently hold `entry.fn`; leave prior custom replacements untouched.
    for holder in holders:
        if holder.get() is entry.fn:
            holder.set(entry.shim)
            entry.patched.append(holder)


def _restore(entry: Entry) -> None:
    # Visit every patched holder even if restoring an earlier one raises, so no shim remains installed after the trace exits.
    patched, entry.patched = entry.patched, []
    for holder in patched:
        if not holder.alive:
            continue
        try:
            if holder.get() is entry.shim:
                holder.set(entry.fn)
            else:
                warnings.warn(f"{holder!r} no longer holds the kernel shim of {entry.fn.__name__!r}; left as it is")
        except Exception as error:
            warnings.warn(f"{holder!r} could not be restored to {entry.fn.__name__!r}: {error!r}")


def install() -> None:
    """Install kernel shims across all registered holders when the first interleaver opens."""
    global _open
    with _lock:
        _open += 1
        if _open == 1:
            for entry in _KERNELS.values():
                _patch(entry, entry.holders)


def uninstall() -> None:
    """Restore original kernel functions across all patched holders when the last open interleaver closes."""
    global _open
    with _lock:
        _open -= 1
        if _open == 0:
            for entry in _KERNELS.values():
                _restore(entry)


# ---------------------------------------------------------------------------
# Catalogs of supported kernels
# ---------------------------------------------------------------------------


#: Pairs of ``(function, holders)`` yielded by a catalog locator.
Located = Iterable[tuple[Callable, tuple[Attr, ...]]]


class Spec(NamedTuple):
    """Catalog entry declaring how to locate and serve a supported kernel.

    ``locate(context)`` yields ``(function, holders)`` for every function this row
    covers in ``context`` (the value passed to `register_catalog`, such as the
    modeling modules of one model). A row may yield nothing when no matching
    function is present (for example, when an optional kernel library is not
    installed). ``name`` and ``family`` are descriptive metadata; `register_catalog`
    uses only ``locate`` and ``serve``.
    """

    #: Label for this row in the supported-kernels table of docs/usage/kernel.md. One row may cover several functions.
    name: str
    #: Site family served by this row's reader (for example, ``"attention"``).
    family: str
    #: Callable returning ``(function, holders)`` pairs for a given ``context``.
    locate: Callable[[Any], Located]
    #: Serve function passed to `register` for each located function.
    serve: Callable


def attribute(obj: Any, name: str) -> Callable[[Any], Located]:
    """Build a locator for a kernel read as attribute ``name`` of ``obj`` on every call. The attribute must exist on ``obj``."""

    def locate(context: Any) -> Located:
        yield unshimmed(getattr(obj, name)), (Attr(obj, name),)

    return locate


def register_catalog(catalog: Iterable[Spec], context: Any = None) -> None:
    """Register every function each row of ``catalog`` locates in ``context``."""
    for spec in catalog:
        for fn, holders in spec.locate(context):
            register(fn, spec.serve, *holders)
