"""Expose intermediate values from fused kernels during a trace.

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
  with a [`Sites`][nnsight.intervention.kernel.Sites] reader holding the
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
  returns a [`Kernel`][nnsight.intervention.kernel.Kernel] accessor whose
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

Every tensor returned by a site has a
[`Provenance`][nnsight.intervention.kernel.Provenance] attached at ``.provenance``
that records how the value was obtained:

* ``"exact export"``: a tensor written directly by the kernel (such as ``lse`` or
  a chunk-boundary state).
* ``"exact reconstruction"``: a value computed in fp32 from the kernel's inputs
  and exported statistics using the same mathematical identity the kernel uses in
  its backward pass (such as attention probabilities via ``exp(S - lse)``).
* ``"replay"``: a value recomputed in fp32 alongside a kernel that did not export
  usable statistics.
* ``"derived"``: a quantity that the chunked algorithm never forms directly,
  computed by advancing an exported boundary state using the layer's intra-chunk
  recurrence (such as a per-token state inside a chunk; see
  intervention/recurrent.py).

Site implementations are organized by kernel family in sibling modules:
intervention/attention.py (SDPA), intervention/recurrent.py (gated delta rule,
Kimi delta attention, Mamba2, Mamba1), and intervention/experts.py (grouped
mixture-of-experts).

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
  [`register`][nnsight.intervention.kernel.register] as a holder. A forward pass
  instrumented by ``.source`` runs with a copy of its module's globals dict
  created at instrumentation time (`function_like` in intervention/source.py), so
  a kernel called as a bare global of that same module is read from the copied
  dict and does not reach the shim; a kernel looked up as a module attribute, a
  dict entry, or a closure cell does reach the shim.
* Inside a ``torch.compile`` region, the module stack is not updated and the shim
  calls the kernel directly, so a site requested on a compiled module fails with
  ``OutOfOrderError`` ("requested but the model already ran past it"). If the
  kernel call itself runs eagerly inside a compiled module (for example, inside a
  helper decorated with ``torch._dynamo.disable``), the call is attributed to the
  innermost enclosing module that is running eagerly.
"""

from __future__ import annotations

import functools
import math
import operator
import sys
import threading
import warnings
from functools import partial
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import torch
from greenlet import getcurrent, greenlet
from torch._subclasses.fake_tensor import FakeTensor

from .interleaver import Event, Interleaver, Mediator

if TYPE_CHECKING:
    from .envoy import Envoy

#: Location suffix for a module's kernel sites, counted on the same schedule as the module's ``.output``.
SUFFIX = ".kernel"

#: Threshold in bytes above which materializing a requested site slice emits a warning.
_LARGE_SITE_BYTES = 1 << 30


class Provenance(NamedTuple):
    """Metadata describing how a kernel site value was computed and which kernel produced it."""

    #: ``"exact export"`` for a tensor written directly by the kernel;
    #: ``"exact reconstruction"`` for a tensor computed from exported statistics
    #: via an exact identity; ``"replay"`` for a tensor recomputed in fp32
    #: alongside a kernel that exported no usable statistics; ``"derived"`` for a
    #: tensor that the kernel never forms directly, computed by advancing an
    #: exported boundary state with the layer's intra-chunk recurrence (with the
    #: measured boundary drift reported in ``note``).
    kind: str
    #: Name of the kernel that ran.
    kernel: str
    #: Formula or caveat needed to interpret the value.
    note: str


# Allow saved site tensors carrying `.provenance` to load under `torch.load(..., weights_only=True)`.
torch.serialization.add_safe_globals([Provenance])


class Batch(NamedTuple):
    """Row range belonging to the requesting worker within the trace's combined batch."""

    #: Total batch rows across all invokes in the combined forward pass.
    total: int
    #: Starting row index for the requesting worker.
    start: int
    #: Number of rows belonging to the requesting worker.
    size: int


class Sites:
    """Reader for a single kernel execution, delivered to ``"{module}.kernel"``.

    Subclasses list their supported site names in ``names`` and implement one
    method per site with signature ``name(spec) -> (tensor, Provenance)``, where
    ``spec`` is a normalized slice dictionary. The reader retains the kernel's
    captured inputs and exported statistics until all workers waiting on that
    kernel execution have read their slices.

    When the trace uses a batcher (which includes single-invoke
    `TransformersModel` traces), ``batch`` describes the requesting worker's row
    range in the combined batch; otherwise ``batch`` is ``None``. Readers whose
    outputs have batch as dimension 0 ignore ``batch`` and let `Kernel._request`
    narrow the result afterwards. Readers whose internal buffers flatten the batch
    and sequence dimensions use ``batch`` to reshape and select the worker's rows.
    """

    names: tuple[str, ...] = ()

    def read(self, site: str, spec: dict[str, Any], batch: Batch | None) -> tuple[torch.Tensor, Provenance]:
        if site not in self.names:
            raise ValueError(f"this kernel has no site {site!r}; available: {', '.join(self.names)}")
        return getattr(self, site)(spec)


class Unavailable(Sites):
    """Placeholder reader returned when a kernel call's sites cannot be served.

    The kernel itself still runs normally. If a worker requests a site from this
    call, `read` raises ``ValueError`` with ``reason`` on the worker's line.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def read(self, site: str, spec: dict[str, Any], batch: Batch | None) -> tuple[torch.Tensor, Provenance]:
        raise ValueError(self.reason)


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

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.obj!r}, {self.key!r})"


class Item(Attr):
    """Storage location that holds a kernel function as a dictionary entry. Restoring writes the original value back without deleting the key."""

    def get(self) -> Any:
        return self.obj[self.key]

    def set(self, value: Any) -> None:
        self.obj[self.key] = value


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
    return getattr(fn, "__kernel_shim__", fn)


def _serves(frame: Frame | None) -> bool:
    """Return whether a kernel call with ``frame`` at the top of the stack should serve a parked worker."""
    return frame is not None and requested(frame.interleaver, frame.path) and frame.greenlet is getcurrent()


def shim(entry: Entry) -> Callable:
    """Build the wrapper installed at ``entry``'s holders while a trace is open.

    When called, the shim checks the innermost module frame on the current
    thread. If a worker is parked on that module's ``.kernel`` location and the
    current greenlet is the one that entered the module, the shim runs
    ``entry.serve`` and delivers the resulting [`Sites`][nnsight.intervention.kernel.Sites]
    reader to the location. Kernel calls made on another greenlet (such as a
    worker calling a kernel function on saved tensors while a second worker waits
    on the module's ``.kernel``) or with no parked waiter call ``entry.fn``
    directly.

    Because ``fn`` is the single free variable called by name inside the shim,
    ``.source`` peels the shim like a standard decorator (`peel_index` in
    intervention/source.py) to reach the underlying kernel function. For builtin
    kernels, which are not Python functions and have no source, the
    ``__kernel_shim__`` attribute allows `decorator_chain` to step onto the
    builtin after peeling so `compiled` raises ``SourceNotAvailable``. When
    ``.source`` drills into a Python kernel and `rewrap` rebuilds this shim
    around the source-instrumented copy, ``fn`` holds that copy (`fn is not
    entry.fn`); the shim executes the instrumented copy and delivers an
    [`Unavailable`][nnsight.intervention.kernel.Unavailable] reader to the
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

    kernel_shim.__kernel_shim__ = fn  # lets `decorator_chain` step from the shim onto a builtin kernel
    return kernel_shim


def register(fn: Callable, serve: Callable, *holders: Attr) -> None:
    """Register ``serve`` as the site handler for ``fn`` at the storage locations ``holders``.

    ``serve(fn, args, kwargs)`` must return ``(output, sites)``, where ``output``
    is the unmodified return value of ``fn(*args, **kwargs)`` and ``sites`` is a
    [`Sites`][nnsight.intervention.kernel.Sites] instance.

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


def warn_if_large(shape: tuple[int, ...], site: str, axes: str, dtype: torch.dtype = torch.float32) -> None:
    """Emit a warning if materializing ``shape`` in ``dtype`` exceeds `_LARGE_SITE_BYTES`."""
    size = math.prod(shape) * dtype.itemsize
    if size > _LARGE_SITE_BYTES:
        warnings.warn(
            f"kernel site {site!r} materializes {size / (1 << 30):.1f} GiB in {dtype}; pass {axes} to request a slice",
            stacklevel=3,
        )


def check_axes(spec: dict[str, Any], site: str, allowed: tuple[str, ...]) -> None:
    """Raise ``ValueError`` if ``spec`` contains axis names outside ``allowed``."""
    unknown = set(spec) - set(allowed)
    if unknown:
        raise ValueError(f"kernel site {site!r} takes {', '.join(allowed)}; got {', '.join(sorted(unknown))}")


def squeeze(value: torch.Tensor, spec: dict[str, Any], axes: dict[str, int]) -> torch.Tensor:
    """Squeeze each axis in ``axes`` (mapping axis name to dimension) that was indexed by a single int, from highest dimension to lowest."""
    for name, axis in sorted(axes.items(), key=lambda item: -item[1]):
        if isinstance(spec.get(name), int):
            value = value.squeeze(axis)
    return value


# ---------------------------------------------------------------------------
# Slice specification helpers
# ---------------------------------------------------------------------------


def _normalize(spec: dict[str, Any]) -> dict[str, Any]:
    """Convert user-supplied axis indices in ``spec`` into canonical Python types.

    Single integer indices (anything accepted by ``operator.index`` or a 0-d
    integer tensor) become ``int`` and cause that axis to be squeezed later. A
    ``slice`` is kept as a ``slice``. A list, tuple, or 1-D+ integer tensor
    becomes a tuple of ``int``s, and a boolean tensor is converted to the
    positions where it is ``True``. ``None`` selects the full axis. Booleans,
    floats, and floating-point or complex tensors are rejected with ``TypeError``.
    """
    normalized: dict[str, Any] = {}
    for name, value in spec.items():
        if isinstance(value, torch.Tensor):
            if isinstance(value, FakeTensor):
                raise TypeError(f"kernel site index {name}= is a tensor created under scan, whose values are unknown; pass an int, slice or list")
            if value.dtype is torch.bool:
                value = value.nonzero().flatten()
            if value.is_floating_point() or value.is_complex():
                raise TypeError(f"kernel site index {name}= is a {value.dtype} tensor; pass integer indices")
            value = int(value) if value.dim() == 0 else tuple(int(index) for index in value.flatten().tolist())
        if value is None or isinstance(value, slice):
            normalized[name] = value
        elif isinstance(value, bool):
            raise TypeError(f"kernel site index {name}={value!r} is a bool; pass an int, slice, list or tensor")
        elif isinstance(value, (list, tuple)):
            normalized[name] = tuple(operator.index(index) for index in value)
        else:
            try:
                normalized[name] = operator.index(value)
            except TypeError:
                raise TypeError(f"kernel site index {name}={value!r} is not an int, slice, list or tensor") from None
    return normalized


def indices(value: Any, size: int, device: torch.device | None = None) -> slice | torch.Tensor:
    """Convert a normalized spec entry for an axis of length ``size`` into a ``slice`` or index tensor.

    ``None`` becomes ``slice(None)``. A single ``int`` becomes a unit-length
    ``slice`` so the dimension stays in place during computation and can be
    removed at the end by `squeeze`. A ``slice`` is returned as-is. A tuple of
    ``int``s becomes a ``torch.long`` tensor on ``device`` for ``index_select``.
    Integer indices are bounds-checked here so that ``IndexError`` points to the
    user's line in the trace.
    """
    if value is None:
        return slice(None)
    if isinstance(value, slice):
        return value
    if isinstance(value, int):
        if not -size <= value < size:
            raise IndexError(f"kernel site index {value} is out of range for an axis of size {size}")
        return slice(value % size, value % size + 1)
    if any(not -size <= index < size for index in value):
        raise IndexError(f"kernel site indices {value} are out of range for an axis of size {size}")
    return torch.tensor([index % size for index in value], dtype=torch.long, device=device)


def select(tensor: torch.Tensor, dim: int, index: slice | torch.Tensor) -> torch.Tensor:
    """Slice ``tensor`` along ``dim`` when ``index`` is a ``slice``, or gather with ``index_select`` when it is a tensor."""
    if isinstance(index, slice):
        return tensor[(slice(None),) * dim + (index,)]
    return tensor.index_select(dim, index)


def positions(index: slice | torch.Tensor, size: int, device: torch.device | None = None) -> torch.Tensor:
    """Return the selected positions along an axis of length ``size`` as a 1-D tensor on ``device``."""
    return torch.arange(size, device=device)[index] if isinstance(index, slice) else index


def offsets(value: Any, size: int) -> list[int]:
    """The positions along an axis of ``size`` that a normalized spec entry selects, as Python ints.

    Built without a tensor, so it also works under ``model.scan`` (fake tensors carry no values).
    """
    index = indices(value, size)
    if isinstance(index, slice):
        return list(range(size))[index]
    return [position % size for position in value]


# ---------------------------------------------------------------------------
# Worker-side accessor
# ---------------------------------------------------------------------------


class Kernel:
    """Accessor for kernel sites inside a module, reached as ``envoy.kernel``.

    Each attribute on `Kernel` names a site. Calling that attribute with optional
    slice keyword arguments parks the worker until the kernel executes and returns
    the requested tensor with a [`Provenance`][nnsight.intervention.kernel.Provenance]
    at ``.provenance``::

        attn.kernel.probs(heads=4, query=-1)
        attn.kernel.lse()

    Available site names depend on the kernel that runs inside the module.
    Requesting an unknown site raises ``ValueError`` listing the available sites.
    """

    def __init__(self, envoy: Envoy) -> None:
        self._envoy = envoy

    def __getattr__(self, site: str) -> Callable[..., torch.Tensor]:
        if site.startswith("_"):
            raise AttributeError(site)
        return partial(self._request, site)

    def _request(self, site: str, **spec: Any) -> torch.Tensor:
        path = self._envoy.path
        spec = _normalize(spec)
        location = f"{path}{SUFFIX}"
        mediator = Mediator.current(location)
        sites = Mediator.event(Event.VALUE, location)
        batcher = mediator.interleaver.batcher
        batch = None
        if batcher is not None:
            start, size = mediator.batch_group if mediator.batch_group is not None else (0, batcher.total)
            batch = Batch(batcher.total, start, size)
        value, provenance = sites.read(site, spec, batch)
        if batcher is not None:
            value = batcher.narrow(value, mediator.batch_group)
        # Attach provenance after batch narrowing because slicing a tensor creates a view without custom attributes.
        value.provenance = provenance
        return value

    def __repr__(self) -> str:
        return f"Kernel({self._envoy.path!r})"
