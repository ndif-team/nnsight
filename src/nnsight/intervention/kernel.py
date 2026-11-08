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

* A global registry maps each fused callable to a *serve function*. When `run_op`
  (in intervention/source.py) intercepts a call to a registered callable and a
  worker is waiting on the enclosing module's kernel location, it runs the serve
  function. The serve function returns the kernel's normal output along with a
  [`Sites`][nnsight.intervention.kernel.Sites] reader that holds the captured
  inputs and statistics. `run_op` then passes this reader to the location
  ``"{module}.kernel"``. The reader only computes a site when a worker requests
  it.
* A pass-through set (`_THROUGH`) lists Python wrapper functions (such as
  transformers' attention interfaces) that `run_op` automatically instruments on
  the fly when a worker is waiting on the module's kernel location. This ensures
  that the inner kernel call is intercepted after preprocessing steps like RoPE,
  KV-cache updates, and tensor layout transpositions.
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
`Mediator.occurrence`). That counter advances on every call to the module even on
steps where the module's forward was uninstrumented, so a request scoped to step
``n`` via ``tracer.iter`` always matches step ``n``.

Limitations:
* Sites are read-only, and returned tensors are detached from autograd.
* Because sites are served when the kernel executes, a site must be requested
  before reading any value produced later in the forward pass.
* Accessing ``envoy.kernel`` instruments the module's forward pass. If ``.kernel``
  is accessed for the first time while the module is already in the middle of an
  uninstrumented forward call (for example, after reading a submodule inside the
  same parent module), the request raises `OutOfOrderError`.
"""

from __future__ import annotations

import math
import operator
import warnings
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import torch
from torch._subclasses.fake_tensor import FakeTensor

from .interleaver import Event, Interleaver, Mediator, OutOfOrderError

if TYPE_CHECKING:
    from .envoy import Envoy

#: Maps each registered fused callable to its serve function ``(fn, args, kwargs) -> (output, Sites)``.
_KERNELS: dict[Callable, Callable] = {}

#: Python wrapper functions that `run_op` instruments on the fly to reach a registered kernel inside them.
_THROUGH: set[Callable] = set()

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


def register(fn: Callable, serve: Callable) -> None:
    """Register ``serve`` as the site handler for ``fn``.

    ``serve(fn, args, kwargs)`` must return ``(output, sites)``, where ``output``
    is the unmodified return value of ``fn(*args, **kwargs)`` and ``sites`` is a
    [`Sites`][nnsight.intervention.kernel.Sites] instance.
    """
    _KERNELS[fn] = serve


def through(*fns: Callable) -> None:
    """Mark wrapper functions ``fns`` to be instrumented on the fly when a worker waits on their module's kernel."""
    _THROUGH.update(fns)


def _registered(fn: Callable, table: dict | set) -> Any:
    # Unhashable callables cannot be in the registry; return None instead of raising TypeError during a forward pass.
    try:
        return table.get(fn) if isinstance(table, dict) else fn in table
    except TypeError:
        return None


def serving(fn: Callable) -> Callable | None:
    """Return the serve function registered for ``fn``, or ``None``."""
    return _registered(fn, _KERNELS)


def passes_through(fn: Callable) -> bool:
    """Return whether ``fn`` is a registered wrapper on the path to a kernel."""
    return bool(_registered(fn, _THROUGH))


def requested(interleaver: Interleaver, module_path: str) -> bool:
    """Return whether any worker is currently parked on ``module_path``'s kernel location."""
    return f"{module_path}{SUFFIX}" in interleaver.parked


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

    def __init__(self, envoy: Envoy, instrumented: bool) -> None:
        self._envoy = envoy
        # Records whether the module's forward was already source-instrumented
        # before this `.kernel` access, so `_request` can detect if the module was
        # already mid-call in an uninstrumented forward pass.
        self._instrumented = instrumented

    def __getattr__(self, site: str) -> Callable[..., torch.Tensor]:
        if site.startswith("_"):
            raise AttributeError(site)
        return partial(self._request, site)

    def _request(self, site: str, **spec: Any) -> torch.Tensor:
        path = self._envoy.path
        spec = _normalize(spec)
        location = f"{path}{SUFFIX}"
        mediator = Mediator.current(location)
        if mediator.occurrence(f"{path}.input") > mediator.occurrence(f"{path}.output") and not self._instrumented:
            # The module is already inside its forward pass (because a submodule
            # inside it was read earlier in the same trace block), and it entered
            # that call before `.kernel` instrumented the forward.
            raise OutOfOrderError(
                f"'{location}' was requested while {path!r} is already running without an "
                f"instrumented forward; request the site before reading the module's output "
                f"or anything inside it, or access `.kernel` once before the trace"
            )
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
