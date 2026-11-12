"""Kernel sites: what a served kernel call offers a trace, and how a trace asks for it.

A kernel *site* is one value a fused kernel call can serve, such as ``probs`` or
``lse``. The serve function of a kernel (see intervention/kernel.py for how a
call reaches it) returns a [`Sites`][nnsight.intervention.sites.Sites] reader
holding the call's captured inputs and exported statistics; the reader computes
a site only when a worker requests it.

On the user side, [`Envoy.kernel`][nnsight.intervention.envoy.Envoy.kernel]
returns a [`Kernel`][nnsight.intervention.sites.Kernel] accessor whose
attributes correspond to site names. Calling a site parks the worker on
``"{module}.kernel"``, receives the reader when the kernel runs, computes the
requested slice on the worker thread, and narrows the result to the worker's
batch rows.

Every tensor returned by a site has a
[`Provenance`][nnsight.intervention.sites.Provenance] attached at ``.provenance``
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
  modeling/kernels/recurrent.py).

This module also holds the slicing helpers readers share (`indices`, `select`,
`positions`, `offsets`, `squeeze`, `check_axes`, `warn_if_large`), so every
family applies the same indexing rules to its axis keywords.
"""

from __future__ import annotations

import math
import operator
import warnings
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import torch
from torch._subclasses.fake_tensor import FakeTensor

from .interleaver import Event, Mediator

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
    the requested tensor with a [`Provenance`][nnsight.intervention.sites.Provenance]
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
