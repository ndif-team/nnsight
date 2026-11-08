"""Read values a fused kernel computes but never writes out.

A fused kernel takes its inputs from memory and writes its output back, and
everything in between lives in registers: FlashAttention never materializes the
attention probabilities, a chunked state-space kernel never forms the state at a
token inside a chunk. Those values are what interpretability methods read, and the
usual way to get them, swapping the kernel for an eager implementation, changes
the arithmetic: under bf16 an eager attention's probabilities differ from the
kernel's by up to 19% at the top key (Qwen3-8B, 56-token prompt; see
docs/usage/kernel.md), while the model's logits move by a KL of 1e-3. The eager
value belongs to a different computation from the one the model runs.

The kernel's own backward pass already solves this. It saves a small *sufficient
statistic* (attention saves the per-row logsumexp, a chunked scan saves its
chunk-boundary states) and recomputes the intermediate from the inputs and that
statistic. This module lets a trace do the same: run the kernel the model would
have run, keep the statistic the wrapper drops, and reconstruct only the slice the
user asked for. A *site* is one such reconstructible value (``probs``, ``lse``).

Mechanism, on the one primitive the interleaver has (a location string and
[`Interleaver.handle`][nnsight.intervention.interleaver.Interleaver.handle]):

* A registry maps a fused callable to a *serve function*. `run_op` (see
  intervention/source.py) looks the callable of every instrumented call up here;
  a match runs the serve function instead of the plain call. It returns the same
  output the call would have and a [`Sites`][nnsight.intervention.kernel.Sites]
  reader holding the kernel's inputs and statistics, which `run_op` hands to
  ``"{module}.kernel"``, one location per module, on every fire. The reader
  computes a site when a worker asks, so a fire nobody reads costs the capture
  and nothing else.
* A through set names Python wrappers (transformers' attention interfaces) to
  drill into on the way down when a worker is parked on the module's kernel
  location, so the kernel call inside them reaches `run_op` with the arguments the
  kernel sees: after RoPE, after the cache update, in the kernel's layout.
* The worker side is [`Envoy.kernel`][nnsight.intervention.envoy.Envoy.kernel]: a
  [`Kernel`][nnsight.intervention.kernel.Kernel] accessor whose attributes are
  sites. A call parks the worker on the module's kernel location, receives the
  reader, computes its slice there, in the worker, and narrows it to its own
  batch rows. Index errors and unknown sites therefore raise on the user's line.

.. code-block:: python

    with model.trace(prompt):
        p = model.model.layers[6].self_attn.kernel.probs(heads=4, query=-1).save()

    p.provenance   # Provenance(kind='exact reconstruction', kernel='aten._scaled_dot_product_flash_attention', ...)

Every served value carries a [`Provenance`][nnsight.intervention.kernel.Provenance]
naming what kind of value it is: an *exact export* the kernel wrote (the
logsumexp, a chunk-boundary state), an *exact reconstruction* from exported
statistics (probabilities as ``exp(S - lse)`` in fp32, the formula the kernel's
backward uses), a *replay* when the kernel exported nothing and the value was
recomputed in fp32 beside it, or *derived* when the value is absent from the
kernel's algorithm and is stepped from an exported statistic by the layer's own
recurrence (the state at a token inside a chunk; see intervention/recurrent.py).
The site classes live beside this module: intervention/attention.py (SDPA),
intervention/recurrent.py (gated delta rule, Mamba2), intervention/experts.py
(grouped mixture-of-experts).

The kernel fires inside the module's call, so the kernel location's occurrence
is the module's: the interleaver counts ``"{module}.kernel"`` as
``"{module}.output"`` (see `Mediator.occurrence`), which advances on every call of
the module whether or not its forward is instrumented. A ``tracer.iter`` request
pinned to step ``n`` is therefore served at step ``n``.

Constraints: reads only, and the served tensors carry no gradient; a site is
served when the kernel fires, so it must be requested before anything later in
the forward is read, like every other location; the module's forward is
instrumented by the first access to ``envoy.kernel``, which is in time unless the
module is already running (a read inside it earlier in the same block), in which
case the request raises and says so.
"""

from __future__ import annotations

import math
import operator
import warnings
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import torch

from .interleaver import Event, Interleaver, Mediator, OutOfOrderError

if TYPE_CHECKING:
    from .envoy import Envoy

#: Fused callable -> serve function ``(fn, args, kwargs) -> (output, Sites)``.
_KERNELS: dict[Callable, Callable] = {}

#: Python wrappers drilled into on the way down to a registered kernel.
_THROUGH: set[Callable] = set()

#: Location suffix of a module's kernel sites; counted as the module's ``.output``.
SUFFIX = ".kernel"

#: A requested slice whose fp32 result would exceed this many bytes warns first.
_LARGE_SITE_BYTES = 1 << 30


class Provenance(NamedTuple):
    """What kind of value a kernel site served, and from which kernel."""

    #: ``"exact export"`` for a tensor the kernel wrote, ``"exact reconstruction"``
    #: for one computed from exported statistics by an identity, ``"replay"`` for
    #: one recomputed in fp32 beside a kernel that exported nothing usable,
    #: ``"derived"`` for one the kernel's algorithm never forms, stepped from an
    #: exported statistic by the layer's own recurrence (the note carries the
    #: measured drift of that stepping).
    kind: str
    #: The kernel that ran, by name.
    kernel: str
    #: The formula or caveat a reader needs to interpret the value.
    note: str


# A saved site loads under ``torch.load``'s default ``weights_only=True``.
torch.serialization.add_safe_globals([Provenance])


class Batch(NamedTuple):
    """The requesting worker's rows in the trace's combined batch."""

    #: Rows in the combined forward (every invoke's, stacked on dim 0).
    total: int
    #: The worker's first row.
    start: int
    #: The worker's row count.
    size: int


class Sites:
    """What a serve function hands the module's kernel location: one fire's reader.

    A subclass lists its site names in ``names`` and defines one method per site,
    ``name(spec) -> (tensor, Provenance)``, taking the normalized slice spec. The
    reader holds the kernel's inputs and statistics until the last worker served
    on this fire lets go of it. ``batch`` is the worker's rows in the trace's
    batch when the trace has a batcher (every `TransformersModel` trace does, with
    one invoke too), else ``None``; a reader whose rows are batch-major ignores it
    and leaves the narrowing to the worker, one whose rows are a flattened token
    axis uses it to serve the worker's own rows.
    """

    names: tuple[str, ...] = ()

    def read(self, site: str, spec: dict[str, Any], batch: Batch | None) -> tuple[torch.Tensor, Provenance]:
        if site not in self.names:
            raise ValueError(f"this kernel has no site {site!r}; available: {', '.join(self.names)}")
        return getattr(self, site)(spec)


class Unavailable(Sites):
    """What a serve function hands over when this call's sites cannot be read.

    The kernel ran as the model called it; the reason surfaces on the worker's
    line, at the request, as a ``ValueError``.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def read(self, site: str, spec: dict[str, Any], batch: Batch | None) -> tuple[torch.Tensor, Provenance]:
        raise ValueError(self.reason)


def register(fn: Callable, serve: Callable) -> None:
    """Serve ``fn``'s sites with ``serve``.

    ``serve(fn, args, kwargs)`` returns ``(output, sites)``: what ``fn(*args,
    **kwargs)`` returns, unchanged, and a [`Sites`][nnsight.intervention.kernel.Sites].
    """
    _KERNELS[fn] = serve


def through(*fns: Callable) -> None:
    """Drill into ``fns`` on the way down when a worker waits on their module's kernel."""
    _THROUGH.update(fns)


def _registered(fn: Callable, table: dict | set) -> Any:
    # A callable that cannot be hashed is not a kernel; say so rather than raise
    # in the middle of a forward.
    try:
        return table.get(fn) if isinstance(table, dict) else fn in table
    except TypeError:
        return None


def serving(fn: Callable) -> Callable | None:
    """The serve function registered for ``fn``, or ``None``."""
    return _registered(fn, _KERNELS)


def passes_through(fn: Callable) -> bool:
    """Whether ``fn`` is a wrapper to drill into on the way to a kernel."""
    return bool(_registered(fn, _THROUGH))


def requested(interleaver: Interleaver, module_path: str) -> bool:
    """Whether a worker is parked on ``module_path``'s kernel location."""
    return f"{module_path}{SUFFIX}" in interleaver.parked


def warn_if_large(shape: tuple[int, ...], site: str, axes: str, dtype: torch.dtype = torch.float32) -> None:
    """Warn before materializing a site whose result exceeds `_LARGE_SITE_BYTES`; ``axes`` names its slice arguments."""
    size = math.prod(shape) * dtype.itemsize
    if size > _LARGE_SITE_BYTES:
        warnings.warn(
            f"kernel site {site!r} materializes {size / (1 << 30):.1f} GiB in {dtype}; pass {axes} to request a slice",
            stacklevel=3,
        )


def check_axes(spec: dict[str, Any], site: str, allowed: tuple[str, ...]) -> None:
    """Raise ``ValueError`` for a spec key ``site`` does not take."""
    unknown = set(spec) - set(allowed)
    if unknown:
        raise ValueError(f"kernel site {site!r} takes {', '.join(allowed)}; got {', '.join(sorted(unknown))}")


def squeeze(value: torch.Tensor, spec: dict[str, Any], axes: dict[str, int]) -> torch.Tensor:
    """Drop each axis in ``axes`` (name -> dim) whose spec entry is a single int, highest dim first."""
    for name, axis in sorted(axes.items(), key=lambda item: -item[1]):
        if isinstance(spec.get(name), int):
            value = value.squeeze(axis)
    return value


# ---------------------------------------------------------------------------
# Slice specs: what the worker asks for, in a form a site method can index by
# ---------------------------------------------------------------------------


def _normalize(spec: dict[str, Any]) -> dict[str, Any]:
    """The spec as plain Python values.

    An index (anything ``operator.index`` accepts, a 0-d integer tensor) stays a
    single position and drops its axis, as torch indexing does; a ``slice`` stays a
    slice; a list or tensor of indices becomes a tuple of ints, a boolean tensor
    the positions where it is ``True``; ``None`` is the whole axis. A bool, a float
    or a float tensor is refused: torch gives each a different meaning from an
    index, and none of them is a slice of a head or a position.
    """
    normalized: dict[str, Any] = {}
    for name, value in spec.items():
        if isinstance(value, torch.Tensor):
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
    """A normalized spec entry as something a tensor dim can be indexed by.

    ``None`` is the whole axis, an int one position (a slice of length one, so the
    axis is dropped afterwards, once every site is computed), a slice itself, and a
    tuple of ints a ``LongTensor`` for ``index_select``, created on ``device`` (the
    tensor it will index, so no host-to-device copy happens on the request). Positions
    are bounds-checked here, where the request's traceback points at the user's line.
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
    """``tensor`` narrowed along ``dim`` by a slice or gathered by an index tensor."""
    if isinstance(index, slice):
        return tensor[(slice(None),) * dim + (index,)]
    return tensor.index_select(dim, index)


def positions(index: slice | torch.Tensor, size: int, device: torch.device | None = None) -> torch.Tensor:
    """The positions along an axis of ``size`` that ``index`` selects, as a tensor on ``device``."""
    return torch.arange(size, device=device)[index] if isinstance(index, slice) else index


# ---------------------------------------------------------------------------
# The worker side
# ---------------------------------------------------------------------------


class Kernel:
    """The sites inside the kernel an envoy's module calls, reached as ``envoy.kernel``.

    An attribute is a site; calling it with the slice to read parks the worker
    until the kernel fires and returns the value, a real tensor carrying a
    [`Provenance`][nnsight.intervention.kernel.Provenance] as ``.provenance``::

        attn.kernel.probs(heads=4, query=-1)
        attn.kernel.lse()

    Which sites exist depends on the kernel; asking for one it does not define
    raises, naming the ones it has.
    """

    def __init__(self, envoy: Envoy, instrumented: bool) -> None:
        self._envoy = envoy
        # Whether the module's forward was instrumented before this access, so a
        # call already in progress is running the instrumented body.
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
            # The module is mid-call (a value inside it, or its output, was read
            # first) with the forward it had before this request instrumented it,
            # so this call's kernel will not be seen.
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
        # Set here, on the worker's own view: a view carries no attributes over.
        value.provenance = provenance
        return value

    def __repr__(self) -> str:
        return f"Kernel({self._envoy.path!r})"
