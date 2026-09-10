"""Batch several invoke inputs into one forward and scope interventions to rows.

``with model.trace() as tracer:`` may contain several ``with tracer.invoke(x):``
blocks. Their inputs are combined into a single batched forward, and each block's
interventions see its own rows of an activation the batcher can read the rows of.

A [`Batcher`][nnsight.intervention.batching.Batcher] (one per trace) collects each invoke's input and assigns it a
``batch_group`` — a ``[start, size]`` row range in the combined batch. At run time
[`Batcher.narrow`][nnsight.intervention.batching.Batcher.narrow] slices a full batched activation down to a block's rows when
it reads, and [`Batcher.widen`][nnsight.intervention.batching.Batcher.widen] splices an edit back into the full tensor. The row math
is dim-0 only; the model's `_batch` equalizes everything else (e.g. sequence
length) when it builds the combined input.

Which values that math applies to is decided by the leading dim, so it is a rule
about shapes rather than about provenance. A value is scoped when its leading dim
is the combined batch size, or a whole multiple of it — the multiple covers a model
that folds tokens (or tokens and heads) into the batch axis before a module, as
every transformers MoE block does ahead of its router. A leading dim that is
neither is a layout this batcher cannot read: the value is served whole to every
invoke, and a write to it — an in-place edit or a replacement — is reported (see
[`Batcher.narrow`][nnsight.intervention.batching.Batcher.narrow]), because it acts
on the whole batch rather than on the invoke that made it.

Two consequences a block's author meets. Equalizing the sequence length pads every
invoke out to the batch's longest input, so a position index counted from the left
names a different token depending on what else is in the batch, while one counted
from the right does not.

Batching only actually narrows when there are two or more non-empty invokes — a
lone invoke *is* the whole batch, so it sees every row untouched, and neither the
row scoping nor that row check applies to it.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Optional

import torch

from ..util import apply

if TYPE_CHECKING:
    from .envoy import Envoy

# A row range in the combined batch, or None for an empty/whole-batch invoke.
BatchGroup = Optional[list]


class SkipParts:
    """Skip replacements collected across the invokes of one batched forward.

    ``.skip()`` bypasses a module's body and substitutes a value for its output.
    In a batched forward there is no body output to splice into — the body didn't
    run — so the combined output is built from the invokes' replacements alone
    (see [`Batcher.gather_skip`][nnsight.intervention.batching.Batcher.gather_skip] / [`Batcher.assemble_skip`][nnsight.intervention.batching.Batcher.assemble_skip]). Each entry
    is a ``(group, replacement)`` pair.
    """

    def __init__(self) -> None:
        self.parts: list[tuple[list, Any]] = []


def concat(structures: list) -> Any:
    """Concatenate a list of like-shaped structures along dim 0, leaf by leaf.

    Every structure is one invoke's skip replacement, so they share a shape:
    tensors concatenate, containers recurse in parallel, and a non-tensor leaf
    (a ``None`` in a layer's output tuple, say) is taken from the first — they
    agree by construction.
    """
    first = structures[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(structures, dim=0)
    if isinstance(first, (list, tuple)):
        combined = [concat([s[i] for s in structures]) for i in range(len(first))]
        return type(first)(combined)
    if isinstance(first, dict):
        return {key: concat([s[key] for s in structures]) for key in first}
    return first


class Batcher:
    """Collects invoke inputs for one trace and builds the combined forward input.

    Each `add` records an invoke's input and returns its ``batch_group``.
    [`assemble`][nnsight.intervention.batching.Batcher.assemble] hands the collected invokes to the model's `_batch` to
    produce the actual ``(args, kwargs)`` for the run. `narrow`/`widen`
    scope a batched activation to a group's rows and splice an edit back; a model
    whose batch layout isn't a plain dim-0 stack overrides the per-tensor
    `_narrow_tensor`/`_widen_tensor` (e.g. diffusion's
    classifier-free-guidance doubling) and picks its subclass via
    `_batcher_class`.
    """

    def __init__(self, envoy: "Envoy", kwargs: Optional[dict] = None) -> None:
        self.envoy = envoy
        # The trace's forward kwargs (num_images_per_prompt, ...), for subclasses
        # whose row math depends on them; the base layout ignores them.
        self.kwargs = kwargs or {}
        # The row-contributing input sets, in order — each an (inputs, kwargs) that
        # `_batch` stacks into the combined forward.
        self.invokes: list[tuple] = []
        # Params-only adds (zero rows, e.g. max_new_tokens) fold their kwargs here;
        # assemble() lays them over the combined call.
        self.extra_kwargs: dict = {}
        self.total = 0
        # Tensors served whole because no row rule matched, by id: the tensor, torch's
        # version counter at the moment it was served, the location it came from and
        # its leading dim. `_report_unscoped` turns an edit to one into a warning, and
        # empties this — so what is held is one visit's unscoped values, no more.
        self.unscoped: dict[int, tuple] = {}

    def narrow(self, value: Any, group: BatchGroup) -> Any:
        """Slice every batched tensor in ``value`` down to ``group``'s rows.

        A tensor is scoped when its leading dim is [`total`][nnsight.intervention.batching.Batcher.total] (the
        combined batch size) or a whole multiple of it; anything else passes through
        whole and is remembered, so a write to it warns rather than acting on the
        whole batch silently. Returns the whole value when not actually batching or
        for a groupless (empty) invoke.
        """
        # An in-place edit doesn't come back through the batcher, so it can only be
        # noticed after the fact — here, next time anything is served.
        self._report_unscoped()
        if not self.batching or group is None:
            return value
        return apply(value, lambda tensor: self._narrow_tensor(tensor, group), torch.Tensor)

    def _narrow_tensor(self, tensor: torch.Tensor, group: list) -> torch.Tensor:
        """Slice one batched tensor down to ``group``'s rows.

        Base layout: dim 0 stacks the invokes' rows, either one row per row of batch
        (leading dim [`total`][nnsight.intervention.batching.Batcher.total]) or ``k``
        of them — a model that flattens tokens into the batch axis before a module
        (an MoE router's ``(B*T, D)``) keeps the invokes in the same order, ``k =
        shape[0] // total`` rows each, so the group scales by ``k``. A leading dim
        that is neither passes through whole and is remembered for
        `_report_unscoped`. Overridden for non-stacked layouts.
        """
        start, size = group
        rows = tensor.shape[0] if tensor.ndim else 0
        # A leading dim of a coincidental multiple (four experts, two invokes) is
        # sliced too; the shape is all there is to go on.
        k = rows // self.total if self.total and rows % self.total == 0 else 0
        if k:
            view = tensor.narrow(0, start * k, size * k)
            # Mark the slice so a `.backward()` gradient hook can tell it apart from a
            # user-made view: it isn't in the loss graph (the model runs on the full
            # batch), so its hook must redirect to the storage-owning base that is
            # (see intervention/backward.py). A bare marker — not the parent tensor —
            # keeps saved activations cheap to serialize.
            view._nnsight_batch = True
            return view
        self.unscoped[id(tensor)] = (
            tensor, tensor._version, self._location(group), rows
        )
        return tensor

    def _location(self, group: list) -> str:
        """Name the location whose value is being served to ``group``, for a warning.

        The worker that asked for it is parked on it for the length of the narrow
        (or widen), so its pending request names it; a value nobody asked for — a
        `Cache` observation — has no such worker.
        """
        for mediator in self.envoy.interleaver.mediators:
            if mediator.batch_group is group and mediator.pending is not None:
                return f"`{mediator.pending.provider}`"
        return "a value"

    def _report_unscoped(self) -> None:
        """Warn for any value served whole that has been edited in place since.

        A value no row rule matched goes to every invoke as it is, so an in-place
        edit to it acts on the whole batch — and, unlike a replacement, it never
        passes through `widen`, so torch's version counter is the only thing that
        records it happened.
        """
        if not self.unscoped:
            return
        for tensor, version, location, rows in self.unscoped.values():
            if tensor._version == version:
                continue
            warnings.warn(
                f"An in-place edit to {location} applies to every invoke: its "
                f"leading dimension ({rows}) is neither the batch size "
                f"({self.total}) nor a multiple of it, so nnsight served the whole "
                "batch rather than this invoke's rows. Trace this input on its own, "
                "or give the model a `_batcher_class` that knows the layout."
            )
        self.unscoped.clear()

    def widen(self, full: Any, group: BatchGroup, edited: Any) -> Any:
        """Splice ``edited`` (a block's rows) back into ``full`` (the whole batch).

        Walks ``full`` and ``edited`` in parallel; for each batched tensor in
        ``full`` writes the corresponding ``edited`` tensor into the group's rows via
        `_widen_tensor`. Returns ``edited`` unchanged when not batching or for
        a groupless invoke.
        """
        self._report_unscoped()
        if not self.batching or group is None:
            return edited

        def merge(full_value: Any, edited_value: Any) -> Any:
            if isinstance(full_value, torch.Tensor):
                return self._widen_tensor(full_value, group, edited_value)
            if isinstance(full_value, (list, tuple)):
                merged = [merge(f, e) for f, e in zip(full_value, edited_value)]
                # namedtuples take positional fields, not a single iterable.
                if isinstance(full_value, tuple) and hasattr(full_value, "_fields"):
                    return type(full_value)(*merged)
                return type(full_value)(merged)
            if isinstance(full_value, dict):
                # Copy first to preserve the exact type (OrderedDict, a HF
                # ModelOutput, ...) rather than rebuilding a plain dict, matching
                # how apply/narrow keep the container type.
                merged = full_value.copy()
                for key in full_value:
                    merged[key] = merge(full_value[key], edited_value[key])
                return merged
            return edited_value

        return merge(full, edited)

    def _widen_tensor(self, full: torch.Tensor, group: list, edited: torch.Tensor) -> torch.Tensor:
        """Write ``edited`` into ``full``'s ``group`` rows (base dim-0-stack layout).

        The row rule is `_narrow_tensor`'s: the leading dim is
        [`total`][nnsight.intervention.batching.Batcher.total] or a whole multiple of
        it. Anything else has no rows this invoke owns, so there is nowhere to splice
        the replacement — ``full`` stands, and the drop is warned about unless the
        block handed back what it was given. Overridden for non-stacked layouts.
        """
        start, size = group
        rows = full.shape[0] if full.ndim else 0
        k = rows // self.total if self.total and rows % self.total == 0 else 0
        if not k:
            if edited is not full:
                warnings.warn(
                    f"A replacement for {self._location(group)} was dropped: its "
                    f"leading dimension ({rows}) is neither the batch size "
                    f"({self.total}) nor a multiple of it, so nnsight can't tell "
                    "which rows belong to this invoke. Trace this input on its own, "
                    "or give the model a `_batcher_class` that knows the layout."
                )
            return full
        # cat (not in-place) keeps autograd correct for leaves/views and avoids
        # aliasing when `edited` is a narrowed view of `full`.
        pre = full.narrow(0, 0, start * k)
        post = full.narrow(0, (start + size) * k, rows - (start + size) * k)
        return torch.cat([pre, edited, post], dim=0)

    def gather_skip(self, running: Any, group: BatchGroup, replacement: Any) -> Any:
        """Collect one invoke's skip ``replacement`` for its ``group``'s rows.

        A lone invoke *is* the whole batch, so its replacement is the output
        outright. With two or more, there's no body output to splice into — the
        skip fires before the body runs — so accumulate the replacements and let
        [`assemble_skip`][nnsight.intervention.batching.Batcher.assemble_skip] build the combined output once every invoke's is in.
        """
        if not self.batching or group is None:
            return replacement
        if not isinstance(running, SkipParts):
            running = SkipParts()
        running.parts.append((group, replacement))
        return running

    def assemble_skip(self, running: Any) -> Any:
        """Concatenate collected skip replacements into the full-batch output.

        A no-op unless ``running`` is the [`SkipParts`][nnsight.intervention.batching.SkipParts] a batched skip built.
        The replacements must tile the whole batch: every invoke has to skip the
        module, since a shared forward can't run for only the rows that didn't.
        """
        if not isinstance(running, SkipParts):
            return running
        parts = sorted(running.parts, key=lambda part: part[0][0])
        covered = 0
        for (start, size), _ in parts:
            if start != covered:
                break
            covered += size
        if covered != self.total:
            raise ValueError(
                "A batched `.skip()` has to cover every row: skip the module in "
                "every invoke, or none — a shared forward can't run for only the "
                "rows an invoke left unskipped."
            )
        return concat([replacement for _, replacement in parts])

    def add(self, *inputs: Any, **kwargs: Any) -> BatchGroup:
        """Record one added input set; return its ``[start, size]`` row group.

        A set that contributes no rows — params only (e.g. ``max_new_tokens=``) or an
        empty ``invoke()`` — returns ``None`` (a groupless, whole-batch worker) and
        folds its kwargs into [`extra_kwargs`][nnsight.intervention.batching.Batcher.extra_kwargs], so [`assemble`][nnsight.intervention.batching.Batcher.assemble] lays them onto
        the combined call.
        """
        size = self.envoy._batch_size(*inputs, **kwargs)
        if not size:
            self.extra_kwargs.update(kwargs)
            return None
        group = [self.total, size]
        self.total += size
        self.invokes.append((inputs, kwargs))
        return group

    @property
    def batching(self) -> bool:
        """Whether narrowing applies — true once two or more invokes contribute rows."""
        return len(self.invokes) > 1

    def assemble(self, fn: Any) -> tuple:
        """Build the combined ``(args, kwargs)`` for ``fn`` from the collected input
        sets; params-only kwargs ([`extra_kwargs`][nnsight.intervention.batching.Batcher.extra_kwargs]) are laid on top and win."""
        args, kwargs = self.envoy._batch(self.invokes, fn)
        return args, {**kwargs, **self.extra_kwargs}
