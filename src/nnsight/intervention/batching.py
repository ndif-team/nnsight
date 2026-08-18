"""Batch several invoke inputs into one forward and scope interventions to rows.

``with model.trace() as tracer:`` may contain several ``with tracer.invoke(x):``
blocks. Their inputs are combined into a single batched forward, and each block's
interventions see only *its* rows of every activation.

A [`Batcher`][nnsight.intervention.batching.Batcher] (one per trace) collects each invoke's input and assigns it a
``batch_group`` — a ``[start, size]`` row range in the combined batch. At run time
[`Batcher.narrow`][nnsight.intervention.batching.Batcher.narrow] slices a full batched activation down to a block's rows when
it reads, and [`Batcher.widen`][nnsight.intervention.batching.Batcher.widen] splices an edit back into the full tensor. The row math
runs along the axis `Batcher._batch_dim` reports (dim 0 in the base layout);
the model's `_batch` equalizes everything else (e.g. sequence length) when it
builds the combined input.

Batching only actually narrows when there are two or more non-empty invokes — a
lone invoke *is* the whole batch, so it sees every row untouched.
"""

from __future__ import annotations

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

    def narrow(self, value: Any, group: BatchGroup) -> Any:
        """Slice every batched tensor in ``value`` down to ``group``'s rows.

        A tensor is batched only when its leading dim equals [`total`][nnsight.intervention.batching.Batcher.total] (the
        combined batch size), so non-batched tensors pass through untouched. Returns
        the whole value when not actually batching or for a groupless (empty) invoke.
        """
        if not self.batching or group is None:
            return value
        return apply(value, lambda tensor: self._narrow_tensor(tensor, group), torch.Tensor)

    def _batch_dim(self, tensor: torch.Tensor) -> Optional[int]:
        """Axis of ``tensor`` the batch groups index, or ``None`` when it isn't batched.

        Base layout: a tensor whose leading dim is [`total`][nnsight.intervention.batching.Batcher.total] is a dim-0 stack of
        every invoke's rows; anything else isn't batched. Overridden when the
        batched axis can sit elsewhere (see
        [`VLLMBatcher`][nnsight.modeling.vllm.batching.VLLMBatcher], whose
        Transformers-backend activations are ``[1, total_tokens, hidden]``).
        """
        if tensor.shape[0] == self.total:
            return 0
        return None

    def _narrow_tensor(self, tensor: torch.Tensor, group: list) -> torch.Tensor:
        """Slice one batched tensor down to ``group``'s rows.

        Narrows to ``[start, start + size)`` along the axis `_batch_dim`
        reports; a tensor it calls unbatched passes through. Overridden for
        non-stacked layouts.
        """
        start, size = group
        dim = self._batch_dim(tensor)
        if dim is not None:
            view = tensor.narrow(dim, start, size)
            # Mark the slice so a `.backward()` gradient hook can tell it apart from a
            # user-made view: it isn't in the loss graph (the model runs on the full
            # batch), so its hook must redirect to the storage-owning base that is
            # (see intervention/backward.py). A bare marker — not the parent tensor —
            # keeps saved activations cheap to serialize.
            view._nnsight_batch = True
            return view
        return tensor

    def widen(self, full: Any, group: BatchGroup, edited: Any) -> Any:
        """Splice ``edited`` (a block's rows) back into ``full`` (the whole batch).

        Walks ``full`` and ``edited`` in parallel; for each batched tensor in
        ``full`` writes the corresponding ``edited`` tensor into the group's rows via
        `_widen_tensor`. Returns ``edited`` unchanged when not batching or for
        a groupless invoke.
        """
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
        """Write ``edited`` into ``full``'s ``group`` rows.

        Splices along the axis `_batch_dim` reports; a tensor it calls
        unbatched passes through. Overridden for non-stacked layouts.
        """
        dim = self._batch_dim(full)
        if dim is None:
            return full
        start, size = group
        # cat (not in-place) keeps autograd correct for leaves/views and avoids
        # aliasing when `edited` is a narrowed view of `full`.
        pre = full.narrow(dim, 0, start)
        post = full.narrow(dim, start + size, self.total - start - size)
        return torch.cat([pre, edited, post], dim=dim)

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
