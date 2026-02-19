"""Batching support for multi-invoke traces.

This module provides the batching infrastructure that enables multiple invokes
to share a single forward pass. Each invoke's input is combined into one batch,
and during interleaving each invoke sees only its slice of the batch.

There are two types of invokes:

- **Input invokes**: ``tracer.invoke(input)`` — provides input data that contributes
  to the batch. Each input invoke gets a ``batch_group = [start, size]`` that
  specifies its slice of the batch dimension.

- **Empty invokes**: ``tracer.invoke()`` (no arguments) — operates on the **entire**
  batch from all previous input invokes. Empty invokes get ``batch_group = None``,
  so ``narrow()`` returns the full batch and ``swap()`` replaces the full batch.
  This is useful for running different intervention logic on the combined batch,
  or for breaking up interventions across multiple invokes to avoid
  execution-order conflicts within a single invoke.

To support multiple input invokes, model classes must subclass :class:`Batchable`
and implement :meth:`_prepare_input` and :meth:`_batch`. See
``nnsight.modeling.language.LanguageModel`` for a reference implementation.
Without these methods, you can still use one input invoke and any number of
empty invokes.
"""

from typing import Any, Tuple, List, Optional, Union

import torch
from ..util import apply, applyn


class Batchable:
    """Abstract mixin that defines how a model's inputs are prepared and batched.

    Subclasses should override :meth:`_prepare_input` and :meth:`_batch` to
    enable multiple input invokes in a single trace. The base ``Envoy`` class
    inherits from ``Batchable`` but does not override these methods, so only
    a single input invoke is supported by default.

    See ``LanguageModel`` for a full implementation that handles tokenization,
    padding, and attention mask construction.
    """

    ### Abstract methods ###

    def _prepare_input(
        self, *inputs, **kwargs
    ) -> tuple[tuple[Any], dict[str, Any], int]:
        """Normalize raw user input into a consistent format for batching.

        Called once per invoke with whatever arguments the user passed to
        ``tracer.invoke(*args, **kwargs)`` or ``model.trace(*args, **kwargs)``.

        Args:
            *inputs: Positional arguments from the invoke call.
            **kwargs: Keyword arguments from the invoke call.

        Returns:
            A 3-tuple of ``(args, kwargs, batch_size)`` where:
            - ``args``: Normalized positional arguments.
            - ``kwargs``: Normalized keyword arguments.
            - ``batch_size``: Number of samples in this invoke's input.
              Return 0 for empty invokes (no real input data).
        """

        if inputs or kwargs:
            return inputs, kwargs, 1

        return inputs, kwargs, 0

    def _batch(
        self, batched_input, *args, **kwargs
    ) -> tuple[tuple[Any], dict[str, Any]]:
        """Combine a new invoke's prepared input with the already-batched inputs.

        Called when a second (or subsequent) input invoke needs to be merged
        into the existing batch. The first invoke's prepared input is stored
        directly; this method is only called starting from the second invoke.

        Args:
            batched_input: A tuple of ``(batched_args, batched_kwargs)`` from
                all previous invokes combined.
            *args: The new invoke's prepared positional arguments
                (output of :meth:`_prepare_input`).
            **kwargs: The new invoke's prepared keyword arguments
                (output of :meth:`_prepare_input`).

        Returns:
            A 2-tuple of ``(combined_args, combined_kwargs)`` representing
            all invokes' inputs merged into one batch.

        Raises:
            NotImplementedError: If not overridden. The error message explains
                how to implement batching and the empty-invoke alternative.
        """

        raise NotImplementedError(
            "Batching is not implemented for this model. "
            "Multiple invokers with inputs require `_prepare_input()` and `_batch()` methods "
            "on your model class (see `LanguageModel` for a reference implementation). "
            "Without these methods, you can still use one invoke with input and additional "
            "empty invokes (no arguments) — empty invokes operate on the entire batch and "
            "are useful for breaking up interventions to avoid execution-order conflicts."
        )


class Batcher:
    """Manages input batching and per-invoke slicing for a single trace.

    One ``Batcher`` is created per trace. As invokes are defined, :meth:`batch`
    accumulates their inputs into a single batch. During interleaving,
    :meth:`narrow` extracts each invoke's slice and :meth:`swap` replaces it.

    Attributes:
        batched_args: Combined positional arguments from all input invokes.
        batched_kwargs: Combined keyword arguments from all input invokes.
        last_batch_group: The ``[start, size]`` for the most recent input invoke.
        needs_batching: True once there are 2+ input invokes (narrowing needed).
        current_value: The current activation value being narrowed/swapped.
    """

    def __init__(self, *args, **kwargs):

        self.batched_args = None
        self.batched_kwargs = None

        self.last_batch_group: Optional[List[int]] = None
        self.needs_batching = False

        self.current_value: Optional[Any] = None

    @property
    def total_batch_size(self):
        """Total number of samples across all input invokes."""

        return sum(self.last_batch_group)

    def batch(
        self, batchable: Batchable, *args, **kwargs
    ) -> Tuple[Tuple[Any, Any], Optional[List[int]]]:
        """Register an invoke's input and return its batch group.

        For input invokes (args/kwargs provided), this calls
        ``batchable._prepare_input()`` and optionally ``batchable._batch()``
        to merge with previous inputs. For empty invokes (no args), returns
        ``batch_group=None`` which tells :meth:`narrow` to return the full batch.

        Args:
            batchable: The model instance (implements :class:`Batchable`).
            *args: Positional arguments from the invoke.
            **kwargs: Keyword arguments from the invoke.

        Returns:
            A 2-tuple of ``((args, kwargs), batch_group)`` where
            ``batch_group`` is ``[start_idx, batch_size]`` for input invokes
            or ``None`` for empty invokes.
        """

        if args or kwargs:

            args, kwargs, batch_size = batchable._prepare_input(*args, **kwargs)

            batch_group = [0, batch_size] if batch_size else None

            if self.batched_args is None:
                self.batched_args = args
                self.batched_kwargs = kwargs

                self.last_batch_group = batch_group

                return (args, kwargs), batch_group

            self.batched_args, self.batched_kwargs = batchable._batch(
                (self.batched_args, self.batched_kwargs), *args, **kwargs
            )

            if batch_group is None:
                return (args, kwargs), None

            if self.last_batch_group is None:
                self.last_batch_group = batch_group
            else:
                self.last_batch_group = [sum(self.last_batch_group), batch_size]
                self.needs_batching = True

            return (args, kwargs), self.last_batch_group

        return (args, kwargs), None

    def narrow(self, batch_group: Optional[List[int]]):
        """Extract an invoke's slice from the current activation value.

        For input invokes, narrows each tensor along dimension 0 using the
        invoke's ``batch_group = [start, size]``. For empty invokes
        (``batch_group=None``), returns the entire batch unmodified.

        Args:
            batch_group: ``[start_idx, batch_size]`` for input invokes,
                or ``None`` for empty invokes (returns full batch).

        Returns:
            The narrowed (or full) activation data.
        """

        data = self.current_value

        if not self.needs_batching or batch_group is None:
            return data

        batch_start, batch_size = batch_group

        if batch_start == -1:
            return data

        def _narrow(acts: torch.Tensor):

            if acts.shape[0] == self.total_batch_size:
                return acts.narrow(0, batch_start, batch_size)

            return acts

        return apply(
            data,
            _narrow,
            torch.Tensor,
        )

    def swap(self, batch_group: Optional[List[int]], swap_value: Any):
        """Replace an invoke's slice in the current activation value.

        For input invokes, splices ``swap_value`` into the correct batch
        slice. For empty invokes (``batch_group=None``), replaces the
        entire ``current_value``.

        Handles two cases for tensor replacement:
        - If the tensor is a leaf with ``requires_grad`` or has a base tensor
          (view), uses ``torch.cat`` to avoid in-place modification issues.
        - Otherwise, uses direct index assignment for efficiency.

        Args:
            batch_group: ``[start_idx, batch_size]`` for input invokes,
                or ``None`` for empty invokes (replaces full batch).
            swap_value: The new value to splice in.
        """

        if not self.needs_batching or batch_group is None:
            self.current_value = swap_value
            return

        batch_start, batch_size = batch_group

        if batch_start == -1:
            self.current_value = swap_value
            return

        def _swap(current_value: torch.Tensor, swap_value: torch.Tensor):

            if current_value.shape[0] == self.total_batch_size:

                needs_concat = (
                    current_value.requires_grad and current_value.is_leaf
                ) or current_value._base is not None

                if needs_concat:
                    pre = current_value.narrow(0, 0, batch_start)
                    post = (
                        current_value.narrow(
                            0,
                            batch_start + batch_size,
                            current_value.shape[0] - batch_start - batch_size,
                        )
                        if self.total_batch_size == current_value.shape[0]
                        else current_value
                    )

                    return torch.cat([pre, swap_value, post], dim=0)

                else:
                    current_value[batch_start : batch_start + batch_size] = swap_value

            return current_value

        self.current_value = applyn(
            [self.current_value, swap_value], _swap, torch.Tensor
        )
