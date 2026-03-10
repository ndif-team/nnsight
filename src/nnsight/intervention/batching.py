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

from functools import partial
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

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

    @classmethod
    def _batcher_class(cls) -> ClassVar[Type["Batcher"]]:
        return Batcher

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

    def narrow(self, batch_group: Optional[List[int]]) -> Any:
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

        if batch_group[0] == -1:
            return data

        return apply(
            data,
            partial(self._narrow, batch_group),
            torch.Tensor,
        )

    def swap(self, batch_group: Optional[List[int]], swap_value: Any) -> None:
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

        if batch_group[0] == -1:
            self.current_value = swap_value
            return

        self.current_value = applyn(
            [self.current_value, swap_value], 
            partial(self._swap, batch_group), 
            torch.Tensor
        )

    def _narrow(self, batch_group: Optional[List[int]], acts: torch.Tensor) -> torch.Tensor:
        '''
        Narrow a tensor to a specific batch group.

        Args:
            batch_group (List[int]): The [start_idx, batch_size] of the batch group to extract.
            acts (torch.Tensor): The input tensor to narrow. The first dimension should
                correspond to the batch dimension.

        Returns:
            torch.Tensor: The narrowed tensor containing only the specified batch group.
        '''
        batch_start, batch_size = batch_group

        if acts.shape[0] == self.total_batch_size:
            return acts.narrow(0, batch_start, batch_size)

        return acts

    def _swap(self, batch_group: Optional[List[int]], current_value: torch.Tensor, swap_value: torch.Tensor) -> torch.Tensor:
        '''
        Swap a tensor with a new value in a specific batch group.

        Args:
            batch_group (List[int]): The [start_idx, batch_size] of the batch group to modify.
            current_value (torch.Tensor): The tensor containing current values to be modified.
            swap_value (torch.Tensor): The new values to insert at the specified batch group.

        Returns:
            torch.Tensor: The modified tensor with swapped values. May be a new tensor
                (concatenation) or the modified input tensor (in-place assignment).
        '''
        batch_start, batch_size = batch_group

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


class DiffusionBatcher(Batcher):
    """
    A specialized batcher for diffusion models that handles multiple images per prompt and guided diffusion.

    This class extends the base Batcher to support diffusion model-specific batching scenarios,
    including multiple images per prompt and guided diffusion with conditional/unconditional
    guidance.

    The DiffusionBatcher handles three main tensor batch size scenarios:
    1. Regular batch size (total_batch_size)
    2. Image batch size (total_batch_size * num_images_per_prompt)  
    3. Guided diffusion batch size (total_batch_size * num_images_per_prompt * 2)

    Attributes:
        num_images_per_prompt (int): Number of images to generate per prompt. Defaults to 1.
        image_batch_groups (List[Tuple[int, int]]): Batch groups scaled for multiple images
            per prompt, where each tuple contains (batch_start, batch_size).
    """

    def __init__(self, *args, **kwargs):

        self.num_images_per_prompt: int = kwargs.get("num_images_per_prompt", 1)
        self._image_batch_groups: Dict[int, Tuple[int, int]] = dict()
        self.last_image_batch_group: Optional[Tuple[int, int]] = None

        super().__init__(*args, **kwargs)

    @property
    def image_batch_groups(self) -> Dict[int, Tuple[int, int]]:
        return self._image_batch_groups

    def batch(self, batchable: Batchable, *args, **kwargs) -> Tuple[Tuple[Any, Any], Optional[List[int]]]:
        """
        Batch inputs for diffusion models, accounting for multiple images per prompt.

        This method extends the base batcher functionality to handle diffusion models that can
        generate multiple images per prompt. It creates image-specific batch groups by scaling
        the regular batch groups by the number of images per prompt.

        Args:
            batchable (Batchable): The batchable object that implements the batching interface.
            *args: Variable length argument list to be batched.
            **kwargs: Arbitrary keyword arguments to be batched.

        Returns:
            Tuple[Tuple[Any, Any], Union[int, None]]: A tuple containing:
                - A tuple of (batched_args, batched_kwargs)
                - The batch group index (int) or None if no batching was needed
        """     
        input_args, batch_group = super().batch(batchable, *args, **kwargs)

        if batch_group is not None:
            batch_start, batch_size = batch_group

            if not self.needs_batching:
                image_batch_group = (0, batch_size*self.num_images_per_prompt)
            else:
                image_batch_group = (sum(self.last_image_batch_group), batch_size*self.num_images_per_prompt)

            self.image_batch_groups[batch_start] = image_batch_group
            self.last_image_batch_group = image_batch_group

        return input_args, batch_group

    def _narrow(self, batch_group: Optional[List[int]], acts: torch.Tensor) -> torch.Tensor:
        """
        Extract a specific batch group from a tensor, handling diffusion model batch scenarios.

        Args:
            batch_group (int): The index of the batch group to extract.
            acts (torch.Tensor): The input tensor to narrow. The first dimension should
                correspond to the batch dimension.

        Returns:
            torch.Tensor: The narrowed tensor containing only the specified batch group.
                For guided diffusion (2x batch size), returns concatenated unconditional
                and conditional parts. For other cases, returns the appropriate slice.
        """
        if (acts.shape[0] == self.total_batch_size):
            batch_start, batch_size = batch_group
            
            return acts.narrow(0, batch_start, batch_size)

        elif acts.shape[0] == self.total_batch_size * self.num_images_per_prompt:
            batch_start, batch_size = self.image_batch_groups[batch_group[0]]

            return acts.narrow(0, batch_start, batch_size)

        elif acts.shape[0] == self.total_batch_size * self.num_images_per_prompt * 2: # with guidance
            batch_start, batch_size = self.image_batch_groups[batch_group[0]]

            uncond_batch = acts.narrow(0, batch_start, batch_size)

            cond_batch = acts.narrow(0, batch_start + self.total_batch_size * self.num_images_per_prompt, batch_size)

            return torch.cat([uncond_batch, cond_batch], dim=0)
        
        return acts

    def _swap(self, batch_group: int, current_value: torch.Tensor, swap_value: torch.Tensor) -> torch.Tensor:
        """
        Replace values in a specific batch group with new values, handling diffusion model scenarios.

        This method swaps values for a specific batch group in a tensor, supporting different
        diffusion model batch configurations. It automatically determines whether to use
        in-place assignment or concatenation based on tensor properties (gradient requirements).

        Args:
            batch_group (int): The index of the batch group to modify.
            current_value (torch.Tensor): The tensor containing current values to be modified.
            swap_value (torch.Tensor): The new values to insert at the specified batch group.

        Returns:
            torch.Tensor: The modified tensor with swapped values. May be a new tensor
                (concatenation) or the modified input tensor (in-place assignment).
        """
        needs_concat = (current_value.requires_grad and current_value.is_leaf) or current_value._base is not None
            
        if current_value.shape[0] == self.total_batch_size or current_value.shape[0] == self.total_batch_size * self.num_images_per_prompt:
            batch_start, batch_size = batch_group if current_value.shape[0] == self.total_batch_size else self.image_batch_groups[batch_group[0]]
        
            if needs_concat:
                pre = current_value.narrow(0, 0, batch_start)

                post = current_value.narrow(0, batch_start+batch_size, current_value.shape[0] - batch_start - batch_size) if self.total_batch_size == current_value.shape[0] else current_value
                
                return torch.cat([pre, swap_value, post], dim=0)
            
            else:
                current_value[batch_start:batch_start+batch_size] = swap_value

        elif current_value.shape[0] == self.total_batch_size * self.num_images_per_prompt * 2: # with guidance
            batch_start, batch_size = self.image_batch_groups[batch_group[0]]

            uncond_swap, cond_swap = swap_value.chunk(2, dim=0)

            if needs_concat:
                pre_uncond = current_value.narrow(0, 0, batch_start)
                pre_cond = current_value.narrow(0, batch_start+batch_size, self.total_batch_size*self.num_images_per_prompt - batch_start - batch_size)
                post = current_value.narrow(0, batch_start+self.total_batch_size*self.num_images_per_prompt+batch_size, -1)
                
                return torch.cat([pre_uncond, uncond_swap, pre_cond, cond_swap, post], dim=0)
            
            else:
                current_value[batch_start:batch_start+batch_size] = uncond_swap
                current_value[batch_start + self.total_batch_size*self.num_images_per_prompt:batch_start+self.total_batch_size*self.num_images_per_prompt+batch_size] = cond_swap
                
        return current_value
