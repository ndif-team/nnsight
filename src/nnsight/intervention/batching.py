from functools import partial
from typing import Any, ClassVar, List, Optional, Tuple, Union

import torch

from ..util import apply, applyn


class Batchable:
    
    ### Abstract methods ###
    
    def _prepare_input(self, *inputs, **kwargs):
        
        return inputs, kwargs
    
    def _batch(self, batched_input, *args, **kwargs):
        
        raise NotImplementedError("Batching not implemented for this model and is required for multiple invokers")

    @classmethod
    def _batcher_class(self) -> ClassVar:

        return Batcher
    
    
class Batcher:
     
    def __init__(self, *args, **kwargs):
        
        self.batched_args = args
        self.batched_kwargs = kwargs
        
        self.cached_batch_groups: Optional[List[Tuple[int, int]]] = None
        
        self.first_input = True
        self.needs_batching = False
        
        self.current_value = None
        
        self._batch_groups: List[Tuple[int, int]] = []
        self._total_batch_size = None
        
    @property
    def total_batch_size(self):
        
        if self._total_batch_size is None:
            
            self._total_batch_size = sum(self.batch_groups[-1])
            
        return self._total_batch_size

    @property
    def batch_groups(self) -> List[Tuple[int, int]]:
        return self._batch_groups

    @batch_groups.setter
    def batch_groups(self, value: List[Tuple[int, int]]):
        self._batch_groups = value
        self._total_batch_size = None

    def batch(self, batchable: Batchable, *args, **kwargs) -> Tuple[Tuple[Any, Any], Union[int, None]]:
        
        if args or kwargs:
            
            args, kwargs = batchable._prepare_input(*args, **kwargs)
            
            if self.first_input:
                self.batched_args = args
                self.batched_kwargs.update(kwargs)
                
                self.batch_groups.append((-1, -1))
                
                self.first_input = False
                
            else:
                
                if self.batch_groups[0] == (-1, -1):
                    (self.batched_args, self.batched_kwargs), batch_size = batchable._batch(None, *self.batched_args, **self.batched_kwargs)
                    
                    self.batch_groups[0] = (0, batch_size)
                                        
                (self.batched_args, self.batched_kwargs), batch_size = batchable._batch((self.batched_args, self.batched_kwargs), *args, **kwargs)
                                    
                self.batch_groups.append((sum(self.batch_groups[-1]), batch_size))
                
                self.needs_batching = True

            return (args, kwargs), len(self.batch_groups) - 1

        return (args, kwargs), None

    def cache_batch_groups(self, new_batch_groups: List[Tuple[int, int]]):
        """
        Cache the batch groups for the current batch.

        Args:
            new_batch_groups (List[Tuple[int, int]]): The new batch groups to use as current batch groups.
        """

        if not self.needs_batching or self.cached_batch_groups != None:
            return

        
        self.cached_batch_groups = self.batch_groups
        self.batch_groups = new_batch_groups
    
    def restore_batch_groups(self):
        """
        Restore the batch groups to the previous batch groups.
        """

        if not self.needs_batching or self.cached_batch_groups is None:
            return
        
        self.batch_groups = self.cached_batch_groups
        self.cached_batch_groups = None
        self._total_batch_size = None

    def narrow(self, batch_group: Union[int, None], data: Any):

        if not self.needs_batching or batch_group == None:
            return data

        return apply(
            data,
            partial(self._narrow, batch_group),
            torch.Tensor,
        )
        
    def swap(self, batch_group: Union[int, None], swap_value: Any):
        
        if not self.needs_batching or batch_group == None:
            self.current_value = swap_value
            return
        
        self.current_value = applyn(
            [self.current_value, swap_value], 
            partial(self._swap, batch_group), 
            torch.Tensor
        )
    
    def _narrow(self, batch_group: int, acts: torch.Tensor):
        
        batch_start, batch_size = self.batch_groups[batch_group]

        if acts.shape[0] == self.total_batch_size:
            return acts.narrow(0, batch_start, batch_size)

        return acts

    def _swap(self, batch_group: int, current_value: torch.Tensor, swap_value: torch.Tensor):
        
        batch_start, batch_size = self.batch_groups[batch_group]
            
        if current_value.shape[0] == self.total_batch_size:
        
            needs_concat = (current_value.requires_grad and current_value.is_leaf) or current_value._base is not None
            
            if needs_concat:
                pre = current_value.narrow(0, 0, batch_start)
                post = current_value.narrow(0, batch_start+batch_size, current_value.shape[0] - batch_start - batch_size) if self.total_batch_size == current_value.shape[0] else current_value
                
                return torch.cat([pre, swap_value, post], dim=0)
            
            else:
                current_value[batch_start:batch_start+batch_size] = swap_value
                
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
        self._image_batch_groups: List[Tuple[int, int]] = list()
        super().__init__(*args, **kwargs)

    @property
    def image_batch_groups(self) -> List[Tuple[int, int]]:
        return self._image_batch_groups

    def batch(self, batchable: Batchable, *args, **kwargs) -> Tuple[Tuple[Any, Any], Union[int, None]]:
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
                
        output = super().batch(batchable, *args, **kwargs)

        if self.needs_batching:

            for batch_start, batch_size in self.batch_groups:
                if len(self.image_batch_groups) == 0:
                    self.image_batch_groups.append((0, batch_size*self.num_images_per_prompt))
                else:
                    self.image_batch_groups.append((sum(self.image_batch_groups[-1]), batch_size*self.num_images_per_prompt))

        return output

    def _narrow(self, batch_group: int, acts: torch.Tensor) -> torch.Tensor:
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

            batch_start, batch_size = self.batch_groups[batch_group]
            
            return acts.narrow(0, batch_start, batch_size)

        elif acts.shape[0] == self.total_batch_size * self.num_images_per_prompt:

            batch_start, batch_size = self.image_batch_groups[batch_group]

            return acts.narrow(0, batch_start, batch_size)

        elif acts.shape[0] == self.total_batch_size * self.num_images_per_prompt * 2: # with guidance

            batch_start, batch_size = self.image_batch_groups[batch_group]

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

            batch_start, batch_size = self.batch_groups[batch_group] if current_value.shape[0] == self.total_batch_size else self.image_batch_groups[batch_group]
        
            if needs_concat:
                pre = current_value.narrow(0, 0, batch_start)
                post = current_value.narrow(0, batch_start+batch_size, current_value.shape[0] - batch_start - batch_size) if self.total_batch_size == current_value.shape[0] else current_value
                
                return torch.cat([pre, swap_value, post], dim=0)
            
            else:
                current_value[batch_start:batch_start+batch_size] = swap_value

        elif current_value.shape[0] == self.total_batch_size * self.num_images_per_prompt * 2: # with guidance

            batch_start, batch_size = self.image_batch_groups[batch_group]

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
