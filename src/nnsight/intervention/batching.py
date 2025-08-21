

from typing import Any, Tuple, List, Optional, Union

import torch
from ..util import apply, applyn

class Batchable:
    
    ### Abstract methods ###
    
    def _prepare_input(self, *inputs, **kwargs):
        
        return inputs, kwargs
    
    def _batch(self, batched_input, *args, **kwargs):
        
        raise NotImplementedError("Batching not implemented for this model and is required for multiple invokers")
    
    
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
            
    def narrow(self, batch_group: Union[int, None], data:Any):

        if not self.needs_batching or batch_group == None:
            return data
    
        batch_start, batch_size = self.batch_groups[batch_group]
        
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
        
        
    def swap(self, batch_group: Union[int, None], swap_value: Any):
        
        if not self.needs_batching or batch_group == None:
            self.current_value = swap_value
            return
                
        batch_start, batch_size = self.batch_groups[batch_group]

        if batch_start == -1:
            self.current_value = swap_value
            return
        
        def _swap(current_value: torch.Tensor, swap_value: torch.Tensor):
            
            if current_value.shape[0] == self.total_batch_size:
            
                needs_concat = (current_value.requires_grad and current_value.is_leaf) or current_value._base is not None
              
                if needs_concat:
                    pre = current_value.narrow(0, 0, batch_start)
                    post = current_value.narrow(0, batch_start+batch_size, current_value.shape[0] - batch_start - batch_size) if self.total_batch_size == current_value.shape[0] else current_value
                    
                    return torch.cat([pre, swap_value, post], dim=0)
                
                else:
                    current_value[batch_start:batch_start+batch_size] = swap_value
                    
            return current_value
                
        self.current_value = applyn([self.current_value, swap_value], _swap, torch.Tensor)


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
            