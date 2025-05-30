

from typing import Any, Tuple

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
        
        self.batch_groups = []
        
        self.first_input = True
        self.needs_batching = False
        
        self.current_value = None
        
        self._total_batch_size = None
        
        
    @property
    def total_batch_size(self):
        
        if self._total_batch_size is None:
            
            self._total_batch_size = sum(self.batch_groups[-1])
            
        return self._total_batch_size
            

        
    def batch(self, batchable: Batchable, *args, **kwargs):
        
        if args or kwargs:
            
            args, kwargs = batchable._prepare_input(*args, **kwargs)
            
            if self.first_input:
                self.batched_args = args
                self.batched_kwargs.update(kwargs)
                
                self.batch_groups.append((-1, -1))
                
                self.first_input = False
                
            else:
                
                if self.batch_groups[-1][1] == -1:
                    (self.batched_args, self.batched_kwargs), batch_size = batchable._batch(None, *self.batched_args, **self.batched_kwargs)
                    
                    self.batch_groups[-1] = (0, batch_size)
                                        
                (self.batched_args, self.batched_kwargs), batch_size = batchable._batch((self.batched_args, self.batched_kwargs), *args, **kwargs)
                    
                self.batch_groups.append((sum(self.batch_groups[-1]), batch_size))
                
                self.needs_batching = True
                  
        else:
            self.batch_groups.append((-1, -1))
            
            
    def narrow(self, batch_group: int, data:Any):

        if not self.needs_batching:
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
        
        
    def swap(self, batch_group, swap_value: Any):
        
        if not self.needs_batching:
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
            