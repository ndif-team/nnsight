import ast
from typing import TYPE_CHECKING, Any, Optional, Callable, List

import torch

from ..interleaver import Interleaver
from .base import ExitTracingException, Tracer
from .invoker import Invoker
from ..batching import Batcher

from .iterator import IteratorProxy
from ..backends.base import Backend
if TYPE_CHECKING:
    from ..interleaver import Mediator, BATCH_GROUP
    from ..envoy import Envoy
else:
    Envoy = Any


class Cache:
    """
    A cache for storing and transforming tensor values during tracing.
    
    This class provides functionality to store tensor values with optional
    transformations such as detaching from computation graph, moving to a
    specific device, or converting to a specific dtype.
    """
    
    def __init__(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, detach: Optional[bool] = False):
        """
        Initialize a Cache with optional transformation parameters.
        
        Args:
            device: Optional device to move tensors to
            dtype: Optional dtype to convert tensors to
            detach: Whether to detach tensors from computation graph
        """
        self.device = device
        self.dtype = dtype
        self.detach = detach
        
        self.cache = {}
        
    def add(self, provider: str, value: Any):
        """
        Add a value to the cache with optional transformations.
        
        Args:
            provider: The key to store the value under
            value: The tensor value to store
        """
        # TODO: util .apply
        
        if self.detach:
            value = value.detach()
        
        if self.device is not None:
            value = value.to(self.device)
                
        if self.dtype is not None:
            value = value.to(self.dtype)
            
        self.cache[provider] = value
                


class InterleavingTracer(Tracer):
    """
    Tracer that manages the interleaving of model execution and interventions.
    
    This class coordinates the execution of the model's forward pass and
    user-defined intervention functions through the Interleaver.
    """

    def __init__(self, fn: Callable, model: Envoy, *args, backend: Backend = None, **kwargs):
        """
        Initialize an InterleavingTracer with a function and model.
        
        Args:
            fn: The function to execute (typically the model's forward pass)
            model: The model envoy to intervene on
            *args: Additional arguments to pass to the function
            **kwargs: Additional keyword arguments to pass to the function
        """
        
        self.fn = fn
        self.model = model
        
        self.mediators:List[Mediator] = []
       
        self.batcher = Batcher(**kwargs)
        
        self._cache = None
                        
        super().__init__(*args, backend=backend)
        
        if not hasattr(self, 'model_var_name'):
            self.model_var_name = self.info.node.items[0].context_expr.func.value.id
        if not hasattr(self, 'tracer_var_name'):
            self.tracer_var_name = self.info.node.items[0].optional_vars.id if self.info.node.items[0].optional_vars is not None else "__nnsight_tracer__"
        
   
    def compile(self) -> Callable:
        """
        Compile the captured code block into a callable function.
        
        Returns:
            A callable function that executes the captured code block
        """

        # If Envoy has a default mediators ( created via Envoy.edit() ), add them
        if self.model._default_mediators:
                        
            for mediators in self.model._default_mediators:
                
                self.mediators.append(mediators)
                self.batcher.batch_groups.append((-1,-1))
                
            
        #If positional arguments were passed directly to a tracer, assume one invoker
        if self.args:
            invoker = self.invoke(*self.args, **self.kwargs)
            invoker.info = self.info.copy()
                        
            invoker.__exit__(ExitTracingException, None, None)
            
            self.info.source = ['    pass\n']
                    
        self.info.source = [
            f"def __nnsight_tracer_{id(self)}__(__nnsight_tracing_info__, {self.model_var_name},{self.tracer_var_name}):\n",
            *self.info.source,
            f"    {self.tracer_var_name}.push()\n"
        ]
        
        self.args = tuple()
                        
    

    def execute(self, fn: Callable):
        """
        Execute the compiled function with interventions.
        
        First executes the parent Tracer's execute method to set up the context,
        then creates an Interleaver to manage the interventions during model execution.
        
        Args:
            fn: The compiled function to execute
        """
        
        fn(self.info, self.model, self)
        
        args = self.batcher.batched_args
        kwargs = self.batcher.batched_kwargs
        
        self.batcher.batched_args = tuple()
        self.batcher.batched_kwargs = {}

        interleaver = Interleaver(self.mediators, self, batcher=self.batcher)
        self.model.interleave(interleaver, self.fn, *args, **kwargs)
        
        self.push(interleaver.state)

    ### Public API ####
        
    def invoke(self, *args, **kwargs):
        """
        Create an Invoker to capture and execute an intervention function.
        
        Args:
            *args: Additional arguments to pass to the intervention function
            **kwargs: Additional keyword arguments to pass to the intervention function
            
        Returns:
            An Invoker instance
        """
        #TODO make sure not already executing
        return Invoker(self, *args, **kwargs)
    
    def stop(self):
        """
        Raise an EarlyStopException to stop the execution of the model.
        """
        self.model._interleaver.current.stop()
    

    @property
    def iter(self):
        return IteratorProxy(self.model._interleaver)
    
    def all(self):
        return self.iter[:]

    def cache(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, detach: Optional[bool] = False):
        """
        Get or create a cache for storing intermediate values during tracing.
        
        Args:
            device: Optional device to move tensors to
            dtype: Optional dtype to convert tensors to
            detach: Whether to detach tensors from computation graph
            
        Returns:
            A dictionary containing the cached values
        """
        if self._cache is None:
           self._cache = Cache(device, dtype, detach)
           self.model._interleaver.current.set_user_cache(self._cache)
           
        return self._cache.cache
    
    ### Serialization ###
    
    def __getstate__(self):
        """Get the state of the tracer for serialization."""
        state = super().__getstate__()
        state['fn'] = self.fn.__name__
        state['model_var_name'] = self.model_var_name
        state['tracer_var_name'] = self.tracer_var_name
        state['batcher'] = self.batcher
        state['mediators'] = self.mediators
        
        return state
    
    def __setstate__(self, state):
        """Set the state of the tracer for deserialization."""
        super().__setstate__(state)
        
        self.fn = state['fn']
        self.model_var_name = state['model_var_name']
        self.tracer_var_name = state['tracer_var_name']
        self.mediators = state['mediators']
        self.batcher =  state['batcher']

        self._cache = None
        
    def __setmodel__(self, model:Envoy):
        
        self.model = model
        self.fn = getattr(self.model, self.fn)
        
    def __setframe__(self, frame):
        
        super().__setframe__(frame)
        
        self.info.start_line = 0
        
        for mediator in self.mediators:
            mediator.info.frame = frame
