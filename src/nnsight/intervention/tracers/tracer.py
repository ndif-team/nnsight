from typing import TYPE_CHECKING, Any, Optional, Callable, List

import torch

from ..interleaver import Interleaver, Mediator
from ...tracing.tracer import Tracer
from ...tracing.util import try_catch

if TYPE_CHECKING:
    from ..envoy import Envoy


class Invoker(Tracer):
    """
    Extends the Tracer class to invoke intervention functions.
    
    This class captures code blocks and compiles them into intervention functions
    that can be executed by the Interleaver.
    """
    
    def __init__(self, tracer: Tracer, *args, **kwargs):
        """
        Initialize an Invoker with a reference to the parent tracer.
        
        Args:
            tracer: The parent InterleavingTracer instance
            *args: Additional arguments to pass to the traced function
            **kwargs: Additional keyword arguments to pass to the traced function
        """
        self.tracer = tracer
        
        super().__init__(*args, **kwargs)
    
    def compile(self):
        """
        Compile the captured code block into an intervention function.
        
        The function is wrapped with try-catch logic to handle exceptions
        and signal completion to the mediator.
        
        Returns:
            A callable intervention function
        """
        self.info.source = [
            "def ifn(mediator, tracing_info):\n",
            *try_catch(self.info.source, 
                       exception_source=["mediator.exception(exception)\n"],
                       else_source=["mediator.end()\n"],)
        ]
        
        source = "".join(
            self.info.source
        )
                                
        local_namespace = {}
        
        # Execute the function definition in the local namespace
        exec(source, {**self.info.frame.f_globals, **self.info.frame.f_locals}, local_namespace)
        
        return local_namespace["ifn"]
            
    def execute(self, fn: Callable):
        """
        Execute the compiled intervention function.
        
        Creates a new Mediator for the intervention function and adds it to the
        parent tracer's mediators list.
        
        Args:
            fn: The compiled intervention function
        """
        # TODO: batch the interventions
        
        self.tracer.args = self.args
        self.tracer.kwargs = self.kwargs
            
        self.tracer.mediators.append(Mediator(fn, self.info))


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

    def __init__(self, fn: Callable, model: "Envoy", *args, **kwargs):
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
        
        self.mediators = []
        
        self._cache = None
                        
        super().__init__(*args, **kwargs)
        
    def compile(self) -> Callable:
        """
        Compile the captured code block into a callable function.
        
        Returns:
            A callable function that executes the captured code block
        """
        # Wrap the captured code in a function definition
        self.info.source = [
            "def fn(model, tracer, tracing_info):\n",
            *self.info.source
        ]
        
        source = "".join(self.info.source)
        
        local_namespace = {}

        # Execute the function definition in the local namespace
        exec(source, self.info.frame.f_globals, local_namespace)
        
        return local_namespace["fn"]

    def execute(self, fn: Callable):
        """
        Execute the compiled function with interventions.
        
        First executes the parent Tracer's execute method to set up the context,
        then creates an Interleaver to manage the interventions during model execution.
        
        Args:
            fn: The compiled function to execute
        """
        fn(self.model, self, self.info)
                
        with Interleaver(self.mediators) as interleaver:
            self.model._set_interleaver(interleaver)
            interleaver(self.fn, *self.args, **self.kwargs)
            
        self.model._clear()
        
        if self.info.frame.f_code.co_filename == '<string>':
            self.info.frame.f_globals.update(interleaver.state)
            
        else:    
            self.info.frame.f_locals.update(interleaver.state)
        
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
        return Invoker(self, *args, **kwargs)

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
           self.model._interleaver.set_user_cache(self._cache)
           
        return self._cache.cache