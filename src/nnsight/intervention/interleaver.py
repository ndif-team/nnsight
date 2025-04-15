import inspect
import torch
from typing import Any, Callable, Dict, Tuple, Optional, TYPE_CHECKING
import asyncio
from enum import Enum
from functools import wraps
if TYPE_CHECKING:
    from .envoy import Envoy

class Events(Enum):
    """Enum for different types of events in the interleaving process."""
    VALUE = 'value'      # Request for a value
    CONTINUE = 'continue'  # Signal to continue execution
    EXCEPTION = 'exception'  # Signal that an exception occurred


class Interleaver:
    """
    Manages the interleaving of model execution and interventions.
    
    This class coordinates the flow between the model's forward pass and
    user-defined intervention functions, allowing for inspection and
    modification of intermediate values.
    """
    
    def __init__(self, interventions: Callable) -> None:
        """
        Initialize an Interleaver with intervention functions.
        
        Args:
            interventions: A callable that defines the intervention logic
        """
        self.interventions = interventions
        
        self.original_call = None
        self.value_future = None
        self.swap_future = None
        self.event_future = None
        self.async_loop = None
        
        self.missed_providers = set()
        
    def wrap(self, fn: Callable, include_module: bool = False, name: Optional[str] = None):
        """
        Wrap a function to enable interleaving at its inputs and outputs.
        
        Args:
            fn: The function to wrap
            include_module: Whether to include the module as the first argument
            name: Optional name for the function
            
        Returns:
            A wrapped version of the function
        """
        @wraps(fn)
        def inner(module, *args, **kwargs):
            
            requester = module.__path__ if name is None else f"{module.__path__}.{name}"
            
            # Call our hook before the original __call__
            new_args = self.hook(f"{requester}.input", (args, kwargs))
            
            if new_args is not inspect._empty:
                args, kwargs = new_args

            # Call the original __call__ method
            if include_module:
                value = fn(module, *args, **kwargs)
            else:
                value = fn(*args, **kwargs)
 
            new_value = self.hook(f"{requester}.output", value)
            
            if new_value is not inspect._empty:
                value = new_value
            
            return value
        
        return inner
        
    def __enter__(self):
        """
        Context manager entry point. Replaces torch.nn.Module.__call__ with wrapped version.
        
        Returns:
            The Interleaver instance
        """
        # Save the original __call__ method
        self.original_call = torch.nn.Module.__call__
    
        # Replace the __call__ method with our wrapped version
        torch.nn.Module.__call__ = self.wrap(torch.nn.Module.__call__, include_module=True)
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point. Restores original torch.nn.Module.__call__.
        
        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        self.cancel()
        # Restore the original __call__ method
        torch.nn.Module.__call__ = self.original_call
        self.original_call = None
            
    def __call__(self, fn: Callable, *args, **kwargs):
        """
        Execute a function with interventions.
        
        Args:
            fn: The function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
        """
        self.value_future = asyncio.Future()
        self.event_future = asyncio.Future()
        self.swap_future = asyncio.Future()
        # Create and start the task for running interventions
        self.async_loop = asyncio.get_event_loop()
        self.intervention_task = self.async_loop.create_task(self.interventions(self))
        
        self.wait()
                
        fn(*args, **kwargs)
        
        if self.event_future.done():
            requested_event, requester = self.event_future.result()
            raise ValueError(f"Execution complete but {requester} was not provided. Did you call an Envoy out of order? Investigate why this module was not called.")
        
    def hook(self, provider: Optional[Any] = None, value: Optional[Any] = None):
        """
        Hook function called during model execution to handle interventions.
        
        Args:
            provider: The module or function providing a value
            value: The value being provided
            
        Returns:
            Modified value if a swap is requested, otherwise inspect._empty
        """
        if self.event_future.done():
            event, data = self.event_future.result()
            self.event_future = asyncio.Future()
            
            if event == Events.VALUE:
                return self.handle_value_event(data, provider, value)
            elif event == Events.EXCEPTION:
                self.handle_exception_event(data)
            elif event == Events.CONTINUE:
                self.cancel()
                
        return inspect._empty
    
    def wait(self):
        """Wait for the next event to be set."""
        self.async_loop.run_until_complete(self.event_future)
        
    def handle_value_event(self, requester: Any, provider: Any, value: Any):
        """
        Handle a value event by providing the requested value or recording a missed provider.
        
        Args:
            requester: The module or function requesting a value
            provider: The module or function providing a value
            value: The value being provided
            
        Returns:
            Modified value if a swap is requested, otherwise inspect._empty
        """
        if provider == requester:
            self.set_value(value)

            if provider in self.missed_providers:
                self.missed_providers.remove(provider)
            
            self.wait()
            self.hook()
        else:
            if requester in self.missed_providers:
                raise ValueError(f"Value was missed for {requester}. Did you call an Envoy out of order?")
            
            self.missed_providers.add(provider)         
            self.event_future.set_result((Events.VALUE, requester))        
            
        return self.get_swap(provider)
                      
    def cancel(self):
        """Cancel the intervention task and close the async loop."""
        if not self.intervention_task.cancelled():
            self.intervention_task.cancel()
            
        self.event_future = asyncio.Future()
        self.async_loop.close()
        
    def handle_exception_event(self, exception: Exception):
        """
        Handle an exception event by raising the exception.
        
        Args:
            exception: The exception to raise
        """
        raise exception
                
    def set_value(self, value: Any):
        """
        Set the value for a pending value request.
        
        Args:
            value: The value to provide
        """
        self.value_future.set_result(value)
        self.value_future = asyncio.Future()
        
    async def get_value(self, requester: Any):
        """
        Request a value from a specific module or function.
        
        Args:
            requester: The module or function to request a value from
            
        Returns:
            The requested value
        """
        self.event_future.set_result((Events.VALUE, requester))
        return await self.value_future
    
    def set_swap(self, value: Any, envoy: "Envoy"):
        """
        Set a value to swap during execution.
        
        Args:
            value: The value to swap in
            envoy: The envoy requesting the swap
        """
        self.swap_future.set_result((value, envoy))
    
    def get_swap(self, provider: Any):
        """
        Get a swap value if one is available.
        
        Args:
            provider: The module or function providing a value
            
        Returns:
            The swap value if available, otherwise inspect._empty
        """
        if self.swap_future is not None and self.swap_future.done():
            value, setter = self.swap_future.result()
                        
            if setter != provider:
                raise ValueError(f"Setting {setter} is out of scope.")
            
            self.swap_future = asyncio.Future()
            
            return value
        
        return inspect._empty
        
    def continue_execution(self):
        """Signal that execution should continue without further intervention."""
        self.event_future.set_result((Events.CONTINUE, None))
        
    def exception(self, exception: Exception):
        """
        Signal that an exception occurred during intervention.
        
        Args:
            exception: The exception that occurred
        """
        self.event_future.set_result((Events.EXCEPTION, exception))