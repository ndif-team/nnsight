from __future__ import annotations


import ctypes
import inspect
import time
from types import MethodType
import types
import torch
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING, List
import threading
from enum import Enum
from functools import wraps
from threading import Thread
from queue import Queue
from ..tracing.util import get_frame
from .tracers.backwards import BackwardsTracer
if TYPE_CHECKING:
    from .envoy import Envoy
    from .tracers.tracer import Tracer, Cache
    from .envoy import EnvoySource
class Events(Enum):
    """Enum for different types of events in the interleaving process."""
    VALUE = 'value'      # Request for a value
    SWAP = 'swap'        # Request for a swap
    END = 'continue'  # Signal to continue execution
    EXCEPTION = 'exception'  # Signal that an exception occurred
    REGISTER = 'register'  # Signal that a mediator has been registered
class Cancelation(Exception):
    """Exception raised when a request is canceled."""
    pass

class EarlyStopException(Exception):
    """
    Exception raised to stop the execution of the model.
    """
    pass

class Interleaver:
    """
    Manages the interleaving of model execution and interventions.
    
    This class coordinates the flow between the model's forward pass and
    user-defined intervention functions, allowing for inspection and
    modification of intermediate values.
    """
    
    def __init__(self, mediators: List[Mediator]) -> None:
        """
        Initialize an Interleaver with intervention functions.
        
        Args:
            interventions: A callable that defines the intervention logic
        """
        self.mediators = {mediator.name: mediator for mediator in mediators}
        
        
        self.original_call = None
        self.original_grad = None
        self.original_backward = None
        
        self.missed_providers = set()
        self.state = dict()
        
    def wrap(self, fn: Callable):

        @wraps(fn)
        def inner(module:torch.nn.Module, *args, **kwargs):
            
            requester = module.__path__ 
            # Call our hook before the original __call__
            args, kwargs = self.handle(f"{requester}.input", (args, kwargs))
            
            value = fn(module, *args, **kwargs)
 
            value = self.handle(f"{requester}.output", value)
            
            return value
        
        return inner
    
    def wrap_operation(self, fn: Callable, name:str):
        
        @wraps(fn)
        def inner(*args, **kwargs):
            
            nonlocal fn
                    
            fn = self.handle(f"{name}.fn", fn)
            
            args, kwargs = self.handle(f"{name}.input", (args, kwargs))
            
            value = fn(*args, **kwargs)
 
            value = self.handle(f"{name}.output", value)
            
            return value
        
        return inner
    
    
    def grad(self):
        
        def inner(tensor: torch.Tensor):
            
            requester = id(tensor)
            
            def inner2(grad: torch.Tensor):
                
                grad = self.handle(f"{requester}.grad", grad)
                
                return grad
            
                tensor.register_hook(inner)

            
            
    def wrap_backward(self):
        
        def inner(tensor: torch.Tensor, *args, **kwargs):
            breakpoint()
            with BackwardsTracer(tensor) as tracer:
                pass
            
        return inner
        
      
                
    def __enter__(self):
        """
        Context manager entry point. Replaces torch.nn.Module.__call__ with wrapped version.
        
        Returns:
            The Interleaver instance
        """
        # Save the original __call__ method
        self.original_call = torch.nn.Module.__call__
        self.original_grad = torch.Tensor.register_hook
        self.original_backward = torch.Tensor.backward
        # Replace the __call__ method with our wrapped version
        torch.nn.Module.__call__ = self.wrap(torch.nn.Module.__call__)
      
            
        torch.Tensor.backward = self.wrap_backward()
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
        torch.Tensor.backward = self.original_backward
        
        self.original_call = None
        self.original_backward = None
            
    def __call__(self, fn: Callable, *args, **kwargs):
        """
        Execute a function with interventions.
        
        Args:
            fn: The function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
        """
        
        try:
        
            for mediator in list(self.mediators.values()):
                mediator.start(self)
                
                fn(*args, **kwargs)
            
        except EarlyStopException:
            pass
        
        #TODO check all mediator events
        
        #TODO
        # for mediator in self.mediators:
        #     if mediator.has_pending_event():
        #         requested_event, requester = mediator.get_event()
        #         print(requested_event, requester)
        #         raise ValueError(f"Execution complete but {requester} was not provided. Did you call an Envoy out of order? Investigate why this module was not called.")
        
    ### Provider Methods ###
    
    def handle(self, provider: Optional[Any] = None, value: Optional[Any] = None):
                
        for mediator in list(self.mediators.values()):
            new_value = mediator.handle(provider, value)
            
        if new_value is not value:
            return new_value
            
        return value
            
            
        #TODO concat all the mediators
        
    def cancel(self):
        """Cancel the intervention threads."""
        for mediator in self.mediators.values():
            mediator.cancel()
    
    ### Requester Methods ###
    
    def request(self, requester: Any):
                
        return self.mediators[threading.current_thread().name].request(requester)

    def swap(self, requester: Any, value: Any):
        
        return self.mediators[threading.current_thread().name].swap(requester, value)
    
    def iter(self, mediator: Mediator, iteration: int):
        
        return self.mediators[threading.current_thread().name].iter(mediator, iteration)
    
    def stop(self):
        
        return self.mediators[threading.current_thread().name].stop()
    
    def set_user_cache(self, cache: "Cache"):
        
        self.mediators[threading.current_thread().name].set_user_cache(cache) 
class Mediator:
    """
    Manages the interleaving of model execution and interventions.
    
    This class coordinates the flow between the model's forward pass and
    user-defined intervention functions, allowing for inspection and
    modification of intermediate values.
    """
    
    def __init__(self, intervention: Callable, info: "Tracer.Info", name: Optional[str] = None)  -> None:
       
       self.intervention = intervention
       self.name = name if name else f"Mediator{id(self)}"
       self.info = info
       
       self.event_queue = Queue()
       self.response_queue = Queue()
       self.swap_queue = Queue()
       
       self.thread = None
       
       self.interleaver = None
              
       self.history = set()
       
       self.cache = {}
       
       self.user_cache: "Cache" = None
              
       self._frame = None
       
    def start(self, interleaver: Interleaver):
        
        self.interleaver = interleaver
         
        self.thread = Thread(target=self.intervention, args=(self, self.info), daemon=True, name=self.name)
        self.thread.start()
        
        self.wait()
                
        self.handle()
        
    
    ### Provider Methods ###
        
    def wait(self):
        """Wait for the next event to be set."""
        """Wait until the event queue is not empty."""
        while self.event_queue.empty() and self.thread.is_alive():
            # Keep checking until there's an event in the queue
            time.sleep(0.001)  # Small sleep to prevent CPU spinning
        
    def cancel(self):
        """Cancel the intervention thread."""
        #TODO custom canceled error
        
        self.cache.clear()
        self.history.clear()
        
        self._frame = None
        
        if self.thread.is_alive():        
            self.response_queue.put(Cancelation())
        
    def handle(self, provider: Optional[Any] = None, value: Optional[Any] = None):
        """
        Hook function called during model execution to handle interventions.
        
        Args:
            provider: The module or function providing a value
            value: The value being provided
            
        Returns:
            Modified value if a swap is requested, otherwise inspect._empty
        """
        
        process = not self.event_queue.empty()
        
        while process:
            
            event, data = self.event_queue.get()            
            if event == Events.VALUE:
                process = self.handle_value_event(data, provider, value)
            elif event == Events.SWAP:
                requester, swap_value = data
                process = self.handle_swap_event(requester, provider, swap_value)
            elif event == Events.EXCEPTION:
                process = self.handle_exception_event(data)
            elif event == Events.END:
                self.cancel()
                process = False
            elif event == Events.REGISTER:
                process = self.handle_register_event(data)
            
        result = self.cache.get(provider, value)
            
        if self.user_cache is not None and provider is not None:
            
            self.user_cache.add(provider, result)
        
                    
        return result
    
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
            self.respond(value)
            
        else:
            if requester in self.history:
                self.respond(ValueError(f"Value was missed for {requester}. Did you call an Envoy out of order?"))
            else:
                self.history.add(provider)
                self.event_queue.put((Events.VALUE, requester))
                
                return False
                
        return True
                
    def handle_swap_event(self, requester: Any, provider: Any, swap_value: Any):
        """
        Handle a swap event by providing the requested value.
        """
        if provider == requester:
            self.respond()
        else:
            if requester in self.history:
                self.respond(ValueError(f"Setting {requester} is out of scope for scope {provider}. Did you call an Envoy out of order?"))
            else:
                self.history.add(provider)
                self.event_queue.put((Events.SWAP, (requester, swap_value)))
                
                return False
                
        return True              
        
    def handle_exception_event(self, exception: Exception):
        """
        Handle an exception event by raising the exception.
        
        Args:
            exception: The exception to raise
        """
        
        if not isinstance(exception, Cancelation):
            raise exception
        
        return False
    
    def handle_register_event(self, mediator: Mediator):
        
        # del self.interleaver.mediators[self.name]
        self.interleaver.mediators[mediator.name] = mediator
                
        mediator.start(self.interleaver)
        
        self.response_queue.put(None)
        
        return False
        
        
                
    def respond(self, value: Optional[Any] = None):
        """
        Set the value for a pending value request.
        
        Args:
            value: The value to provide
        """
        
        self.response_queue.put(value)
        
        self.wait()
                
    ### Requester Methods ###
    
    @property
    def frame(self):
        
        if self._frame is None:
            
            frame = inspect.currentframe()
            
            while frame:
                frame = frame.f_back
                if frame and frame.f_code.co_filename == "<string>":
                    break
            self._frame = frame
            
        return self._frame
    
    def push(self):
        
        self.interleaver.state.update(self.frame.f_locals)
        
    def pull(self):
        
        self.interleaver.state.pop('mediator', None)
        self.interleaver.state.pop('tracing_info', None)
        
        self.frame.f_globals.update(self.interleaver.state)
    
    def send(self, event: Events, requester: Any):
        
        self.push()
        
        self.event_queue.put((event, requester))

        response = self.response_queue.get()
        
        self.pull()
        
        if isinstance(response, Exception):
            raise response
    
        return response
      
    def request(self, requester: Any):
        """
        Request a value from a specific module or function.
        
        Args:
            requester: The module or function to request a value from
            
        Returns:
            The requested value
        """
                
        if requester in self.cache:
            return self.cache[requester]
        
        value = self.send(Events.VALUE, requester)
        
        self.cache[requester] = value
        
        return value
    
    def swap(self, requester: Any, value: Any):
        """
        Set a value to swap during execution.
        
        Args:
            value: The value to swap in
            envoy: The envoy requesting the swap
        """
        
        self.send(Events.SWAP, (requester, value))
        
        self.cache[requester] = value
        
    def iter(self, mediator: Mediator, iteration: int):
        
        for i in range(iteration):
                    
            self.send(Events.REGISTER, mediator)
            
            mediator.thread.join()
            
            self.interleaver.mediators.pop(mediator.name)
            
    def stop(self):
        
        self.push()
        
        raise EarlyStopException()
                           
    def end(self):
        """Signal that execution should continue without further intervention."""
        
        self.push()
        
        self.event_queue.put((Events.END, None))
        
    def exception(self, exception: Exception):
        """
        Signal that an exception occurred during intervention.
        
        Args:
            exception: The exception that occurred
        """
        self.event_queue.put((Events.EXCEPTION, exception))
            
    def set_user_cache(self, cache: "Cache"):
        
        self.user_cache = cache
        
        