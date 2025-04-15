import inspect
import torch
from typing import Any, Callable, Dict, Tuple, Optional, TYPE_CHECKING
import asyncio
from enum import Enum
from functools import wraps
if TYPE_CHECKING:
    from .envoy import Envoy

class Events(Enum):
    INPUT = 'input'
    OUTPUT = 'output'
    CONTINUE = 'continue'
    EXCEPTION = 'exception'


class Interleaver:
    
    def __init__(self, interventions: Callable) -> None:
        self.interventions = interventions
        
        self.original_call = None
        self.value_future = None
        self.swap_future = None
        self.event_future = None
        self.async_loop = None
        
        
        
    def wrap(self, fn: Callable, include_module:bool=False, name:Optional[str]=None):
        
        @wraps(fn)
        def inner(module, *args, **kwargs):
            
            requester = module if name is None else (module, name)
            
            # Call our hook before the original __call__
            new_args = self.hook(Events.INPUT, requester, (args, kwargs))
            
            if new_args is not inspect._empty:
                args, kwargs = new_args

            # Call the original __call__ method
            if include_module:
                value = fn(module, *args, **kwargs)
            else:
                value = fn(*args, **kwargs)
            
            new_value = self.hook(Events.OUTPUT, requester, value)
            
            if new_value is not inspect._empty:
                value = new_value
            
            return value
        
        return inner
        
    def __enter__(self):
        # Save the original __call__ method
        self.original_call = torch.nn.Module.__call__
    
        # Replace the __call__ method with our wrapped version
        torch.nn.Module.__call__ = self.wrap(torch.nn.Module.__call__, include_module=True)
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the original __call__ method

        torch.nn.Module.__call__ = self.original_call
        self.original_call = None
            
    def __call__(self, fn: Callable, *args, **kwargs):
        self.value_future = asyncio.Future()
        self.event_future = asyncio.Future()
        self.swap_future = asyncio.Future()
        # Create and start the task for running interventions
        self.async_loop = asyncio.get_event_loop()
        self.intervention_task = self.async_loop.create_task(self.interventions(self))
        
        self.wait()
                
        fn(*args, **kwargs)
        
    def hook(self, event: Events, provider: Any, value: Any):
        if self.event_future is not None and self.event_future.done():
            
            requested_event, data = self.event_future.result()
            
            self.event_future = asyncio.Future()
            
            if requested_event == event:
                return self.handle_value_event(event, data, provider, value)
            elif requested_event == Events.EXCEPTION:
                self.handle_exception_event(data)
            elif requested_event == Events.CONTINUE:
          
                self.cancel()
            else:
                self.event_future.set_result((requested_event, data))
                
        return inspect._empty
    
    def wait(self):
        self.async_loop.run_until_complete(self.event_future)
    
    def handle_value_event(self, event:Events, requester:Any, provider: Any, value: Any):
                        
        if provider == requester:
                        
            self.set_value(value)
            
            self.wait()
            
            requested_event, exception = self.event_future.result()
            
            if requested_event in [Events.CONTINUE, Events.EXCEPTION]:
                self.event_future = None
                self.intervention_task.cancel()
                
                if requested_event == Events.EXCEPTION:
                    raise exception
        else:
            self.event_future.set_result((event, requester))
            
        return self.get_swap(provider, event)
                      
    def cancel(self):
        self.event_future = None
        self.value_future = None
        self.swap_future = None
        self.intervention_task.cancel()
        self.async_loop.close()

    def handle_exception_event(self, exception: Exception):
        self.cancel()
            
        raise exception
                
    def set_value(self, value: Any):
        self.value_future.set_result(value)
        self.value_future = asyncio.Future()
        
    async def get_value(self, event: Events, requester: Any):
        self.event_future.set_result((event, requester))
        return await self.value_future
    
    def set_swap(self, value: Any, envoy: "Envoy", event: Events):
        self.swap_future.set_result((value, envoy, event))
    
    def get_swap(self, provider: Any, event: Events):
        
        if self.swap_future is not None and self.swap_future.done():
            
            value, setter, setter_event = self.swap_future.result()
                        
            if setter is not provider or event != setter_event:
                raise ValueError(f"Setting {setter}.{setter_event.value} is out of scope.")
            
            self.swap_future = asyncio.Future()
            
            return value
        
        return inspect._empty
        
        
        
    def continue_execution(self):
        self.event_future.set_result((Events.CONTINUE, None))
        
    def exception(self, exception: Exception):
        
        self.event_future.set_result((Events.EXCEPTION, exception))