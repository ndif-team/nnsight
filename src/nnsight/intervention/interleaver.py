from __future__ import annotations

import ctypes
import inspect
import threading
import time
from collections import defaultdict
from enum import Enum
from functools import wraps
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from ..util import Patch, Patcher, applyn
from .batching import Batcher
from .tracing.base import WithBlockNotFoundError
from .tracing.util import wrap_exception

if TYPE_CHECKING:
    from .tracing.tracer import Cache, Tracer, InterleavingTracer
    


class Events(Enum):
    """Enum for different types of events in the interleaving process."""

    VALUE = "value"  # Request for a value
    SWAP = "swap"  # Request for a swap
    END = "end"  # Signal to end the execution
    EXCEPTION = "exception"  # Signal that an exception occurred
    REGISTER = "register"  # Signal that a mediator has been registered
    SKIP = "skip"  # Signal that an operation should be skipped
    CONTINUE = "continue"  # Signal that an operation should continue


class Cancelation(Exception):
    """Exception raised when a request is canceled."""

    pass


class EarlyStopException(Exception):
    """
    Exception raised to stop the execution of the model.
    """

    pass

class SkipException(Exception):
    """
    Exception raised to skip the execution of the model.
    """
    def __init__(self, value: Any):
        self.value = value


class Interleaver:
    """
    Manages the interleaving of model execution and interventions.

    This class coordinates the flow between the model's forward pass and
    user-defined intervention functions, allowing for inspection and
    modification of intermediate values.
    """

    def __init__(self, invokers: List[Mediator], tracer: InterleavingTracer, batcher: Batcher = None, user_cache: Optional[Cache] = None) -> None:
        """
        Initialize an Interleaver with mediators.

        Args:
            mediators: A list of Mediator objects that define intervention logic
        """

        self.invokers = invokers
        self.batcher = batcher
        self.tracer = tracer
        self.mediators = {}

        self.patcher = None

        self.state = dict()
        self.iteration_tracker = defaultdict(int)

        self.user_cache: Optional[Cache] = user_cache
        
        #TODO legacy?
        self.default_all = None
        
    def iterate(self, provider:str):
        
        iteration = self.iteration_tracker[provider]

        self.iteration_tracker[provider] += 1

        return f"{provider}.i{iteration}"


    def wrap(self, fn: Callable):
        """
        Wrap a function to intercept inputs and outputs for intervention.

        Args:
            fn: The function to wrap

        Returns:
            A wrapped version of the function
        """

        @wraps(fn)
        def inner(module: torch.nn.Module, *args, **kwargs):

            if not hasattr(module, "__path__"):
                return fn(module, *args, **kwargs)

            provider = module.__path__
            
            # Call our hook before the original __call__
            
            try:
                inputs = self.handle(self.iterate(f"{provider}.input"), (args, kwargs))
            except SkipException as e:
                value = e.value
            else:
                args, kwargs = inputs
                value = fn(module, *args, **kwargs)
            
            value = self.handle(self.iterate(f"{provider}.output"), value)
    
            return value

        return inner

    def wrap_operation(self, fn: Callable, name: str, bound_obj: Optional[Any] = None):
        """
        Wrap an operation to intercept inputs and outputs for intervention.

        Args:
            fn: The function to wrap
            name: The name of the operation
            bound_obj: The object fn is bound to if it is a method

        Returns:
            A wrapped version of the function
        """

        @wraps(fn)
        def inner(*args, **kwargs):

            nonlocal fn

            fn = self.handle(f"{name}.fn", fn)

            args, kwargs = self.handle(f"{name}.input", (args, kwargs))

            if not inspect.ismethod(fn) and bound_obj is not None:
                value = fn(bound_obj, *args, **kwargs)
            else:
                value = fn(*args, **kwargs)

            value = self.handle(f"{name}.output", value)

            return value

        return inner

    def wrap_grad(self):
        """
        Create a hook for gradient intervention.

        Returns:
            A function that can be used to intercept gradients
        """

        def wrap(tensor: torch.Tensor):

            # Only wrap the tensor once
            if tensor._backward_hooks:
                return

            # We are providing the grad of the tensor
            provider = id(tensor)

            # Well need to remove the hook
            hook = None

            # On backwards for this tensor
            def inner(grad: torch.Tensor):

                hook.remove()
                # Inject the grad value
                # Possibly editing it in the process
                grad = self.handle(f"{provider}.grad", grad)

                return grad

            # Register the hook
            hook = tensor.register_hook(inner)

        def getter(tensor: torch.Tensor):

            wrap(tensor)

            requester = id(tensor)

            return self.current.request(f"{requester}.grad")

        def setter(tensor: torch.Tensor, value: torch.Tensor):

            wrap(tensor)

            requester = id(tensor)

            return self.current.swap(f"{requester}.grad", value)

        return property(getter, setter)

    def wrap_backward(self, fn: Callable):
        """
        Wrap the backward method to intercept backpropagation.

        Returns:
            A wrapped version of the backward method
        """
        

        def inner(tensor: torch.Tensor, *args, **kwargs):
            
            from .tracing.backwards import BackwardsTracer

            try:

                tracer = BackwardsTracer(tensor, fn, self, *args, **kwargs)

            except WithBlockNotFoundError:

                return fn(tensor, *args, **kwargs)

            return tracer

        return inner
    
    def check_cache_full(self):
        """
        Print a warning if a module to be cached was missed.
        """
        for invoker in self.invokers:
            for cache in invoker.user_cache:
                if cache.modules:
                    if cache.include_inputs and cache.include_output:
                        for module in cache.modules:
                            if module not in cache.cache or cache.cache[module].inputs is None:
                                print('\033[33m' + "NNsight Warning: A module to be cached was missed! Consider defining the Cache before the module is called." + '\033[0m')
                                return
                    else:
                        if any(module not in cache.cache for module in cache.modules):
                            print('\033[33m' + "NNsight Warning: A module to be cached was missed! Consider defining the Cache before the module is called." + '\033[0m')
                            return


    def __enter__(self):
        """
        Context manager entry point. Replaces torch.nn.Module.__call__ with wrapped version.

        Returns:
            The Interleaver instance
        """

        self.patcher = Patcher(
            [
                Patch(torch.nn.Module, self.wrap(torch.nn.Module.__call__), "__call__"),
                Patch(
                    torch.Tensor, self.wrap_backward(torch.Tensor.backward), "backward"
                ),
            ]
        )

        self.patcher.__enter__()

        # torch.Tensor.backward = self.wrap_backward()
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

        self.patcher.__exit__(None, None, None)

    def __call__(self, fn: Callable, *args, **kwargs):
        """
        Execute a function with interventions.

        Args:
            fn: The function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
        """

        try:

            for invoker in self.invokers:
                invoker.start(self)

            fn(*args, **kwargs)

        except EarlyStopException:
            pass

        # TODO check all mediator events

        # TODO
        # for mediator in self.mediators:
        #     if mediator.has_pending_event():
        #         requested_event, requester = mediator.get_event()
        #         print(requested_event, requester)
        #         raise ValueError(f"Execution complete but {requester} was not provided. Did you call an Envoy out of order? Investigate why this module was not called.")

    ### Provider Methods ###

    def handle(
        self,
        provider: Optional[Any] = None,
        value: Optional[Any] = None,
    ):
        """
        Handle a provider's value, allowing mediators to consume and modify it.

        Args:
            provider: The identifier of the provider
            value: The value being provided

        Returns:
            The original or modified value
        """

                    
        old = self.batcher.current_value
                    
        self.batcher.current_value = value

        batch_size = len(self.batcher.batch_groups)
        skip_count = 0
        skip_values = []
        
        for mediator in self.invokers:

            try:
                mediator.handle(provider)
            except SkipException as e:
                skip_count += 1
                skip_values.append(e.value)

        if skip_count == batch_size:
            def _swap(*args):
                return torch.cat(args, dim=0)

            skip_value = applyn(skip_values, _swap, torch.Tensor)
            raise SkipException(skip_value)
        elif skip_count > 0 and skip_count < batch_size:
            raise ValueError(f"A module skip must be applied to all the invokers defined in the tracer!")
            
        value = self.batcher.current_value
        
        self.batcher.current_value = old

        if len(self.user_cache) > 0:
            for cache in self.user_cache:
                cache.add(provider, value)
            
        return value

    def cancel(self):
        """Cancel all intervention threads."""
        
        for mediator in list(self.mediators.values()):
            mediator.cancel()

        self.mediators = None
        self.tracer = None
        self.batcher = None

    ### Requester Methods ###

    @property
    def current(self) -> Mediator:
        """Get the current mediator."""
        return self.mediators[threading.current_thread().name]



class Mediator:
    """
    Mediates between the model execution and intervention functions.

    This class handles the communication between the model's forward pass and
    user-defined intervention functions, allowing for inspection and
    modification of intermediate values.
    """

    class OutOfOrderError(Exception):
        """
        Exception raised when interventions are defined out of order.
        """
        pass

    def __init__(
        self,
        intervention: Callable,
        info: "Tracer.Info",
        name: Optional[str] = None,
        batch_group: Optional[int] = 0,
    ) -> None:
        """
        Initialize a Mediator with an intervention function.

        Args:
            intervention: The intervention function
            info: Information about the tracing context
            name: Optional name for the mediator
        """
        self.intervention = intervention
        self.name = name if name else f"Mediator{id(self)}"
        self.info = info
        self.batch_group = batch_group
        self.event_queue = Queue()
        self.response_queue = Queue()

        self.thread = None
        self.interleaver = None
        self.child: Mediator = None
        self.history = set()
        self.user_cache: List["Cache"] = list()
        self.iteration = 0

        self.args = list()

    def start(self, interleaver: Interleaver):
        """
        Start the mediator's intervention thread.

        Args:
            interleaver: The interleaver managing this mediator
        """
        self.interleaver = interleaver

        self.interleaver.mediators[self.name] = self

        self.thread = Thread(
            target=self.intervention,
            args=(self, self.info,  *self.args),
            daemon=True,
            name=self.name,
        )
        self.thread.start()

        self.wait()

        self.handle()

    ### Provider Methods ###

    def wait(self):
        """Wait for the next event to be set in the event queue."""
        while self.event_queue.empty() and self.thread.is_alive():
            # Keep checking until there's an event in the queue
            time.sleep(0.001)  # Small sleep to prevent CPU spinning

    def cancel(self):
        """Cancel the intervention thread and clear caches."""
        # TODO custom canceled error

        self.interleaver.mediators.pop(self.name)

        self.history.clear()

        self.thread = None
        
        self.state = None

        if self.thread is not None and self.thread.is_alive():
            # TODO: cancel inactive threads at the end of the model's execution
            self.response_queue.put(Cancelation())

    def handle(self, provider: Optional[Any] = None):
        """
        Handle events in the event queue and process provider values.

        Args:
            provider: The identifier of the provider

        Returns:
            The original or modified value
        """

        if self.child is not None:

            self.child.handle(provider)

            # If the child was canceled
            if self.child.thread is None:

                # Tell the parent (current mediator) to continue
                self.response_queue.put(None)

                # Wait until the parent has an event
                self.wait()

                # Then wipe the child
                self.child = None

                # Continue to handle parent event

        process = not self.event_queue.empty()

        event = None
        
        while process:
            
            value = self.interleaver.batcher.current_value

            event, data = self.event_queue.get()
    
            if event == Events.VALUE:
                process = self.handle_value_event(data, provider, value)
            elif event == Events.SWAP:
                process = self.handle_swap_event(provider, *data)
            elif event == Events.REGISTER:
                process = self.handle_register_event(*data)
            elif event == Events.EXCEPTION:
                process = self.handle_exception_event(data)
            elif event == Events.SKIP:
                try:
                    process = self.handle_skip_event(provider, *data)
                except SkipException as e:
                    if len(self.user_cache) > 0:
                        for cache in self.user_cache:
                            cache.add(provider, e.value)
                    raise e
            elif event == Events.END:
                process = False
            elif event == Events.CONTINUE:
                process = False

        if event == Events.END:
            self.handle_end_event()

        # TODO maybe move this to the interleaver to cache the pre-iteration provider
        if len(self.user_cache) > 0 and provider is not None:

            for cache in self.user_cache:
                cache.add(provider, self.interleaver.batcher.narrow(self.batch_group, self.interleaver.batcher.current_value))
                        
    def handle_end_event(self):
        """
        Handle an end event by stopping the mediator.
        """
        self.cancel()

    def handle_value_event(self, requester: Any, provider: Any, value: Any):
        """
        Handle a value event by providing the requested value or recording a missed provider.

        Args:
            requester: The identifier of the requester
            provider: The identifier of the provider
            value: The value being provided

        Returns:
            Boolean indicating whether to continue processing events
        """

        if provider == requester:

            value = self.interleaver.batcher.narrow(self.batch_group, value)

            self.respond(value)
        else:
            if requester in self.history:
                # TODO needs tests
                self.respond(
                    Mediator.OutOfOrderError(
                        f"Value was missed for {requester}. Did you call an Envoy out of order?"
                    )
                )
            else:
                self.history.add(provider)
                self.event_queue.put((Events.VALUE, requester))

                return False

        return True

    def handle_swap_event(self, provider: Any, requester: Any,  swap_value: Any):
        """
        Handle a swap event by swapping the value if the provider matches the requester.

        Args:
            requester: The identifier of the requester
            provider: The identifier of the provider
            swap_value: The value to swap in

        Returns:
            Boolean indicating whether to continue processing events
        """
        if provider == requester:
            
            self.interleaver.batcher.swap(self.batch_group, swap_value)
     
            self.respond()

            return True

        else:
            if requester in self.history:
                # TODO needs tests
                self.respond(
                    ValueError(
                        f"Setting {requester} is out of scope for scope {provider}. Did you call an Envoy out of order?"
                    )
                )
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

        Returns:
            Boolean indicating whether to continue processing events
        """

        if not isinstance(exception, Cancelation):

            exception = wrap_exception(exception, self.info)

            raise exception

        return False
    
    def handle_skip_event(self, provider:Any, requester: Any, value: Any):
        
        if provider == requester:
            
            self.respond()

            self.history.add(provider)
            
            raise SkipException(value)
        
        else:
            if requester in self.history:
                self.respond(
                    Mediator.OutOfOrderError(
                        f"Value was missed for {requester}. Did you call an Envoy out of order?"
                    )
                )

                return True
            else:
                self.history.add(provider)
                self.event_queue.put((Events.SKIP, (requester, value)))
                
                return False
        
    def handle_register_event(self, mediator: Mediator, fn: Optional[Callable] = None):
        """
        Handle a register event by registering a new mediator.

        Args:
            mediator: The mediator to register

        Returns:
            Boolean indicating whether to continue processing events
        """

        self.child = mediator

        self.child.info.start_line += self.info.start_line - 1

        mediator.start(self.interleaver)

        self.response_queue.put(None)

        if fn is not None:

            fn()

            self.response_queue.put(None)

            self.wait()

            self.handle()

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
        """
        Get the frame of the intervention function.

        Returns:
            The frame of the intervention function
        """

        frame = inspect.currentframe()

        while frame:
            frame = frame.f_back
            if frame and frame.f_code.co_filename.startswith("<nnsight"):
                break
        return frame
    
    def iterate(self, requester: Any):
        
        return f"{requester}.i{self.iteration}"

    def push(self):
        """Push local variables to the interleaver state."""
        self.interleaver.state.update(self.frame.f_locals)

    def pull(self):
        """Pull variables from the interleaver state to the frame globals."""

        for key in list(self.interleaver.state.keys()):
            if key.startswith("__nnsight"):
                self.interleaver.state.pop(key)

        self.frame.f_globals.update(self.interleaver.state)
        self.frame.f_locals.update(self.interleaver.state)

        ctypes.pythonapi.PyFrame_LocalsToFast(
            ctypes.py_object(self.frame), ctypes.c_int(0)
        )

    def send(self, event: Events, requester: Any):
        """
        Send an event to the event queue and wait for a response.

        Args:
            event: The event to send
            requester: The identifier of the requester

        Returns:
            The response from the provider
        """
        self.push()

        self.event_queue.put((event, requester))

        response = self.response_queue.get()

        self.pull()

        if isinstance(response, Exception):
            raise response

        return response

    def request(self, requester: Any):
        """
        Request a value from a specific provider.

        Args:
            requester: The identifier of the provider to request a value from

        Returns:
            The requested value
        """

    
        value = self.send(Events.VALUE, requester)

        return value

    def register(self, mediator: Mediator, fn: Optional[Callable] = None):

        self.send(Events.REGISTER, (mediator, fn))

        self.response_queue.get()

        if fn is not None:
            self.event_queue.put((Events.CONTINUE, None))

            self.response_queue.get()

        self.pull()

    def swap(self, requester: Any, value: Any):
        """
        Set a value to swap during execution.

        Args:
            requester: The identifier of the requester
            value: The value to swap in
        """

        self.send(Events.SWAP, (requester, value))

    def iter(self, mediator: Mediator, iteration: Union[int, slice]):
        """
        Iterate a mediator a specified number of times.

        Args:
            mediator: The mediator to iterate
            iteration: The number of iterations
        """

        if isinstance(iteration, slice):            

            i = iteration.start if iteration.start is not None else self.iteration
            
            stop = iteration.stop

            while True:

                mediator.iteration = i
                mediator.args = list([mediator.iteration])

                self.register(mediator)

                if self.interleaver.default_all is not None and stop is None:
                    stop = self.interleaver.default_all

                i += 1

                if stop is not None and i >= stop:
                    break
                
        elif isinstance(iteration, int):

            mediator.iteration = iteration
            mediator.args = list([mediator.iteration])

            self.register(mediator)
            
    def stop(self):
        """Stop the execution of the model by raising an EarlyStopException."""

        self.push()

        raise EarlyStopException()
    
    def skip(self, requester: Any, value: Any):
        
        self.send(Events.SKIP, (requester, value))

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
        """
        Set the user cache for this mediator.

        Args:
            cache: The cache to set
        """

        self.user_cache.append(cache)

    ### Serialization ###

    def __getstate__(self):
        """Get the state of the mediator for serialization."""
        
        return {
            "name": self.name,
            "info": self.info,
            "batch_group": self.batch_group,
            "intervention": self.intervention,
        }
        
    def __setstate__(self, state):
        """Set the state of the mediator for deserialization."""
        self.name = state["name"]
        self.info = state["info"]
        self.batch_group = state["batch_group"]
        self.intervention = state["intervention"]
        
        self.event_queue = Queue()
        self.response_queue = Queue()

        self.thread = None
        self.interleaver = None
        self.child: Mediator = None
        self.history = set()
        self.user_cache: "Cache" = list()
        self.iteration = 0
        self.args = list()