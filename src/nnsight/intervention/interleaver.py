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

from ..util import Patch, Patcher
from .tracing.util import wrap_exception
from .batching import Batcher
if TYPE_CHECKING:
    from .tracing.tracer import Cache, Tracer



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


class Interleaver:
    """
    Manages the interleaving of model execution and interventions.

    This class coordinates the flow between the model's forward pass and
    user-defined intervention functions, allowing for inspection and
    modification of intermediate values.
    """

    def __init__(
        self, invokers: List[Mediator], batcher: Batcher = None
    ) -> None:
        """
        Initialize an Interleaver with mediators.

        Args:
            mediators: A list of Mediator objects that define intervention logic
        """

        self.invokers = invokers
        self.batcher = batcher

        self.mediators = {}

        self.patcher = None

        self.state = dict()
        self.iteration_tracker = defaultdict(int)


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
            inputs = self.handle(f"{provider}.input", (args, kwargs), itertative=True)

            if inputs is Events.SKIP:
                value = Events.SKIP
            else:
                args, kwargs = inputs
                value = fn(module, *args, **kwargs)

            value = self.handle(f"{provider}.output", value, itertative=True)

            return value

        return inner

    def wrap_operation(self, fn: Callable, name: str):
        """
        Wrap an operation to intercept inputs and outputs for intervention.

        Args:
            fn: The function to wrap
            name: The name of the operation

        Returns:
            A wrapped version of the function
        """

        @wraps(fn)
        def inner(*args, **kwargs):

            nonlocal fn

            fn = self.handle(f"{name}.fn", fn)

            args, kwargs = self.handle(f"{name}.input", (args, kwargs))

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

            return self.request(f"{requester}.grad")

        def setter(tensor: torch.Tensor, value: torch.Tensor):

            wrap(tensor)

            requester = id(tensor)

            return self.swap(f"{requester}.grad", value)

        return property(getter, setter)

    def wrap_backward(self, fn: Callable):
        """
        Wrap the backward method to intercept backpropagation.

        Returns:
            A wrapped version of the backward method
        """
        from .tracing.backwards import BackwardsTracer

        def inner(tensor: torch.Tensor, *args, **kwargs):

            tracer = BackwardsTracer(tensor, fn, self, *args, **kwargs)
            
            if not tracer.with_node:
                return fn(tensor, *args, **kwargs)
            
            return tracer
                
        return inner

    def __enter__(self):
        """
        Context manager entry point. Replaces torch.nn.Module.__call__ with wrapped version.

        Returns:
            The Interleaver instance
        """

        self.patcher = Patcher(
            [
                Patch(torch.nn.Module, self.wrap(torch.nn.Module.__call__), "__call__"),
                Patch(torch.Tensor, self.wrap_grad(), "grad"),
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

    def handle(self, provider: Optional[Any] = None, value: Optional[Any] = None, itertative:bool = False):
        """
        Handle a provider's value, allowing mediators to consume and modify it.

        Args:
            provider: The identifier of the provider
            value: The value being provided

        Returns:
            The original or modified value
        """
        
        if isinstance(provider, str) and itertative:

            iteration = self.iteration_tracker[provider]

            self.iteration_tracker[provider] += 1

            provider = f"{provider}.i{iteration}"

        original_value = value

        swap_happened = False

        for mediator in self.invokers:
                        
            value = mediator.handle(provider, original_value)

            if original_value is Events.SKIP:
                if value is not Events.SKIP:
                    return value
            elif value is Events.SKIP:
                return value
                
        if original_value is not value:
            return value

        return value

        # TODO concat all the mediators

    def cancel(self):
        """Cancel all intervention threads."""
        for mediator in list(self.mediators.values()):
            mediator.cancel()

    ### Requester Methods ###

    @property
    def current(self) -> Mediator:
        """Get the current mediator."""
        return self.mediators[threading.current_thread().name]

    def request(self, requester: Any, itertative:bool = False):
        """
        Request a value from the current mediator.

        Args:
            requester: The identifier of the requester

        Returns:
            The requested value
        """
        return self.current.request(requester, itertative)

    def swap(self, requester: Any, value: Any, itertative:bool = False):
        """
        Swap a value in the current mediator.

        Args:
            requester: The identifier of the requester
            value: The value to swap in
        """
        return self.current.swap(requester, value, itertative)

    def register(self, mediator: Mediator, fn: Optional[Callable] = None):
        """
        Register a mediator with the current mediator.

        Args:
            mediator: The mediator to register
        """
        return self.current.register(mediator, fn)

    def iter(self, mediator: Mediator, iteration: Union[int, slice]):
        """
        Iterate a mediator a specified number of times.

        Args:
            mediator: The mediator to iterate
            iteration: The number of iterations
        """
        return self.current.iter(mediator, iteration)

    def stop(self):
        """Stop the execution of the model."""
        return self.current.stop()

    def set_user_cache(self, cache: "Cache"):
        """
        Set the user cache for the current mediator.

        Args:
            cache: The cache to set
        """
        self.current.set_user_cache(cache)


class Mediator:
    """
    Mediates between the model execution and intervention functions.

    This class handles the communication between the model's forward pass and
    user-defined intervention functions, allowing for inspection and
    modification of intermediate values.
    """

    def __init__(
        self, intervention: Callable, info: "Tracer.Info", name: Optional[str] = None, batch_group: Optional[int] = 0
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
        self.user_cache: "Cache" = None
        self.iteration = 0

        self._frame = None

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
            args=(self, self.info),
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

        self._frame = None

        if self.thread is not None and self.thread.is_alive():
            self.response_queue.put(Cancelation())

    def handle(self, provider: Optional[Any] = None, value: Optional[Any] = None):
        """
        Handle events in the event queue and process provider values.

        Args:
            provider: The identifier of the provider
            value: The value being provided

        Returns:
            The original or modified value

        """


        if self.child is not None:

            value = self.child.handle(provider, value)

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
            
            event, data = self.event_queue.get()
        
            if event == Events.VALUE:
                process = self.handle_value_event(data, provider, value)
            elif event == Events.SWAP:
                requester, swap_value = data
                process, new_value = self.handle_swap_event(requester, provider, swap_value)
                
                if new_value is not inspect._empty:
                    value = new_value
                    
            elif event == Events.REGISTER:
                process = self.handle_register_event(*data)
            elif event == Events.EXCEPTION:
                process = self.handle_exception_event(data)
            elif event == Events.END:
                process = False
            elif event == Events.CONTINUE:
                process = False

        if event == Events.END:
            self.handle_end_event()

        # TODO maybe move this to the interleaver to cache the pre-iteration provider
        if self.user_cache is not None and provider is not None:

            self.user_cache.add(provider, value)

        return value

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
                self.respond(
                    ValueError(
                        f"Value was missed for {requester}. Did you call an Envoy out of order?"
                    )
                )
            else:
                self.history.add(provider)
                self.event_queue.put((Events.VALUE, requester))

                return False

        return True

    def handle_swap_event(self, requester: Any, provider: Any, swap_value: Any):
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
            self.respond()

            if swap_value is Events.SKIP:
                return False, swap_value
            
            return True, swap_value

        else:
            if requester in self.history:
                self.respond(
                    ValueError(
                        f"Setting {requester} is out of scope for scope {provider}. Did you call an Envoy out of order?"
                    )
                )
            else:
                self.history.add(provider)
                self.event_queue.put((Events.SWAP, (requester, swap_value)))

                return False, inspect._empty

        return True, inspect._empty

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

        if self._frame is None:

            frame = inspect.currentframe()

            while frame:
                frame = frame.f_back
                if frame and frame.f_code.co_filename.startswith("<nnsight"):
                    break
            self._frame = frame

        return self._frame

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

    def request(self, requester: Any, itertative:bool = False):
        """
        Request a value from a specific provider.

        Args:
            requester: The identifier of the provider to request a value from

        Returns:
            The requested value
        """

        if itertative:
            requester = f"{requester}.i{self.iteration}"

        value = self.send(Events.VALUE, requester)

        return value

    def register(self, mediator: Mediator, fn: Optional[Callable] = None):

        self.send(Events.REGISTER, (mediator, fn))
        
        self.response_queue.get()
        
        if fn is not None:
            self.event_queue.put((Events.CONTINUE, None))
            
            self.response_queue.get()
            
        self.pull()

    def swap(self, requester: Any, value: Any, itertative:bool = False):
        """
        Set a value to swap during execution.

        Args:
            requester: The identifier of the requester
            value: The value to swap in
        """

        if itertative:
            requester = f"{requester}.i{self.iteration}"

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

            while True:

                mediator.iteration = i

                self.register(mediator)

                i += 1

                if i >= iteration.stop:
                    break

        elif isinstance(iteration, int):

            mediator.iteration = i

            self.register(mediator)

    def stop(self):
        """Stop the execution of the model by raising an EarlyStopException."""

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
        """
        Set the user cache for this mediator.

        Args:
            cache: The cache to set
        """

        self.user_cache = cache
