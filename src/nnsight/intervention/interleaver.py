from __future__ import annotations

import ctypes
import inspect
import re
import threading
import time
import warnings
from collections import defaultdict
from enum import Enum
from functools import wraps
from queue import Queue
from threading import Thread
from types import FrameType
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple,
                    Union)

import torch

from ..util import applyn
from .batching import Batcher
from .tracing.util import wrap_exception, get_non_nnsight_frame, push_variables

if TYPE_CHECKING:
    from .tracing.tracer import Cache, InterleavingTracer, Tracer


class Events(Enum):
    """Enum for different types of events in the interleaving process."""

    VALUE = "value"  # Request for a value
    SWAP = "swap"  # Request for a swap
    END = "end"  # Signal to end the execution
    EXCEPTION = "exception"  # Signal that an exception occurred
    SKIP = "skip"  # Signal that an operation should be skipped
    REGISTER = "register"  # Signal that a child mediator should be registered
    BARRIER = "barrier"  # Signal that a barrier should be set


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
    
    def __init__(
        self,
        invokers: List[Mediator] = None,
        tracer: InterleavingTracer = None,
        batcher: Batcher = None,
        user_cache: Optional[Cache] = None,
    ):
        self.initialize(invokers, tracer, batcher, user_cache)


    def initialize(
        self,
        invokers: List[Mediator],
        tracer: InterleavingTracer,
        batcher: Batcher = None,
        user_cache: Optional[Cache] = None,
    ):

        self.invokers = invokers
        self.tracer = tracer
        self.batcher = batcher if batcher is not None else Batcher()
        self.user_cache = user_cache

        self.mediators: Dict[str, Mediator] = {}
        self.iteration_tracker = defaultdict(int)
        self.default_all = None
        
    def cancel(self):
        """Cancel all intervention threads."""
     
        for mediator in list(self.mediators.values()):
            mediator.cancel()

        self.mediators = None
        self.tracer = None
        self.batcher = None
        self.user_cache = None
        self.invokers = None


    def iterate(self, provider: str):

        iteration = self.iteration_tracker[provider]

        self.iteration_tracker[provider] += 1

        return f"{provider}.i{iteration}"

    def wrap_module(self, module: torch.nn.Module):

        skip = None

        forward = module.forward

        @wraps(module.forward)
        def skippable_forward(*args, **kwargs):

            nonlocal skip

            if skip is None or not self.interleaving:
                try:
                    return forward(*args, **kwargs)
                finally:
                    skip = None

            return skip

        module.forward = skippable_forward

        @torch._dynamo.disable
        def input_hook(module: torch.nn.Module, args, kwargs):

            if not self.interleaving:
                return args, kwargs

            provider = module.__path__

            nonlocal skip

            try:
                inputs = self.handle(f"{provider}.input", (args, kwargs), iterate=True)
            except SkipException as e:
                skip = e.value
            else:
                args, kwargs = inputs

            return args, kwargs

        module.register_forward_pre_hook(input_hook, with_kwargs=True, prepend=True)

        @torch._dynamo.disable
        def output_hook(module: torch.nn.Module, _, output: Any):

            if not self.interleaving:
                return output

            provider = module.__path__

            nonlocal skip

            if skip is not None:

                output = skip

                skip = None

            output = self.handle(f"{provider}.output", output, iterate=True)

            return output

        module.register_forward_hook(output_hook, prepend=True)

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


    @property
    def interleaving(self):
        return getattr(self, "_interleaving", False)

    def __enter__(self):
        
        self._interleaving = True
        
        try:
            for invoker in self.invokers:
                invoker.start(self)

            try:
                self.handle()
            except EarlyStopException:
                pass
        except:
            self._interleaving = False
            raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        self._interleaving = False
        
        
        # If execution was stopped early, ignore and do nothing
        if exc_type is not None and issubclass(exc_type, EarlyStopException):
            return True
        
    def check_dangling_mediators(self):
        
        # If any mediators are still waiting for their values for their events, they probably called an Envoy out of order
        # Or their Envoy was not called.
        for mediator in self.mediators.values():

            if mediator.child is not None:
                mediator = mediator.child

            if not mediator.event_queue.empty():
                requested_event, requester = mediator.event_queue.get()

                if isinstance(requester, tuple):
                    requester = requester[0]

                mediator.respond(
                    ValueError(
                        f"Execution complete but `{requester}` was not provided. Did you call an Envoy out of order? Investigate why this module was not called?"
                    )
                )
                mediator.wait()

                if mediator.name.startswith("Iterator"):
                    try:
                        mediator.handle()
                    except ValueError as e:
                        msg = f"Execution complete but `{requester}` was not provided. This was in an Iterator at iteration {mediator.iteration} so likely this iteration did not happen. If you were using `.iter[:]`, this is likely not an error."
                        warnings.warn(msg)
                else:
                    mediator.handle()
                    
    def check_cache_full(self):
        """
        Print a warning if a module to be cached was missed.
        """
        for invoker in self.invokers:
            for cache in invoker.user_cache:
                if cache.modules:
                    if cache.include_inputs and cache.include_output:
                        for module in cache.modules:
                            if (
                                module not in cache.cache
                                or cache.cache[module].inputs is None
                            ):
                                print(
                                    "\033[33m"
                                    + "NNsight Warning: A module to be cached was missed! Consider defining the Cache before the module is called."
                                    + "\033[0m"
                                )
                                return
                    else:
                        if any(module not in cache.cache for module in cache.modules):
                            print(
                                "\033[33m"
                                + "NNsight Warning: A module to be cached was missed! Consider defining the Cache before the module is called."
                                + "\033[0m"
                            )
                            return

    ### Provider Methods ###
    
    def handle(
        self,
        provider: Optional[Any] = None,
        value: Optional[Any] = None,
        iterate: bool = False,
    ):
        """
        Handle a provider's value, allowing mediators to consume and modify it.

        Args:
            provider: The identifier of the provider
            value: The value being provided

        Returns:
            The original or modified value
        """

        if iterate:
            provider = self.iterate(provider)

        old = self.batcher.current_value

        self.batcher.current_value = value

        skip_count = 0
        skip_values = []

        for mediator in self.invokers:

            try:
                mediator.handle(provider)
            except SkipException as e:
                skip_count += 1
                skip_values.append(e.value)

        if skip_count == len(self.invokers) and self.invokers:

            def _swap(*args):
                return torch.cat(args, dim=0)

            skip_value = applyn(skip_values, _swap, torch.Tensor)
            raise SkipException(skip_value)
        elif skip_count > 0 and skip_count < len(self.invokers):
            raise ValueError(
                f"A module skip must be applied to all the invokers defined in the tracer!"
            )

        value = self.batcher.current_value

        self.batcher.current_value = old

        if (
            self.user_cache is not None
            and len(self.user_cache) > 0
            and provider is not None
        ):
            for cache in self.user_cache:
                cache.add(provider, value)

        return value

    ### Requester Methods ###

    @property
    def current(self) -> Mediator:
        """Get the current mediator."""
        return self.mediators[threading.current_thread().name]

    ### Serialization ###

    def __deepcopy__(self, memo):

        return self


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
        stop: Optional[int] = None,
    ) -> None:
        """
        Initialize a Mediator with an intervention function.

        Args:
            intervention: The intervention function
            info: Information about the tracing context
            name: Optional name for the mediator
            stop: Optional number of times to execute this mediator
        """
        self.intervention = intervention
        self.name = name if name else f"Mediator{id(self)}"
        self.info = info
        self.batch_group = batch_group
        self.event_queue = Queue()
        self.response_queue = Queue()
        
        self.child: Mediator = None

        self.thread = None
        self.interleaver = None
        self.history = set()
        self.user_cache: List["Cache"] = list()
        self.iteration = 0
        self.all_stop: Optional[int] = stop

        self.args = list()

    @property
    def alive(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, interleaver: Interleaver):
        """
        Start the mediator's intervention thread.

        Args:
            interleaver: The interleaver managing this mediator
        """
        self.interleaver = interleaver

        self.interleaver.mediators[self.name] = self

        if not self.alive:

            self.thread = Thread(
                target=self.intervention,
                args=(self, self.info, *self.args),
                daemon=True,
                name=self.name,
            )
            self.thread.start()

            self.wait()

    ### Provider Methods ###

    def wait(self):
        """Wait for the next event to be set in the event queue."""
        while self.event_queue.empty() and self.alive:
            # Keep checking until there's an event in the queue
            time.sleep(0.001)  # Small sleep to prevent CPU spinning

    def cancel(self):
        """Cancel the intervention thread and clear caches."""
        # TODO custom canceled error

        self.interleaver.mediators.pop(self.name)

        self.history.clear()

        self.thread = None

        if self.alive:
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
            
            if not self.child.alive:
                self.child = None
                self.respond()
            else:
                return
            
        process = not self.event_queue.empty()

        event = None
        
        while process:
            
            value = self.interleaver.batcher.current_value

            event, data = self.event_queue.get()

            if event == Events.VALUE:
                process = self.handle_value_event(data, provider, value)
            elif event == Events.SWAP:
                process = self.handle_swap_event(provider, *data)
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
            elif event == Events.REGISTER:
                process = self.handle_register_event(provider, data)
            elif event == Events.BARRIER:
                process = self.handle_barrier_event(provider, data)
            elif event == Events.END:
                process = False

        if event == Events.END:
            self.handle_end_event()

        # TODO maybe move this to the interleaver to cache the pre-iteration provider
        if len(self.user_cache) > 0 and provider is not None:

            for cache in self.user_cache:
                cache.add(
                    provider,
                    self.interleaver.batcher.narrow(
                        self.batch_group, self.interleaver.batcher.current_value
                    ),
                )
                
    def handle_register_event(self, provider: Any, child:Mediator) -> bool:
                
        self.child = child
        child.start(self.interleaver)
        child.handle(provider)
        
        return False

    def handle_barrier_event(self, provider: Any, participants: Set[str]):
        """
        Handle a barrier event by setting a barrier.
        """
        
        if participants is not None:
        
            for mediator in self.interleaver.invokers:
                
                while mediator.child is not None:
                    mediator = mediator.child
                    
                if mediator.name in participants:
               
                    mediator.respond()
        
                    mediator.handle(provider)

                    
                    
                            


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

    def handle_swap_event(self, provider: Any, requester: Any, swap_value: Any):
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

    def handle_skip_event(self, provider: Any, requester: Any, value: Any):

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
    def frame(self) -> FrameType:
        """
        Get the frame of the intervention function.

        Returns:
            The frame of the intervention function
        """
        
       

        frame = get_non_nnsight_frame()

        return frame

    def iterate(self, requester: Any):

        return f"{requester}.i{self.iteration}"

    def push(self):
        """Push local variables to the interleaver state."""
        
        state = {k: v for k, v in self.frame.f_locals.items() if not k.startswith("__nnsight")}
         # this does not handle the case of a fn thats called in an invoker. this will push vars directly to where the invoke was called not the fn. really we need to grad the f_back of the <nnsight> frame. If its in threading.py, then we use info.frame
        push_variables(self.info.frame, state)

    def pull(self):
        """Pull variables from the interleaver state to the frame globals."""

        state = {k: v for k, v in self.info.frame.f_locals.items() if not k.startswith("__nnsight") and k not in self.frame.f_locals}

        push_variables(self.frame, state)

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

        def do_iteration(iter: int):

            mediator.iteration = iter
            mediator.args = list([mediator.iteration])
            
            self.send(Events.REGISTER, mediator)

        if isinstance(iteration, slice):

            i = iteration.start if iteration.start is not None else self.iteration

            stop = iteration.stop

            while True:

                do_iteration(i)

                if stop is None:
                    if self.all_stop is not None:
                        stop = self.all_stop
                
                    elif self.interleaver.default_all is not None:
                        stop = self.interleaver.default_all

                i += 1

                if stop is not None and i >= stop:
                    break

        elif isinstance(iteration, list):

            iteration.sort()

            for i in iteration:
                do_iteration(i)

        elif isinstance(iteration, int):

            do_iteration(iteration)

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
