from __future__ import annotations

import inspect
import warnings
from collections import defaultdict
from enum import Enum
from functools import wraps
from queue import SimpleQueue
from threading import Thread
from types import FrameType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    Iterator,
)

import _thread

import torch

from .. import CONFIG
from ..util import applyn
from .batching import Batcher
from .tracing.util import get_non_nnsight_frame, push_variables, wrap_exception

if TYPE_CHECKING:
    from .tracing.tracer import Cache, InterleavingTracer, Tracer


class Events(Enum):
    """Enum for different types of events in the interleaving process."""

    VALUE = "value"  # Request for a value
    SWAP = "swap"  # Request for a swap
    END = "end"  # Signal to end the execution
    EXCEPTION = "exception"  # Signal that an exception occurred
    SKIP = "skip"  # Signal that an operation should be skipped
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


NNSIGHT_PREFIX = "__nnsight"


class Interleaver:
    """
    Manages the interleaving of model execution and interventions.

    This class coordinates the flow between the model's forward pass and
    user-defined intervention functions, allowing for inspection and
    modification of intermediate values.

    Attributes:
        mediators (Dict[str, Mediator]): A dictionary of mediator names to mediator objects. Each meidator is responsible for a single invoke, or intervention function.
        tracer (Optional[InterleavingTracer]): The tracer object that created this interleaver. Occationaly useful to know the tracer type for this interleaving.
        batcher (Batcher): The batcher object that manages the slice of inputs associtated with each mediator.
        current (Mediator): The current mediator that is being processed. Must be update before resuming a given mediator.
    """

    def __init__(
        self,
        mediators: List[Mediator] = [],
        tracer: InterleavingTracer = None,
        batcher: Batcher = None,
    ):
        """
        Initialize the interleaver for a new interleaving session.

        Args:
            mediators (List[Mediator]): A list of mediator objects.
            tracer (InterleavingTracer): The tracer object that created this interleaver.
            batcher (Batcher): The batcher object that manages the slice of inputs associtated with each mediator.
        """
        self.initialize(mediators, tracer, batcher)

    def initialize(
        self,
        mediators: List[Mediator],
        tracer: InterleavingTracer,
        batcher: Batcher = None,
    ):

        self.mediators: List[Mediator] = mediators

        self.tracer = tracer
        self.batcher = batcher if batcher is not None else Batcher()

        self.default_all = None

        self.current: Mediator = None

    def cancel(self):
        """Cancel all mediators / intervention threads."""

        for mediator in self.mediators:
            mediator.cancel()

        self.mediators = None
        self.tracer = None
        self.batcher = None
        self.default_all = None

        self.current = None

    def iterate_provider(self, provider: str):
        """
        Update a provider string to include which iteration of the provider is being provided.

        Args:
            provider (str): The provider string to update

        Returns:
            str: The updated provider string

        Example:
            >>> provider = "model.transformer.h[0].input"
            >>> iterate_provider(provider)
            "model.transformer.h[0].input.i0"

            >>> provider = "model.transformer.h[0].input"
            >>> iterate_provider(provider)
            "model.transformer.h[0].input.i1"

            >>> provider = "model.transformer.h[0].input"
            >>> iterate_provider(provider)
            "model.transformer.h[0].input.i2"
        """

        mediator = self.current

        iteration = mediator.iteration_tracker[provider]

        return f"{provider}.i{iteration}"

    def iterate_requester(self, requester: str):
        """
        Update a requester string to include which iteration of the requester is being requested.
        This is determined by the current mediator's iteration attribute, or influced by .iter contexts.

        Args:
            requester (str): The requester string to update

        Returns:
            str: The updated requester string
        """

        mediator = self.current

        # Base case: The mediator knows which iteration it wants, which is by default the first (0) iteration
        iteration = mediator.iteration

        # If the iteration is None, it means its unbounded so just request the next unseen iteration
        if iteration is None:
            iteration = mediator.iteration_tracker[requester]
        # If the iteration is a tuple, it means we want to update the iteration of the meidator with a new value
        elif isinstance(iteration, tuple):
            iteration, mediator.iteration = iteration

        return f"{requester}.i{iteration}"

    def wrap_module(self, module: torch.nn.Module):
        """
        Instruments a PyTorch module to intercept inputs and outputs for interleaving.

        Args:
            module (torch.nn.Module): The module to instrument

        Returns:
            None
        """

        skip = None

        forward = module.forward

        # If wrapping the same module more than once, we need to get the original forward function
        pre_wrapped = hasattr(forward, "__nnsight_original_forward__")
        if pre_wrapped:
            forward.__nnsight_input_handle__.remove()
            forward.__nnsight_output_handle__.remove()
            forward = forward.__nnsight_original_forward__

        # Wrap the module's forward function first to enable skipping of the forward pass.
        @wraps(forward)
        def nnsight_forward(*args, **kwargs):

            nonlocal skip

            if skip is None or not self.interleaving:
                try:
                    return forward(*args, **kwargs)
                finally:
                    skip = None

            return skip

        module.forward = nnsight_forward

        # Hook the module's input to intercept and interleave the input values.
        @torch._dynamo.disable
        def input_hook(module: torch.nn.Module, args, kwargs):

            # If not interleaving, just return the original input values.
            if not self.interleaving:
                return args, kwargs

            # NNsight keeps the modules attribute path as the provider string on the module itself.
            provider = module.__path__

            nonlocal skip

            # Provide the input values to the interleaver to be potentially consumed and/or modified by the mediators.
            # Iterate here means this provided can be provided more than once so the provider string will be updated to include the iteration.
            try:
                inputs = self.handle(f"{provider}.input", (args, kwargs), iterate=True)
            # To skip a module, we raise a SkipException with the value we want to return instead.
            # This is the same skip variable that the skippable forward method has refernce to, so we can set it here and it can be handled by the output hook later.
            except SkipException as e:
                skip = e.value
            # If not skipping, just return the potentially modified input values.
            else:
                args, kwargs = inputs

            return args, kwargs

        # Register the input hook to the module's forward pre-hook.
        input_handle = module.register_forward_pre_hook(
            input_hook, with_kwargs=True, prepend=True
        )

        @torch._dynamo.disable
        def output_hook(module: torch.nn.Module, _, output: Any):

            # If not interleaving, just return the original output values.
            if not self.interleaving:
                return output

            # NNsight keeps the modules attribute path as the provider string on the module itself.
            provider = module.__path__

            nonlocal skip

            # If we are skipping, we set the output to the value we want to return and clear the skip variable.
            if skip is not None:

                output = skip

                skip = None

            # Provide the output values to the interleaver to be potentially consumed and/or modified by the mediators.
            # Iterate here means this provided can be provided more than once so the provider string will be updated to include the iteration.
            output = self.handle(f"{provider}.output", output, iterate=True)

            return output

        # Register the output hook to the module's forward post-hook.
        output_handle = module.register_forward_hook(output_hook, prepend=True)

        # Store the original forward function, input handle, and output handle on the wrapped forward function.
        # This is useful for unwrapping the module later in case of re-wrapping the module.
        nnsight_forward.__nnsight_original_forward__ = forward
        nnsight_forward.__nnsight_input_handle__ = input_handle
        nnsight_forward.__nnsight_output_handle__ = output_handle

    def wrap_operation(self, fn: Callable, name: str, bound_obj: Optional[Any] = None):
        """
        Wrap an operation to intercept inputs and outputs for intervention, as well as the function itself.
        Used by Envoy.source to hook into intermediate operations of a forward pass.

        Args:
            fn (Callable): The intermediate operation function to wrap
            name (str): The fully qualified name of the operation
            bound_obj (Optional[Any]): The object fn is bound to if it is a method

        Returns:
            Callable: A wrapped version of the function
        """

        @wraps(fn)
        def inner(*args, **kwargs):

            nonlocal fn

            # Provide the function itself to the interleaver to allows recursive interventions for Envoy.source.
            fn = self.handle(f"{name}.fn", fn)

            # Provide the input values to the interleaver to be potentially consumed and/or modified by the mediators.
            # Iterate here means this provided can be provided more than once so the provider string will be updated to include the iteration.
            args, kwargs = self.handle(f"{name}.input", (args, kwargs), iterate=True)

            # Call the original function/method with the potentially modified input values.
            if not inspect.ismethod(fn) and bound_obj is not None:
                value = fn(bound_obj, *args, **kwargs)
            else:
                value = fn(*args, **kwargs)

            # Provide the output values to the interleaver to be potentially consumed and/or modified by the mediators.
            # Iterate here means this provided can be provided more than once so the provider string will be updated to include the iteration.
            value = self.handle(f"{name}.output", value, iterate=True)

            return value

        return inner

    @property
    def interleaving(self) -> bool:
        """
        Check if the interleaver is currently interleaving.

        Returns:
            bool: True if the interleaver is interleaving, False otherwise
        """
        return getattr(self, "_interleaving", False)

    def __enter__(self):

        # Set the interleaving flag to True to indicate that the interleaver is currently interleaving.
        # Used by a variety of functioanlities that interact with the interleaver.
        # Often to raise an error when one of these functionalities is called outside interleaving.
        self._interleaving = True

        try:
            # Start all the mediators to begin their intervention threads amd wait for their first event.
            for mediator in self.mediators:
                if mediator.alive:
                    continue
                mediator.start(self)

        except:
            # Clear the interleaving flag on error.
            self._interleaving = False
            raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        # Clear the interleaving flag on exit.
        self._interleaving = False

        # Clear the mediators that are no longer alive.
        self.mediators = [mediator for mediator in self.mediators if mediator.alive]

        # Ignore EarlyStopException errors.
        if exc_type is not None and issubclass(exc_type, EarlyStopException):
            return True

    def check_dangling_mediators(self):

        # If any mediators are still waiting for their values for their events, they probably called an Envoy out of order
        # Or their Envoy was not called.
        for mediator in self.mediators:

            if mediator.alive:
                requested_event, requester = mediator.event_queue.get()

                if isinstance(requester, tuple):
                    requester = requester[0]

                mediator.respond(
                    ValueError(
                        f"Execution complete but `{requester}` was not provided. Did you call an Envoy out of order? Investigate why this module was not called."
                    )
                )

                iteration = mediator.iteration

                if iteration != 0:
                    try:
                        mediator.handle()
                    except ValueError as e:
                        msg = f"Execution complete but `{requester}` was not provided. If this was in an Iterator at iteration {iteration} this iteration did not happen. If you were using `.iter[:]`, this is likely not an error."
                        warnings.warn(msg)
                else:
                    mediator.handle()

        self.mediators = []

    def check_cache_full(self):
        """
        Print a warning if a module to be cached was missed.
        """
        for invoker in self.mediators:
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
            provider (Optional[Any]): The identifier of the provider
            value (Optional[Any]): The value being provided
            iterate (bool): Whether to iterate the provider string

        Returns:
            Any: The original or modified value
        """

        # Store the original provider as mediators might modify it.
        original_provider = provider
        # Store the original current provided value to restore after this handle call.
        # This is due to handle calls being potentially recursive and we need to restore the original value after the recursive calls.
        original_value = self.batcher.current_value

        # Set the current value to the value being provided.
        self.batcher.current_value = value

        skip_count = 0
        skip_values = []

        original_current = self.current

        for mediator in self.mediators:

            self.current = mediator

            if iterate:
                provider = self.iterate_provider(original_provider)
            try:
                mediator.handle(provider)
            except SkipException as e:
                skip_count += 1
                skip_values.append(e.value)

            if iterate and mediator.alive:
                mediator.iteration_tracker[original_provider] += 1

        self.current = original_current

        value = self.batcher.current_value

        # Restore the original current value.
        self.batcher.current_value = original_value

        if skip_count:

            if skip_count == len(self.mediators):

                def _swap(*args):
                    return torch.cat(args, dim=0)

                skip_value = applyn(skip_values, _swap, torch.Tensor)
                raise SkipException(skip_value)
            else:
                raise ValueError(
                    f"A module skip must be applied to all the invokers defined in the tracer!"
                )

        return value

    ### Serialization ###

    def __deepcopy__(self, memo):

        return self


class Mediator:
    """
    Mediates between the model execution and a single intervention function.

    This class handles the communication between the model's forward pass and a
    user-defined intervention function, allowing for inspection and
    modification of intermediate values.

    Attributes:
        interleaver (Interleaver): The interleaver that this mediator is currently running in
        intervention (Callable): The intervention function to mediate
        info (Tracer.Info): Information about the tracing context associated with this mediator
        name (Optional[str]): Optional name for the mediator
        batch_group (Optional[List[int]]): Optional batch group for the mediator to determine which slice of tensors are being intervened on
        event_queue (SimpleQueue): Where the mediator (worker thread) puts events to be processed by the interleaver (main thread). Will only ever have 1 or 0 items in the queue.
        response_queue (SimpleQueue): Where the interleaver (main thread) puts responses to events, to then be processed by the mediator (worker thread). Will only ever have 1 or 0 items in the queue.
        worker (Thread): The thread that runs the intervention function
        history (Set[str]): A set of providers that have been seen by the mediator. Used to detect out of order interventions.
        iteration_tracker (Dict[str, int]): A dictionary tracking the number of times each provider has been seen by the mediator.
        iteration (int): The current iteration this mediator is interventing on
        user_cache (List[Cache]): A list of caches to be used by the mediator
        all_stop (Optional[int]): Optional number of times to execute this mediator
    """

    class OutOfOrderError(Exception):
        """
        Exception raised when interventions are defined out of order.
        """

        pass

    class Value:

        def __init__(self):
            self.value = None
            self.lock = _thread.allocate_lock()
            self.lock.acquire()
            self.has_value = False

        def get(self):

            value = self.value
            self.value = None

            self.has_value = False

            return value

        def wait(self):
            self.lock.acquire()

        def put(self, value: Any):
            self.value = value
            self.has_value = True

            self.lock.release()

        def restore(self, value: Any):
            self.value = value
            self.has_value = True

    def __init__(
        self,
        intervention: Callable,
        info: "Tracer.Info",
        name: Optional[str] = None,
        batch_group: Optional[List[int]] = None,
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

        self.interleaver = None

        self.event_queue = Mediator.Value()
        self.response_queue = Mediator.Value()

        self.worker = None

        self.history = set()
        self.user_cache: List["Cache"] = list()
        self.iteration_tracker = defaultdict(int)
        self.iteration = 0
        self.all_stop: Optional[int] = stop
        self.args = list()
        self.cross_invoker = None

        self.original_globals = {}

        self._prev = None

    @property
    def alive(self):

        return self.worker is not None

    def __enter__(self):

        # Store the previous mediator to be restored after this mediator is done as there might be nested mediators.
        self._prev = self.interleaver.current
        # Set the current mediator that is running to this mediator.
        self.interleaver.current = self

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        # Restore the previous mediator to be running after this mediator is done.
        self.interleaver.current = self._prev

    def start(self, interleaver: Interleaver):
        """
        Start the mediator's intervention thread.

        Args:
            interleaver (Interleaver): The interleaver managing this mediator
        """
        self.interleaver = interleaver

        self.original_globals = self.intervention.__globals__.copy()

        self.cross_invoker = (
            len(self.interleaver.mediators) > 1 and CONFIG.APP.CROSS_INVOKER
        )

        # Start the worker thread.
        self.worker = Thread(
            target=self.intervention,
            args=(self, self.info, *self.args),
            daemon=True,
            name=self.name,
        )

        self.interleaver.current = self
        self.worker.start()
        self.event_queue.wait()

        # Handle the first event for each mediator to clear mediators that already ended.
        try:
            self.handle()
        except EarlyStopException:
            pass

        self.interleaver.current = None

    ### Provider Methods ###

    def cancel(self):
        """Cancel the intervention thread and its ephemeral state."""

        self.history = set()
        self.iteration_tracker = defaultdict(int)
        self.iteration = 0
        self.worker = None

        if self.event_queue.has_value:
            self.handle()
            if self.event_queue.has_value:
                self.event_queue.get()
                self.response_queue.put(Cancelation())
                self.event_queue.get()

    def handle(self, provider: Optional[str] = None):
        """
        Process a provider and its value.
        Depending on which event this mediator is waiting on, it will either:
        - Respond with the value
        - Swap (replace) the value with a new value
        - Respond with an out of order error
        - Skip the value
        - Set a barrier
        - End the execution (cancels the mediator)

        Args:
            provider (Optional[str]): The identifier of the provider

        """
        # Check to see if this mediator has an unprocessed eventto start.
        process = self.event_queue.has_value

        event = None

        # Continue processing events until there are no more events to process.
        # Means we can move on to the next mediator and continue the model execution.
        while process:

            event, data = self.event_queue.get()

            if event == Events.VALUE:
                process = self.handle_value_event(data, provider)
            elif event == Events.SWAP:
                process = self.handle_swap_event(provider, *data)
            elif event == Events.EXCEPTION:
                process = self.handle_exception_event(data)
            elif event == Events.SKIP:
                try:
                    process = self.handle_skip_event(provider, *data)
                except SkipException as e:
                    if self.user_cache:
                        for cache in self.user_cache:
                            cache.add(provider, e.value)
                    raise e
            elif event == Events.BARRIER:
                process = self.handle_barrier_event(provider, data)
            elif event == Events.END:
                process = self.handle_end_event()

        if len(self.user_cache) > 0 and provider is not None:

            for cache in self.user_cache:
                cache.add(
                    provider,
                    self.interleaver.batcher.narrow(self.batch_group),
                )

    def handle_value_event(self, requester: Any, provider: Any) -> bool:
        """
        Handle a value event by providing the requested value or recording a missed provider.

        Args:
            requester (str): The identifier of the requester
            provider (str): The identifier of the provider
        Returns:
            bool: Indicating whether the request was fulfilled by this processor, If so, continue processing events.
        """

        # If fulfilled by this processor, respond with the value and continue processing events.
        if provider == requester:

            # Potentially only select a slice of the value if this mediator is part of a batch group.
            value = self.interleaver.batcher.narrow(self.batch_group)

            self.respond(value)
        else:
            # If the requester has been seen before, respond with an out of order error.
            if requester in self.history:
                self.respond(
                    Mediator.OutOfOrderError(
                        f"Value was missed for {requester}. Did you call an Envoy out of order?"
                    )
                )
            else:
                # If the requester has not been seen before, add it to the history and put the value event back in the event queue to be processed later.
                self.history.add(provider)
                self.event_queue.restore((Events.VALUE, requester))

                return False

        return True

    def handle_swap_event(self, provider: Any, requester: Any, swap_value: Any):
        """
        Handle a swap event by swapping the value if the provider matches the requester.

        Args:
            requester (str): The identifier of the requester
            provider (str): The identifier of the provider
            swap_value (Any): The value to swap in

        Returns:
            bool: Indicating whether the swap was fulfilled by this processor, If so, continue processing events.
        """
        # If fulfilled by this processor, swap the value and respond with the value and continue processing events.
        if provider == requester:
            # Swap the value in the batcher. Might only replace a slice of the value if this mediator is part of a batch group.
            self.interleaver.batcher.swap(self.batch_group, swap_value)

            self.respond()

            return True

        else:
            # If the requester has been seen before, respond with an out of order error.
            if requester in self.history:
                self.respond(
                    ValueError(
                        f"Setting {requester} is out of scope for scope {provider}. Did you call an Envoy out of order?"
                    )
                )
            else:
                # If the requester has not been seen before, add it to the history and put the swap event back in the event queue to be processed later.
                self.history.add(provider)
                self.event_queue.restore((Events.SWAP, (requester, swap_value)))

                return False

        return True

    def handle_exception_event(self, exception: Exception):
        """
        Handle an exception event by raising the exception.

        Args:
            exception (Exception): The exception to raise

        Returns:
            bool: Flag to stop processing events.
        """

        self.cancel()

        # Cancelation is okay
        if not isinstance(exception, Cancelation):

            # because of the defered execution of NNsight, we need to rebuild where the execption was in the original user code instead of this execption.
            exception = wrap_exception(exception, self.info)

            raise exception

        return False

    def handle_barrier_event(self, provider: Any, participants: Set[str]):
        """
        Handle a barrier event by setting a barrier.
        """

        if participants is not None:

            original_current = self.interleaver.current

            for mediator in self.interleaver.mediators:

                if mediator.name in participants:

                    self.interleaver.current = mediator

                    mediator.respond()

                    mediator.handle(provider)

            self.interleaver.current = original_current

        return False

    def handle_end_event(self):
        """
        Handle an end event by stopping the mediator.
        """
        self.cancel()

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
                self.event_queue.restore((Events.SKIP, (requester, value)))

                return False

    def respond(self, value: Optional[Any] = None):
        """
        Respond from the interleaver (main thread) to the mediator (worker thread) the value for a pending event.

        Args:
            value (Optional[Any]): The value to provide
        """

        # Respond and resume the mediator thread.
        self.response_queue.put(value)
        self.event_queue.wait()

    ### Requester Methods ###

    def send(self, event: Events, requester: Any):
        """
        Send an event to interleaver (main thread) from this mediator (worker thread), and wait for it to be processed by the interleaver.

        Args:
            event (Events): The event to send
            requester (Any): The identifier of the requester, plus any additional data for the event.

        Returns:
            Any: The response from the provider
        """

        # In multi invoke scenarios, one invoke might reference variables from another invoke. So we need to push and pull the variables to the shared state to make them available to the other invoke.
        # TODO find a way to only push if there are multiple invokers AND they share the same parent frame
        if self.cross_invoker:
            self.push()

        # Send the event
        self.event_queue.put((event, requester))

        # Wait for the interleaver to process the event and respond with the value.
        self.response_queue.wait()
        response = self.response_queue.get()

        # If the response is an exception, raise it.
        if isinstance(response, Exception):
            raise response

        if self.cross_invoker:
            self.pull()

        return response

    def request(self, requester: str):
        """
        Request a value from a specific provider.

        Args:
            requester (str): The identifier of the provider to request a value from

        Returns:
            Any: The requested value
        """

        return self.send(Events.VALUE, requester)

    def swap(self, requester: str, value: Any):
        """
        Send a swap event to replace the value of a provider.

        Args:
            requester (str): The identifier of the requester
            value (Any): The value to swap in
        """

        self.send(Events.SWAP, (requester, value))

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

    @property
    def frame(self) -> FrameType:
        """
        Get the frame of the intervention function.

        Returns:
            The frame of the intervention function
        """

        frame = get_non_nnsight_frame()

        return frame

    def push(self):
        """Push local variables to the interleaver state."""

        if self.info.frame is None:
            return

        state = {
            k: v
            for k, v in self.frame.f_locals.items()
            if not k.startswith(NNSIGHT_PREFIX)
            and (v is not self.original_globals.get(k, None))
        }

        if isinstance(self.info.frame, FrameType):

            # this does not handle the case of a fn thats called in an invoker. this will push vars directly to where the invoke was called not the fn. really we need to grad the f_back of the <nnsight> frame. If its in threading.py, then we use info.frame
            push_variables(self.info.frame, state)

        else:

            self.info.frame.update(state)

    def pull(self):
        """Pull variables from the interleaver state to the frame globals."""

        if self.info.frame is None:
            return

        state = (
            self.info.frame.f_locals
            if isinstance(self.info.frame, FrameType)
            else self.info.frame
        )

        state = {k: v for k, v in state.items() if not k.startswith(NNSIGHT_PREFIX)}

        for key in {**state}:
            if key in self.frame.f_locals:
                del state[key]
            elif key in self.original_globals:
                state[key] = self.original_globals[key]

        push_variables(self.frame, state)

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
            "all_stop": self.all_stop,
            "iteration_tracker": self.iteration_tracker,
        }

    def __setstate__(self, state):
        """Set the state of the mediator for deserialization."""
        self.name = state["name"]
        self.info = state["info"]
        self.batch_group = state["batch_group"]
        self.intervention = state["intervention"]
        self.all_stop = state["all_stop"]
        self.iteration_tracker = state["iteration_tracker"]
        self.event_queue = Mediator.Value()
        self.response_queue = Mediator.Value()

        self.worker = None
        self.interleaver = None
        self.history = set()
        self.user_cache: "Cache" = list()
        self.iteration = 0
        self.args = list()
        self.original_globals = {}
        self.cross_invoker = None
