from __future__ import annotations

import inspect
import types
import warnings
import weakref
from collections import defaultdict
from enum import Enum
from functools import partial, wraps
from threading import Thread
from types import FrameType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Set,
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

        self._interleaving = False
        self.hook_handles = []

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

        self.mediators = []
        self.tracer = None
        self.batcher = None
        self.default_all = None
        self.transform = None

        self.current = None

    def __del__(self):
        """Remove all hooks when the interleaver is garbage collected."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def iterate_provider(self, provider: str):
        """
        Update a provider string to include which iteration of the provider is being provided.

        Args:
            provider (str): The provider string to update

        Returns:
            str: The updated provider string

        Examples:
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
        """Append the current mediator's iteration index to a requester string.

        The iteration is determined by ``mediator.iteration``, which defaults
        to 0 and is updated by :class:`IteratorTracer` when iterating over
        generation steps (e.g. ``tracer.iter[:]``).

        Args:
            requester: The base requester string (e.g. ``"model.layer.0.output"``).

        Returns:
            The requester with iteration suffix (e.g. ``"model.layer.0.output.i0"``).
        """

        mediator = self.current

        iteration = mediator.iteration

        return f"{requester}.i{iteration}"

    def wrap_module(self, module: torch.nn.Module):
        """Prepare a module for lazy hook execution.

        Unlike previous versions that registered permanent input/output hooks
        on every module, this method only installs:

        1. A **skippable forward wrapper** — replaces ``module.forward`` with
           a thin wrapper that checks for ``__nnsight_skip__`` in kwargs.  If
           present, the module's original forward is bypassed and the skip
           value is returned directly.
        2. A **sentinel output hook** — an empty ``register_forward_hook``
           that returns ``output`` unchanged.  This is required because
           PyTorch's ``Module.__call__`` fast-paths when *no* hooks are
           registered: if a module has zero hooks at call time, dynamically
           added hooks during the forward pass will never fire.  The sentinel
           ensures PyTorch always goes through the hook dispatch path so that
           one-shot hooks registered by :func:`hooks.input_hook` and
           :func:`hooks.output_hook` can be picked up mid-forward.

        Actual interception of inputs/outputs is handled lazily by one-shot
        hooks registered on-demand by each mediator (see ``hooks.py``).

        Args:
            module: The PyTorch module to prepare.
        """

        # Check if already wrapped with skippable forward
        pre_wrapped = hasattr(module, "__nnsight_forward__")

        if not pre_wrapped:

            instance_forward = module.forward
            if hasattr(instance_forward, "__self__"):
                # Bound method — unbind to avoid reference cycle:
                # module -> forward -> __self__ -> module
                original_forward = instance_forward.__func__
            elif isinstance(instance_forward, partial):
                # e.g. accelerate's partial(new_forward, module) for device_map —
                # unwrap to avoid reference cycle through partial.args
                original_forward = instance_forward.func
            else:
                original_forward = instance_forward

            module.__nnsight_forward__ = original_forward

            module_ref = weakref.ref(module)

            @wraps(original_forward)
            def nnsight_forward(*args, **kwargs):
                m = module_ref()
                if "__nnsight_skip__" in kwargs:
                    return kwargs.pop("__nnsight_skip__")
                return m.__nnsight_forward__(m, *args, **kwargs)

            module.forward = nnsight_forward

            # Sentinel hook — keeps PyTorch in the hook dispatch path so
            # dynamically registered one-shot hooks fire correctly.
            module.register_forward_hook(lambda _, __, output: output)

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

            for mediator in self.mediators:
                if f"{name}.fn" in mediator.history:
                    mediator.history.remove(f"{name}.fn")

            return value

        return inner

    @property
    def interleaving(self) -> bool:
        """
        Check if the interleaver is currently interleaving.

        Returns:
            bool: True if the interleaver is interleaving, False otherwise
        """
        return self._interleaving

    def __enter__(self):

        # Set the interleaving flag to True to indicate that the interleaver is currently interleaving.
        # Used by a variety of functioanlities that interact with the interleaver.
        # Often to raise an error when one of these functionalities is called outside interleaving.
        self._interleaving = True

        try:
            # Start all the mediators to begin their intervention threads amd wait for their first event.
            for idx, mediator in enumerate(self.mediators):
                mediator.idx = idx
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

    def handle(self, provider: Optional[str] = None, value: Optional[Any] = None):
        """Broadcast a provider value to all mediators.

        This is a simplified handle used only by :meth:`wrap_operation` for
        source-level interventions, where every mediator needs to see the
        value.  Regular module hooks bypass this entirely — they register
        one-shot hooks that call :meth:`Mediator.handle` directly.

        Args:
            provider: The provider string identifying this value.
            value: The value being provided.
        """
        for mediator in self.mediators:
            mediator.handle(provider, value)

    def check_dangling_mediators(self):

        # If any mediators are still waiting for their values for their events, they probably called an Envoy out of order
        # Or their Envoy was not called.
        for mediator in self.mediators:

            if mediator.alive:
                requested_event, requester = mediator.event_queue.get()

                if isinstance(requester, tuple):
                    requester = requester[0]

                iteration = mediator.iteration

                mediator.respond(
                    ValueError(
                        f"Execution complete but `{requester}` was not provided. Did you call an Envoy out of order? Investigate why this module was not called."
                    )
                )

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
            self.restore(value)

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
        self.idx = None
        self.info = info
        self.batch_group = batch_group

        self.interleaver = None

        self.event_queue = Mediator.Value()
        self.response_queue = Mediator.Value()

        self.worker = None

        self.skip_container = None

        self.history = set()
        self.user_cache: List["Cache"] = list()
        self.iteration_tracker = defaultdict(int)
        self.iteration = 0
        self.all_stop: Optional[int] = stop
        self.args = list()
        self.cross_invoker = None

        self.original_globals = {}

        self._prev = None

        self.transform = None

        self.lock = 0

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

        self.transform = None

        # Only copy globals that the intervention code actually references.
        all_globals = self.intervention.__globals__
        co_names = self.intervention.__code__.co_names
        self.original_globals = {
            k: all_globals[k] for k in co_names if k in all_globals
        }

        self.cross_invoker = (
            len(self.interleaver.mediators) > 1 and CONFIG.APP.CROSS_INVOKER
        )

        # Capture the current CUDA stream so the worker thread uses it.
        # Worker threads default to the NULL stream (stream 0), but
        # vLLM (and other frameworks) run on a non-default stream.
        # PyTorch creates non-default streams with cudaStreamNonBlocking,
        # which disables implicit synchronization with the NULL stream.
        # Without propagating the stream, worker-thread CUDA ops (clone,
        # fill) race with main-thread ops on the compute stream.
        if torch.cuda.is_available():
            _caller_stream = torch.cuda.current_stream()
        else:
            _caller_stream = None

        _intervention = self.intervention
        _args = (self, self.info, *self.args)

        def _worker_target():
            if _caller_stream is not None:
                torch.cuda.set_stream(_caller_stream)
            _intervention(*_args)

        # Start the worker thread.
        self.worker = Thread(
            target=_worker_target,
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

    def handle(self, provider: Optional[str] = None, value: Optional[Any] = None):
        """Process a provided value against this mediator's pending event.

        Called directly by one-shot hooks (see ``hooks.py``) or by
        :meth:`Interleaver.handle` for source operations.  This method
        saves and restores the interleaver's ``current`` mediator and the
        batcher's ``current_value`` / ``current_provider``, so it is safe
        to call from within any hook without corrupting shared state.

        Depending on the pending event, this will:
        - **VALUE**: Deliver the value to the worker thread.
        - **SWAP**: Replace the value in the batcher.
        - **SKIP**: Inject ``__nnsight_skip__`` into kwargs so
          ``nnsight_forward`` bypasses the module.
        - **BARRIER**: Synchronize participating mediators.
        - **END**: Cancel the mediator (intervention finished).
        - **EXCEPTION**: Re-raise the exception from the worker thread.

        Args:
            provider: The provider string (e.g. ``"model.layer.0.output.i0"``).
            value: The value being provided (e.g. the module's output tensor).

        Returns:
            The (potentially modified) value from ``batcher.current_value``.
        """

        prev_current = self.interleaver.current
        prev_value = self.interleaver.batcher.current_value
        prev_provider = self.interleaver.batcher.current_provider

        self.interleaver.current = self
        self.interleaver.batcher.current_value = value
        self.interleaver.batcher.current_provider = provider

        # Check to see if this mediator has an unprocessed eventto start.
        process = self.event_queue.has_value

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

        value = self.interleaver.batcher.current_value

        self.interleaver.current = prev_current
        self.interleaver.batcher.current_value = prev_value
        self.interleaver.batcher.current_provider = prev_provider

        return value

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

            if self.transform:
                value = self.transform(value)

                self.interleaver.batcher.swap(self.batch_group, value)

                self.transform = None
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
            
            prev_current = self.interleaver.current

            for mediator in self.interleaver.mediators:

                if mediator.name in participants:
                    
                    self.interleaver.current = mediator

                    mediator.respond()

                    mediator.handle(provider, self.interleaver.batcher.current_value)
                    
            self.interleaver.current = prev_current

        return False

    def handle_end_event(self):
        """
        Handle an end event by stopping the mediator.
        """
        self.cancel()

        return False

    def handle_skip_event(self, provider: Any, requester: Any, value: Any):
        """Handle a skip event by injecting the replacement value into kwargs.

        Instead of raising a ``SkipException`` (as in the old permanent-hook
        approach), the replacement value is placed into ``kwargs`` under the
        ``__nnsight_skip__`` key.  The ``nnsight_forward`` wrapper installed
        by :meth:`Interleaver.wrap_module` checks for this key and returns
        the value directly, bypassing the module's original forward method.

        This approach works with the one-shot hook system because the input
        hook has access to ``(args, kwargs)`` via the batcher and can
        modify kwargs in-place before the forward call.
        """

        if provider == requester:

            _, kwargs = self.interleaver.batcher.current_value

            kwargs["__nnsight_skip__"] = value

            self.respond()

            return True

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

            self.info.frame.f_locals.update(state)

    def pull(self):
        """Pull variables from the interleaver state to the frame globals."""

        if self.info.frame is None:
            return

        state = self.info.frame.f_locals

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
