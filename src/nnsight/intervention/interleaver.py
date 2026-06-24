from __future__ import annotations

import inspect
import types
import warnings
import weakref
from collections import defaultdict
from enum import Enum
from functools import partial, wraps
from types import FrameType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Optional,
    Protocol,
    Set,
    Union,
    runtime_checkable,
)

from greenlet import greenlet

import torch

from .. import CONFIG
from ..util import applyn
from functools import partial
from .batching import Batcher
from .tracing.util import get_non_nnsight_frame, push_variables, wrap_exception

if TYPE_CHECKING:
    from .tracing.tracer import Cache, InterleavingTracer, Tracer


from typing import TypeVar

T = TypeVar("T")


@runtime_checkable
class IEnvoy(Protocol):
    """Interface for objects that participate in the interleaving system.

    Any object that uses :class:`eproperty` descriptors must satisfy this
    protocol by providing:

    Attributes:
        interleaver: The :class:`Interleaver` managing execution flow.
        path: Optional provider path prefix used to build requester/provider
            strings (e.g. ``"model.transformer.h.0"``).  May be ``None`` or
            empty — :meth:`eproperty._build_requester` falls back to the
            eproperty key alone in that case.  This is how tracer-level
            eproperties such as :attr:`InterleavingTracer.result` work
            without a path prefix.

    Notes:
        Implementors that have no meaningful path (e.g. tracers) do **not**
        need to declare a ``path`` attribute — :meth:`eproperty._build_requester`
        uses ``getattr(obj, "path", "")`` so a missing attribute is treated
        the same as ``None`` / ``""``. The attribute is declared
        ``Optional[str]`` here for type clarity.
    """

    interleaver: "Interleaver"
    path: Optional[str]


class eproperty(property):
    """A descriptor for defining hookable properties on :class:`IEnvoy` objects.

    ``eproperty`` exposes values through the interleaving request/swap
    mechanism. During a trace, reading an ``eproperty`` issues a blocking
    request to the interleaver; writing to it schedules a swap.

    The decorated stub
    ------------------

    Every ``eproperty`` is defined by decorating a *stub method*. The body
    of that stub is **never executed for its return value** — it is a
    placeholder whose only jobs are:

    1. Donate its ``__name__`` and ``__doc__`` to the descriptor (the name
       becomes the default ``key``; the docstring is what users see in
       ``help(model.transformer.h[0].output)``).
    2. Carry the **decorators stacked on top of it** that perform the real
       work — registering the PyTorch hook (or operation hook) that will
       eventually deliver the value the user is about to ``request()``.

    Concretely::

        @eproperty()
        @requires_output       # ← does the work: registers a one-shot
        def output(self): ...  #   forward hook on self._module before
                               #   the request blocks the worker thread.

    On every ``__get__`` the descriptor calls ``self._hook(obj)`` (the
    decorated stub). Because the decorator wraps the empty stub, that
    invocation runs the decorator's pre-setup — typically registering the
    appropriate hook so the value will arrive when the model executes —
    and then calls the (no-op) stub. The descriptor then issues the
    actual ``request(requester)`` call, which blocks until the hook fires.

    The pre-setup decorators live in :mod:`nnsight.intervention.hooks`:

    - :func:`requires_output` / :func:`requires_input` — module-level
      one-shot forward / pre-forward hooks (used by ``Envoy``).
    - :func:`requires_operation_output` / :func:`requires_operation_input`
      — operation-level hooks for ``.source`` tracing (used by
      ``OperationEnvoy``).
    - Custom backends (e.g. vLLM) supply their own decorators in the same
      pattern; the contract is "make sure a provider for this requester
      string will fire before ``request()`` blocks".

    A bare ``eproperty`` with no pre-setup decorator is also valid for
    things that are provided externally — e.g. ``InterleavingTracer.result``
    is fed by ``Envoy.interleave`` calling ``self.interleaver.handle("result", ...)``,
    so no per-access hook setup is needed.

    Path / key resolution
    ---------------------

    The interleaver is obtained from ``obj.interleaver``. The path prefix
    is obtained from ``obj.path`` if the attribute exists and is truthy;
    if absent or empty, the key alone is used as the requester string.
    This is how tracer-level eproperties like ``InterleavingTracer.result``
    work without a path prefix.

    Supported implementors
    ----------------------

    Any class satisfying the :class:`IEnvoy` protocol can host
    eproperties.  In this codebase that includes:

    - :class:`Envoy` — module-level ``.output``, ``.input``, ``.inputs``
    - :class:`OperationEnvoy` — operation-level ``.output``, ``.input``,
      ``.inputs`` (source tracing)
    - :class:`InterleavingTracer` — tracer-level ``.result``
    - vLLM :class:`VLLM` — ``.logits``, ``.samples``

    Args:
        key: The interleaving key appended to ``obj.path``
            (``<path>.<key>``).  Defaults to the stub function's name.
            Multiple eproperties can share a key (e.g. ``Envoy.input``
            and ``Envoy.inputs`` both use ``"input"``) to provide
            different views on the same underlying value.
        description: A short label shown in the repr tree.  Only eproperties
            with a description appear in the tree.
        iterate: Whether to append an iteration suffix (``.i0``, ``.i1``, …).
            Defaults to ``True``.
    """

    def __init__(self, key: str = None, description: str = None, iterate: bool = True):
        super().__init__()

        self.name: str = None
        self.key = key
        self.description = description
        self.iterate = iterate

        self._hook: Callable = None
        self._postprocess: Optional[Callable] = None
        self._preprocess: Optional[Callable] = None
        self._transform: Optional[Callable] = None

    def __call__(self, hook: Callable[..., T]) -> "T | eproperty":
        """Register the decorated stub.

        ``hook`` is the user's stub method (e.g. ``def output(self): ...``)
        with any pre-setup decorators from :mod:`nnsight.intervention.hooks`
        already applied. The body is treated as a no-op; what matters is
        what the decorators do when ``hook(obj)`` is invoked from
        :meth:`__get__` — typically registering a one-shot PyTorch hook so
        the value will be produced by the time the request blocks. The
        stub's ``__name__`` becomes the default ``key``.
        """
        self.name = hook.__name__
        self._hook = hook
        if self.key is None:
            self.key = self.name
        return self

    def postprocess(self, func: Callable) -> "eproperty":
        """Register a post-processing function called on ``__set__``.

        Runs on the user-supplied value just before it is swapped into the
        running model. Used by :class:`Envoy.input` to repack a single value
        back into the ``(args, kwargs)`` shape the model's hook expects.
        """
        self._postprocess = func
        return self

    def preprocess(self, func: Callable) -> "eproperty":
        """Register a pre-processing function called on ``__get__``.

        Runs on the raw value pulled from the interleaver before it is
        returned to the user. Used by :class:`Envoy.input` to extract the
        first positional argument from ``(args, kwargs)``.

        When a corresponding :meth:`transform` is also registered, the value
        returned by ``preprocess`` is captured by the transform's closure —
        in-place mutations the user makes are visible inside the transform.
        """
        self._preprocess = func
        return self

    def transform(self, func: Callable) -> "eproperty":
        """Register a one-shot ``__get__`` -> swap-back transform.

        ``transform`` complements :meth:`preprocess`. When ``preprocess``
        returns a *new* object (a clone, a reshape, a view onto a slice),
        in-place edits the user makes to that object are invisible to the
        running model — the model still holds the original value. ``transform``
        closes that loop: at request time the preprocessed value is bound into
        the callable via ``functools.partial`` and parked on the current
        mediator; once the user is done with their edits and the worker yields
        control, the mediator invokes the transform and ``batcher.swap``s the
        return value back into the model.

        The function signature is ``transform() -> Any`` (no args — the value
        is captured by the closure). Whatever the transform returns replaces
        the original model-side value for the rest of the forward pass.

        Use cases
        ---------

        - **Safe mutable view**: ``preprocess`` returns ``value.clone()`` so
          users can ``thing[:] = 0`` without aliasing surprises; the transform
          returns the (mutated) clone so the model still sees their edits.
        - **Per-head attention access**: ``preprocess`` reshapes
          ``[B, S, H]`` into ``[B, n_heads, S, head_dim]``; the transform
          reshapes back to ``[B, S, H]`` so the model continues with the
          user-edited heads.

        Example::

            class MyEnvoy(Envoy):
                @eproperty(key="output")
                @requires_output
                def heads(self): ...

                @heads.preprocess
                def heads(self, value):
                    # Expose attention heads as a separate dim.
                    B, S, H = value.shape
                    return value.view(B, S, self.n_heads, H // self.n_heads)\
                                .transpose(1, 2)

                @heads.transform
                @staticmethod
                def heads(value):
                    # Reshape back to the model's [B, S, H] layout.
                    return value.transpose(1, 2).reshape(value.shape[0], value.shape[2], -1)

        Notes
        -----

        - Transform is **one-shot per access** — it fires once when the value
          event for that access is processed and is then cleared.
        - The transform can be a plain function (commonly decorated with
          ``@staticmethod``) since the preprocessed value already carries the
          context via the closure.
        - If you only want to view the value and don't intend to swap it
          back, omit ``transform`` and just use ``preprocess``.
        """
        self._transform = func
        return self

    def _build_requester(self, obj: IEnvoy) -> str:
        path = getattr(obj, "path", "")
        return f"{path}.{self.key}" if path else self.key

    def __get__(self, obj: IEnvoy, owner: Any) -> Any:

        if obj is None:
            return self

        interleaver = obj.interleaver

        if interleaver.interleaving:

            requester = self._build_requester(obj)

            # Run the decorated stub. We don't care about its return value —
            # what matters is the side effect of any pre-setup decorators
            # stacked on it (see hooks.py: `requires_output`, `requires_input`,
            # operation variants). Those decorators register the one-shot
            # PyTorch hook that will eventually deliver the value to the
            # `request()` call below.
            self._hook(obj)

            if self.iterate:
                requester = interleaver.iterate_requester(requester)

            value = interleaver.current.request(requester)

            if self._preprocess is not None:
                value = self._preprocess(obj, value)

            if self._transform is not None:
                # Bind the preprocessed value into the transform NOW (at request
                # time) rather than passing it at fire time. The partial holds
                # a reference to the same object the user is about to receive,
                # so any in-place mutations are visible when the mediator later
                # invokes `self.transform()` and swaps the result back into the
                # model. See :meth:`Mediator.handle_value_event`.
                interleaver.current.transform = partial(self._transform, value)

        else:
            label = self._build_requester(obj)
            raise ValueError(f"Cannot access `{label}` outside of interleaving.")

        return value

    def __set__(self, obj: IEnvoy, value: Any):

        if self._postprocess is not None:
            value = self._postprocess(obj, value)

        interleaver = obj.interleaver

        if interleaver.interleaving:

            requester = self._build_requester(obj)

            self._hook(obj)

            if self.iterate:
                requester = interleaver.iterate_requester(requester)

            interleaver.current.swap(requester, value)

        else:
            label = self._build_requester(obj)
            raise ValueError(f"Cannot set `{label}` outside of interleaving.")

    def provide(self, obj: IEnvoy, value: Any) -> Any:
        """Provide a value from the model side into the interleaving system."""
        requester = self._build_requester(obj)
        return obj.interleaver.handle(
            requester,
            value,
            iterate=self.iterate,
        )


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


def _store_deferred_exception(mediator: Any, exception: Exception) -> None:
    """Capture a deferred exception + metadata on the mediator.

    Populates ``deferred_exception`` alongside three serialization-friendly
    fields read by ``intervention.errors.capture_deferred``:

    - ``_deferred_type_name``: original exception class name, captured BEFORE
      any ``wrap_exception`` call that would replace the class with a dynamic
      ``NNsightException`` subclass.
    - ``_deferred_traceback``: formatted traceback string of the original
      exception, for debugging across the server boundary where the user
      cannot inspect the live frames.
    - ``_deferred_is_control_flow``: True for ``EarlyStopException`` (raised
      by ``tracer.stop()``), so server paths can filter intentional control
      flow out of error responses without name compares.
    """
    import traceback as _tb

    mediator._deferred_type_name = type(exception).__name__
    mediator._deferred_traceback = "".join(
        _tb.format_exception(type(exception), exception, exception.__traceback__)
    )
    mediator._deferred_is_control_flow = isinstance(exception, EarlyStopException)
    mediator.deferred_exception = exception


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

        # Set by the vLLM model runner around ``execute_model`` so that
        # exceptions raised inside the worker are stored on each
        # mediator instead of bubbling up and killing the engine.
        self.defer_exceptions = False

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
        """Cancel all mediators / intervention threads.

        After each mediator's worker thread is torn down, every hook it
        registered (module one-shot, cache, operation, gradient, iter
        tracker) is removed via :meth:`Mediator.remove_hooks`. This is
        the single cleanup path for all dynamic hooks registered during
        the session.
        """

        for mediator in self.mediators:
            mediator.cancel()
            mediator.remove_hooks()

        self.mediators = []
        self.tracer = None
        self.batcher = None
        self.default_all = None
        self.transform = None

        self.current = None

    def iterate_requester(self, requester: str):
        """Append the current mediator's iteration index to a requester string.

        The iteration is determined by one of two sources:

        - If ``mediator.iteration`` is set (user is inside an explicit
          ``tracer.iter[i]`` loop), use that value directly.  This is how
          iterator tracers constrain requests to a specific generation step.
        - If ``mediator.iteration`` is ``None`` (user is not in an iter
          context, or a one-shot hook has just cleared it after matching),
          fall back to ``mediator.iteration_tracker[requester]``.  The
          tracker is maintained by persistent hooks registered by
          :class:`IteratorTracer` (see :func:`register_iter_hooks`), which
          increment it after every forward pass for each module path.

        This dual-mode behavior lets the same requester syntax work both
        inside and outside an iter loop.

        Args:
            requester: The base requester string (e.g. ``"model.layer.0.output"``).

        Returns:
            The requester with iteration suffix (e.g. ``"model.layer.0.output.i0"``).
        """

        mediator = self.current

        iteration = (
            mediator.iteration
            if mediator.iteration is not None
            else mediator.iteration_tracker[requester]
        )

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
                    entries = kwargs.pop("__nnsight_skip__")
                    if len(entries) == 1:
                        return entries[0][1]
                    # Multi-invoke skip: each mediator's hook contributed its
                    # narrow value. Sort by batch start and concat along dim 0
                    # so the splice matches the model's expected batch order.
                    entries.sort(key=lambda e: e[0][0] if e[0] is not None else -1)
                    values = [v for _, v in entries]
                    return applyn(values, lambda *t: torch.cat(t, dim=0), torch.Tensor)
                source_accessor = getattr(m, "__source_accessor__", None)

                # Once a SourceAccessor exists for this module (built on the
                # first ``.source`` access by anyone), route through it so the
                # injected forward fires the per-op ``wrap`` lookups. The
                # injected forward has its own fast path — ``wrap`` returns
                # ``fn`` unchanged for unhooked operations — so the per-call
                # cost is just a dict lookup + bool check per call site.
                #
                # We deliberately do *not* gate on ``source_accessor.hooked``
                # here: hooks may be registered mid-forward (e.g. an op-level
                # hook registered after the worker resumes from an upstream
                # module hook), and an entry-time check would have already
                # taken the un-injected path by then.
                if source_accessor is not None:
                    return source_accessor(m, *args, **kwargs)
                return m.__nnsight_forward__(m, *args, **kwargs)

            module.forward = nnsight_forward

            # Sentinel hook — keeps PyTorch in the hook dispatch path so
            # dynamically registered one-shot hooks fire correctly.
            module.register_forward_hook(lambda _, __, output: output)

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

        # Swallow internal control-flow exceptions so they don't escape the
        # ``with self.interleaver:`` block in server worker paths and kill
        # the engine. ``EarlyStopException`` is ``tracer.stop()`` control
        # flow; ``Cancelation`` is internal mediator bookkeeping (raised
        # when ``mediator.cancel()`` is called during cleanup).
        if exc_type is not None and issubclass(
            exc_type, (EarlyStopException, Cancelation)
        ):
            return True

    def handle(
        self,
        provider: Optional[str] = None,
        value: Optional[Any] = None,
        iterate: bool = False,
    ):
        """Broadcast a provider value to all mediators.

        Used by :meth:`eproperty.provide` and :meth:`Envoy.interleave` to
        push values (e.g. vLLM logits, generation results) into the
        interleaving system.  Unlike module hooks, these values are not
        produced by the normal PyTorch forward pass — they are pushed
        from the model runner side, so this method acts as a fan-out to
        every mediator and bumps the per-mediator iteration counter for
        the provider path when ``iterate=True``.

        When ``iterate=True``, the per-mediator ``iteration_tracker`` for
        this provider path is bumped after each mediator processes the
        value.  This mirrors the behavior of the persistent iter hooks
        registered by :class:`IteratorTracer`, but for values that flow
        through ``provide()`` instead of a PyTorch forward hook.

        Args:
            provider: The provider string identifying this value.
            value: The value being provided.
            iterate: Whether to append an iteration suffix to the provider
                and bump the per-mediator tracker.

        Returns:
            The (potentially modified) value after the final mediator has
            handled it.  Used by ``operation_fn_hook`` for recursive
            source tracing, where the injected function is returned from
            :meth:`Mediator.handle` via a SWAP event and flows back
            through this broadcast.
        """
        original_provider = provider

        result = value

        for mediator in self.mediators:
            if iterate:
                iteration = mediator.iteration_tracker[original_provider]
                provider = f"{original_provider}.i{iteration}"

            result = mediator.handle(provider, value)

            if iterate:
                mediator.iteration_tracker[original_provider] += 1

        return result

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
                    Mediator.MissedProviderError(
                        f"Execution complete but `{requester}` was not provided. Did you call an Envoy out of order? Investigate why this module was not called."
                    )
                )

                if iteration != 0:
                    try:
                        mediator.handle()
                    except Mediator.MissedProviderError as e:
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
        event_queue (Pending): Single-slot holder where the worker greenlet's most recent event is parked for the interleaver (main greenlet) to process. Holds 1 or 0 events.
        worker (greenlet): The greenlet that runs the intervention function. Switches back to ``_resumer`` whenever it needs a value.
        _resumer (greenlet): The greenlet that most recently resumed the worker; refreshed on every resume so nested interleaving unwinds correctly.
        history (Set[str]): A set of providers that have been seen by the mediator. Used to detect out of order interventions.
        iteration_tracker (Dict[str, int]): Per-provider-path counter maintained by
            :func:`IteratorTracer.register_iter_hooks` (and by
            :meth:`Interleaver.handle` for provided values).  One-shot
            intervention hooks read this to know which generation step
            is currently firing.  Defaults to 0 for any path that has
            not yet been observed.
        iteration (Optional[int]): The target iteration this mediator is
            currently constrained to, or ``None``.  Set by
            :class:`IteratorTracer` before each yield so subsequent
            requests target the correct step.  Cleared back to ``None``
            by a one-shot hook after it matches a non-zero target, so
            later requests in the same intervention fall back to the
            tracker-based "current step" resolution.
        user_cache (List[Cache]): A list of caches to be used by the mediator
        all_stop (Optional[int]): Optional number of times to execute this mediator
    """

    class MissedProviderError(Exception):
        """
        Exception raised when a provider is missed.
        """

        pass

    class OutOfOrderError(MissedProviderError):
        """
        Exception raised when interventions are defined out of order.
        """

        pass

    class Pending:
        """A single-slot holder for the worker greenlet's most recent event.

        The greenlet ``switch`` itself hands over the thread of control and
        provides all the synchronization, so — unlike the old thread-based
        handoff — this needs no lock. It is just somewhere to park the event
        the worker emitted until :meth:`Mediator.handle` consumes it (or
        restores it, when the matching provider has not fired yet).
        """

        def __init__(self):
            self.value = None
            self.has_value = False

        def get(self):

            value = self.value
            self.value = None

            self.has_value = False

            return value

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

        # The worker's pending event, parked here by the main side after each
        # ``switch`` until ``handle`` processes it. No response queue is needed:
        # the response is simply the return value of switching back into the
        # worker (see :meth:`respond` / :meth:`send`).
        self.event_queue = Mediator.Pending()

        # The worker greenlet, plus the greenlet it should switch back to the
        # next time it needs a value. ``_resumer`` is refreshed on every resume
        # (see :meth:`_resume`) rather than fixed once, so a worker that
        # triggers a *sibling* mediator's hooks — e.g. by calling an attached
        # module that is itself hooked — returns control to the exact point
        # that resumed it. This is what makes nested interleaving work.
        self.worker: Optional[greenlet] = None
        self._resumer: Optional[greenlet] = None

        self.skip_container = None

        self.history = set()
        self.user_cache: List["Cache"] = list()
        self.hooks: List[Any] = list()
        self.iteration_tracker = defaultdict(int)
        self.iteration = 0
        # The live iter-loop handle list while this mediator is iterating
        # (set by IteratorTracer); lets a `.source` first touched mid-loop
        # register its op counters into the loop. None when not iterating.
        self._active_iter_handles: Optional[List] = None
        self.all_stop: Optional[int] = stop
        self.args = list()
        self.cross_invoker = None

        self.original_globals = {}

        # Set by ``handle_exception_event`` when the interleaver is in
        # defer mode (vLLM); collected by the model runner from each
        # mediator and shipped back to the client as
        # ``saves["__nnsight_exceptions__"][base_id]``.
        self.deferred_exception = None
        # Metadata captured at deferral time for server-side serialization.
        # See ``_store_deferred_exception`` and ``intervention.errors``.
        self._deferred_type_name: Optional[str] = None
        self._deferred_traceback: Optional[str] = None
        self._deferred_is_control_flow: bool = False

        self._prev = None

        # One-shot transform callback for the next value event. Set by
        # :meth:`eproperty.__get__` when an eproperty with a registered
        # ``transform`` is accessed (already bound to the preprocessed value
        # via ``functools.partial``); consumed and cleared by
        # :meth:`handle_value_event` after the value is delivered to the
        # worker.
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
        Start the mediator's intervention greenlet.

        The intervention runs as a greenlet on *this* OS thread rather than a
        separate worker thread. Control ping-pongs between whoever drives the
        model and the worker via ``switch`` (see :meth:`_resume`): only one is
        ever running, so there is no real concurrency, no locks, and no
        CUDA-stream divergence to repair — the worker shares the caller's
        thread and therefore its current stream.

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

        _intervention = self.intervention
        _args = (self, self.info, *self.args)

        def _worker_target(*_):
            # On the first switch, greenlet passes the switch value as an
            # argument to the run function; it carries no meaning here.
            _intervention(*_args)

        self.worker = greenlet(run=_worker_target)

        self.interleaver.current = self

        # Start the worker and park the first event it emits. The switch
        # return value is whatever the worker hands back via ``send`` / ``end``
        # / ``exception``.
        self.event_queue.restore(self._resume())

        # Handle the first event for each mediator to clear mediators that already ended.
        try:
            self.handle()
        except EarlyStopException:
            pass

        self.interleaver.current = None

    ### Provider Methods ###

    def cancel(self):
        """Cancel the intervention greenlet and its ephemeral state."""

        self.history = set()
        self.iteration_tracker = defaultdict(int)
        self.iteration = 0

        # If the worker is still suspended waiting on a provider that never
        # fired, resume it with a Cancelation so it unwinds its stack; the
        # terminal event it emits on the way out is discarded.
        if self.event_queue.has_value:
            self.handle()
            if self.event_queue.has_value:
                self.event_queue.get()
                self._resume(Cancelation())

        self.worker = None

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
                process = self.handle_skip_event(provider, *data)
            elif event == Events.BARRIER:
                process = self.handle_barrier_event(provider, data)
            elif event == Events.END:
                process = self.handle_end_event()

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

            # If the eproperty had a `transform` callback registered, fire it
            # now and swap the result back into the model. The preprocessed
            # value was bound into `self.transform` via `partial` at request
            # time (see `eproperty.__get__`), so any in-place mutations the
            # worker made between `respond` and now are visible inside the
            # transform's closure. Cleared after firing — transforms are
            # one-shot per value access.
            if self.transform:
                value = self.transform()

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

            # In vLLM mode, defer the exception so the engine stays
            # alive.  The mediator is already cancelled (above), so
            # subsequent hooks will skip it.  Other mediators keep
            # running and the model runner ships this exception back to
            # the client alongside any saves that were already collected.
            #
            # Capture metadata BEFORE ``wrap_exception`` rewrites the type
            # and traceback — the dynamic ``NNsightException`` subclass and
            # rebuilt traceback are useful for the local-raise path but
            # lose information across the server boundary, where the
            # client only sees what we serialize.  ``_store_deferred_exception``
            # also records ``deferred_exception`` itself, so the assignment
            # below the wrap_exception call replaces it with the wrapped
            # form for any downstream consumer that wants the rich type.
            if self.interleaver.defer_exceptions:
                _store_deferred_exception(self, exception)

            # because of the defered execution of NNsight, we need to rebuild where the execption was in the original user code instead of this execption.
            exception = wrap_exception(exception, self.info)

            if self.interleaver.defer_exceptions:
                self.cancel()
                self.deferred_exception = exception
                return False

            raise exception

        return False

    def handle_barrier_event(self, provider: Any, participants: Set[str]):
        """
        Handle a barrier event by setting a barrier.

        Propagates each participant's nested handle return value back
        into ``batcher.current_value``.  Without this, a SWAP fired in
        a participant's body during the barrier walk produces a new
        tensor (concat path) that ``Mediator.handle``'s ``prev_value``
        restore immediately discards — making cross-invoke transfers
        of swapped values silently no-op.  The nested handle's return
        value is the post-restore value (captured before the restore
        runs in :meth:`Mediator.handle`), so re-assigning it here
        carries the swap forward to the outer handle context.
        """

        if participants is not None:

            prev_current = self.interleaver.current

            for mediator in self.interleaver.mediators:

                if mediator.name in participants:

                    self.interleaver.current = mediator

                    mediator.respond()

                    result = mediator.handle(
                        provider, self.interleaver.batcher.current_value
                    )

                    self.interleaver.batcher.current_value = result

            self.interleaver.current = prev_current

        return False

    def handle_end_event(self):
        """
        Handle an end event by stopping the mediator.
        """
        self.cancel()

        return False

    def handle_skip_event(self, provider: Any, requester: Any, value: Any):
        """Handle a skip event by appending the replacement value into kwargs.

        Instead of raising a ``SkipException`` (as in the old permanent-hook
        approach), each mediator's replacement value is appended to a list
        under ``kwargs["__nnsight_skip__"]`` along with that mediator's
        ``batch_group``.  The ``nnsight_forward`` wrapper installed by
        :meth:`Interleaver.wrap_module` consumes the list, sorts by batch
        start, and concats along dim 0 so multi-invoke skips produce a
        full-batch tensor that downstream modules can consume.

        This approach works with the one-shot hook system because the input
        hook has access to ``(args, kwargs)`` via the batcher and can
        modify kwargs in-place before the forward call.
        """

        if provider == requester:

            _, kwargs = self.interleaver.batcher.current_value

            entries = kwargs.setdefault("__nnsight_skip__", [])
            entries.append((self.batch_group, value))

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

        # Resume the worker with the response and park whatever event it emits
        # next. The switch return value IS the worker's next event (or its END
        # / EXCEPTION signal) — there is no separate response queue.
        self.event_queue.restore(self._resume(value))

    def _resume(self, value: Optional[Any] = None) -> Any:
        """Switch into the worker, returning the next event it emits.

        Records the calling greenlet as ``_resumer`` first, so the worker
        switches *back here* (via :meth:`send` / :meth:`end` / :meth:`exception`)
        rather than to a fixed greenlet. That is what lets a worker resume a
        sibling mediator mid-body and have control unwind correctly.
        """

        self._resumer = greenlet.getcurrent()

        return self.worker.switch(value)

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

        # Hand control back to whoever resumed us with this event. The switch
        # returns once the interleaver responds — with the value, or with an
        # exception to raise here in the worker.
        response = self._resumer.switch((event, requester))

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

        # Hand control back for good; the worker greenlet is done.
        self._resumer.switch((Events.END, None))

    def exception(self, exception: Exception):
        """
        Signal that an exception occurred during intervention.

        Args:
            exception: The exception that occurred
        """
        self._resumer.switch((Events.EXCEPTION, exception))

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

    def remove_hooks(self):
        """Remove every hook registered on behalf of this mediator.

        Drains ``self.hooks`` — the single list that tracks module one-shot
        hooks, cache hooks, operation hooks, gradient hooks, and iter-tracker
        hooks. ``.remove()`` is idempotent on every handle type used here, so
        calling this is safe even if some hooks have already self-removed
        after firing.
        """
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    ### Serialization ###

    def __getstate__(self):
        """Get the state of the mediator for serialization."""

        return {
            "name": self.name,
            "idx": self.idx,
            "info": self.info,
            "batch_group": self.batch_group,
            "intervention": self.intervention,
            "all_stop": self.all_stop,
            "iteration_tracker": self.iteration_tracker,
        }

    def __setstate__(self, state):
        """Set the state of the mediator for deserialization."""
        self.name = state["name"]
        self.idx = state["idx"]
        self.info = state["info"]
        self.batch_group = state["batch_group"]
        self.intervention = state["intervention"]
        self.all_stop = state["all_stop"]
        self.iteration_tracker = state["iteration_tracker"]
        self.event_queue = Mediator.Pending()

        self.worker = None
        self._resumer = None
        self.interleaver = None
        self.history = set()
        self.user_cache: "Cache" = list()
        self.hooks: List[Any] = list()
        self.iteration = 0
        self._active_iter_handles: Optional[List] = None
        self.args = list()
        self.original_globals = {}
        self.cross_invoker = None
        self.deferred_exception = None
        self._deferred_type_name = None
        self._deferred_traceback = None
        self._deferred_is_control_flow = False
