"""Dynamic one-shot hook registration for lazy hook execution.

Instead of permanently registering hooks on every module (which incurs overhead
on every forward pass regardless of whether any intervention is active), hooks
are registered **on-demand** by each mediator and **self-remove** after firing.

How this connects to ``eproperty``
----------------------------------

The ``requires_*`` decorators in this file are the **pre-setup glue** between
an :class:`~nnsight.intervention.interleaver.eproperty` stub and the model.
An eproperty stub like::

    @eproperty()
    @requires_output     # ← lives in this file
    def output(self): ...

has an empty body. The ``eproperty`` descriptor invokes that stub on every
``__get__`` (``self._hook(obj)``) purely to fire the decorator's side effect:
register a one-shot PyTorch hook on the underlying module so that the value
the worker is about to ``request()`` will actually be produced. Without a
decorator from this module (or an equivalent provider-side ``handle()``
call elsewhere), the request would block forever.

Module contents
---------------

- :func:`add_ordered_hook` — inserts a PyTorch hook into a module's hook dict
  at the correct position relative to other mediators, preserving mediator
  execution order.
- :func:`input_hook` / :func:`output_hook` — create one-shot hooks for a
  specific mediator.  Each hook tracks iteration count, waits for the correct
  iteration, self-removes, then delegates to ``mediator.handle()``.
- :func:`requires_output` / :func:`requires_input` — decorators for
  ``eproperty`` stub functions (e.g. ``Envoy.output``, ``Envoy.input``) that
  ensure the appropriate one-shot hook is registered before the mediator
  requests a value.  If the value is already being provided (checked via
  ``batcher.current_provider``), hook registration is skipped.
- :func:`requires_operation_output` / :func:`requires_operation_input` —
  operation-level analogues used by :class:`OperationEnvoy` for source
  tracing; they install hooks on the OperationEnvoy's hook lists rather
  than on a PyTorch module.
"""

from functools import partial, wraps
from typing import Any, Callable, List, Optional, TYPE_CHECKING
from .interleaver import Mediator
import torch
from torch.utils.hooks import RemovableHandle
from ..util import apply

if TYPE_CHECKING:
    from nnsight.intervention.envoy import Envoy
    from nnsight.intervention.source import OperationAccessor, OperationEnvoy
else:
    Envoy = Any
    OperationEnvoy = Any
    OperationAccessor = Any


class OperationHookHandle:
    """A ``RemovableHandle``-shaped handle for operation-level hooks.

    Operation hooks live on plain lists (``op_accessor.pre_hooks``,
    ``op_accessor.post_hooks``, ``op_accessor.fn_hooks``) rather than PyTorch's
    module hook dicts, so PyTorch's :class:`~torch.utils.hooks.RemovableHandle`
    doesn't apply. This handle lets callers treat them uniformly with module
    handles: store in ``mediator.hooks`` and call ``.remove()`` at cancel.

    ``remove()`` is idempotent — calling it more than once (e.g. once from
    the hook's self-removal path, once from ``Mediator.remove_hooks``) is
    a no-op after the first successful removal.
    """

    __slots__ = ("_target", "_hook")

    def __init__(self, target: List[Callable], hook: Callable):
        self._target = target
        self._hook = hook

    def remove(self) -> None:
        if self._hook is None:
            return
        try:
            self._target.remove(self._hook)
        except ValueError:
            pass
        self._hook = None
        self._target = None


def add_ordered_hook(module: torch.nn.Module, hook: Callable, type: str) -> Any:
    """Register a hook on a module, inserted in mediator-index order.

    When multiple mediators hook the same module, their hooks must fire in
    mediator order (i.e. the order the invokes were defined).  PyTorch fires
    hooks in dict-insertion order, so this function inserts the new hook at the
    correct position by sorting on the ``mediator_idx`` attribute attached to
    each hook function.

    Args:
        module: The PyTorch module to hook.
        hook: The hook callable.  Must have a ``mediator_idx`` attribute.
        type: ``"input"`` for a forward pre-hook, ``"output"`` for a forward
            hook.

    Returns:
        A :class:`~torch.utils.hooks.RemovableHandle` that can remove the hook.
    """

    if type == "input":

        handle = RemovableHandle(
            module._forward_pre_hooks, extra_dict=module._forward_pre_hooks_with_kwargs
        )

        module._forward_pre_hooks_with_kwargs[handle.id] = True

        hook_dict = module._forward_pre_hooks

    elif type == "output":

        handle = RemovableHandle(
            module._forward_hooks,
            extra_dict=module._forward_hooks_with_kwargs,
        )
        hook_dict = module._forward_hooks

    if len(hook_dict) == 0:
        hook_dict[handle.id] = hook
        return handle

    # Insert the hook into the dict at the position corresponding to its mediator_idx.
    # The dict is kept sorted by .mediator_idx so hooks fire in the correct order.
    hook_mediator_idx = getattr(hook, "mediator_idx", float("-inf"))

    items = list(hook_dict.items())
    inserted = False
    new_items = []
    for k, v in items:
        existing_idx = getattr(v, "mediator_idx", float("-inf"))
        if not inserted and hook_mediator_idx < existing_idx:
            new_items.append((handle.id, hook))
            inserted = True
        new_items.append((k, v))
    if not inserted:
        new_items.append((handle.id, hook))
    hook_dict.clear()
    hook_dict.update(new_items)

    return handle


def input_hook(mediator: Mediator, module: torch.nn.Module, path: str) -> Any:
    """Register a one-shot forward pre-hook for a mediator on a module.

    The hook target iteration is captured at registration time:

    - If ``mediator.iteration`` is set (i.e. the request was issued inside
      an ``tracer.iter[i]`` loop), that value is the target.
    - Otherwise, the current ``iteration_tracker[path]`` value is the
      target — this is the "next" forward pass that hasn't yet fired for
      this module.

    On each forward call, the hook checks whether the tracker has
    advanced to the target iteration.  The tracker is **not** maintained
    by this hook — it is bumped by the persistent iter hooks registered
    by :class:`IteratorTracer` (see :func:`register_iter_hooks` in
    ``iterator.py``) after each forward pass.  This means the hook only
    needs to compare; it doesn't count.

    When the tracker matches:

    1. If ``iteration != 0``, the mediator's ``iteration`` attribute is
       cleared (set to ``None``) so subsequent requests fall back to the
       tracker again — the "I was explicitly in iter[N]" mode ends.
    2. The hook self-removes via ``handle.remove()``.
    3. The hook delegates to :meth:`Mediator.handle` to deliver the
       value to the worker thread (and process any SWAP/SKIP events).

    Args:
        mediator: The mediator requesting this hook.
        module: The PyTorch module to hook.
        path: The provider path prefix (e.g. ``"model.layer.0.input"``).

    Returns:
        A :class:`~torch.utils.hooks.RemovableHandle` for the registered hook.
    """

    handle = None
    iteration = (
        mediator.iteration
        if mediator.iteration is not None
        else mediator.iteration_tracker[path]
    )

    def hook(module: torch.nn.Module, args: Any, kwargs: Any) -> Any:

        # Wait until the iter tracker has advanced to our target step.
        if mediator.iteration_tracker[path] != iteration:
            return args, kwargs

        # Non-zero iterations came from an explicit iter[N] constraint;
        # clearing here returns the mediator to the default "next step"
        # behavior for any subsequent requests in the same intervention.
        if iteration != 0:
            mediator.iteration = None

        nonlocal handle
        handle.remove()

        args, kwargs = mediator.handle(f"{path}.i{iteration}", (args, kwargs))

        return args, kwargs

    hook.mediator_idx = mediator.idx

    handle = add_ordered_hook(module, hook, "input")
    mediator.hooks.append(handle)

    return handle


def output_hook(mediator: Mediator, module: torch.nn.Module, path: str) -> Any:
    """Register a one-shot forward hook for a mediator on a module.

    Behaves identically to :func:`input_hook` but intercepts the module's
    output rather than its input.  See :func:`input_hook` for the full
    details on iteration tracking and the ``mediator.iteration = None``
    reset behavior.

    Args:
        mediator: The mediator requesting this hook.
        module: The PyTorch module to hook.
        path: The provider path prefix (e.g. ``"model.layer.0.output"``).

    Returns:
        A :class:`~torch.utils.hooks.RemovableHandle` for the registered hook.
    """

    handle = None
    iteration = (
        mediator.iteration
        if mediator.iteration is not None
        else mediator.iteration_tracker[path]
    )

    def hook(module: torch.nn.Module, _, output: Any) -> Any:

        if mediator.iteration_tracker[path] != iteration:
            return output

        if iteration != 0:
            mediator.iteration = None

        nonlocal handle
        handle.remove()

        output = mediator.handle(f"{path}.i{iteration}", output)

        return output

    hook.mediator_idx = mediator.idx

    handle = add_ordered_hook(module, hook, "output")
    mediator.hooks.append(handle)

    return handle


def requires_output(fn):
    """Decorator that ensures a one-shot output hook is registered before
    the wrapped eproperty stub runs.

    Used on ``eproperty`` stubs (e.g. ``Envoy.output``) so that the
    appropriate PyTorch hook is installed on the underlying module at the
    moment the user accesses the value during a trace.

    Iteration resolution mirrors :func:`output_hook` — the target iteration
    is ``mediator.iteration`` when set (inside ``tracer.iter[N]``) or
    ``iteration_tracker[path]`` otherwise.

    If ``batcher.current_provider`` already matches the requester for this
    module and iteration, hook registration is skipped — the value is
    already being provided in the current hook's call chain (e.g. when
    ``output`` and another eproperty sharing the same key are accessed
    back-to-back inside the same mediator step).
    """

    @wraps(fn)
    def wrapper(self: Envoy, *args, **kwargs):

        interleaver = self.interleaver

        mediator = interleaver.current

        iteration = (
            mediator.iteration
            if mediator.iteration is not None
            else mediator.iteration_tracker[f"{self.path}.output"]
        )

        requester = f"{self.path}.output.i{iteration}"

        if interleaver.batcher.current_provider != requester:

            output_hook(mediator, self._module, f"{self.path}.output")

        return fn(self, *args, **kwargs)

    return wrapper


def requires_input(fn):
    """Decorator that ensures a one-shot input hook is registered before
    the wrapped eproperty stub runs.

    Used on ``eproperty`` stubs (e.g. ``Envoy.inputs``, ``Envoy.input``,
    ``Envoy.skip``) so that the appropriate PyTorch pre-hook is installed
    on the underlying module when the user accesses (or assigns to) the
    value during a trace.

    See :func:`requires_output` for iteration resolution and the
    ``current_provider`` skip logic.
    """

    @wraps(fn)
    def wrapper(self: Envoy, *args, **kwargs):

        interleaver = self.interleaver

        mediator = interleaver.current

        iteration = (
            mediator.iteration
            if mediator.iteration is not None
            else mediator.iteration_tracker[f"{self.path}.input"]
        )

        requester = f"{self.path}.input.i{iteration}"

        if interleaver.batcher.current_provider != requester:

            input_hook(mediator, self._module, f"{self.path}.input")

        return fn(self, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Persistent cache hooks
# ---------------------------------------------------------------------------


def cache_output_hook(
    cache, module: torch.nn.Module, path: str, batcher, mediator
) -> RemovableHandle:
    """Register a persistent output hook that records values into a Cache.

    Unlike one-shot intervention hooks, cache hooks are **not** self-removing.
    They fire on every forward pass and append values to the cache.  They are
    assigned ``mediator_idx = float('inf')`` so they always fire **after** any
    intervention hooks, ensuring the cache captures post-intervention values.

    Args:
        cache: The :class:`Cache` object to record values into.
        module: The PyTorch module to hook.
        path: The module's envoy path (e.g. ``"model.transformer.h.0"``).
        batcher: The :class:`Batcher` instance for narrowing batched values.
        mediator: The owning mediator; ``mediator.batch_group`` is read live
            on each forward pass so decode-step position updates are reflected.

    Returns:
        A :class:`~torch.utils.hooks.RemovableHandle` for the registered hook.
    """

    def hook(module: torch.nn.Module, input: Any, output: Any) -> None:
        batch_group = mediator.batch_group
        # Skip entirely when the owning request is not scheduled in this
        # forward pass — otherwise the hook would narrow with stale positions
        # or store another request's tokens in this cache.
        if batch_group is None or batch_group[0] == -1:
            return
        value = output
        if batcher.needs_batching:
            value = apply(value, partial(batcher._narrow, batch_group), torch.Tensor)
        cache.add(path, "output", value)

    hook.mediator_idx = float("inf")

    handle = add_ordered_hook(module, hook, "output")
    mediator.hooks.append(handle)
    return handle


def cache_input_hook(
    cache, module: torch.nn.Module, path: str, batcher, mediator
) -> RemovableHandle:
    """Register a persistent input hook that records values into a Cache.

    Behaves like :func:`cache_output_hook` but intercepts the module's input
    (as ``(args, kwargs)``).

    Args:
        cache: The :class:`Cache` object to record values into.
        module: The PyTorch module to hook.
        path: The module's envoy path (e.g. ``"model.transformer.h.0"``).
        batcher: The :class:`Batcher` instance for narrowing batched values.
        mediator: The owning mediator; ``mediator.batch_group`` is read live
            on each forward pass so decode-step position updates are reflected.

    Returns:
        A :class:`~torch.utils.hooks.RemovableHandle` for the registered hook.
    """

    def hook(module: torch.nn.Module, args: Any, kwargs: Any) -> None:
        batch_group = mediator.batch_group
        if batch_group is None or batch_group[0] == -1:
            return
        value = (args, kwargs)
        if batcher.needs_batching:
            value = apply(value, partial(batcher._narrow, batch_group), torch.Tensor)
        cache.add(path, "inputs", value)

    hook.mediator_idx = float("inf")

    handle = add_ordered_hook(module, hook, "input")
    mediator.hooks.append(handle)
    return handle


# ---------------------------------------------------------------------------
# Operation eproperty decorators
# ---------------------------------------------------------------------------


def requires_operation_output(fn):
    """Decorator for OperationEnvoy eproperty stubs that need an output hook.

    Equivalent to :func:`requires_output` but registers an operation-level
    hook via :func:`operation_output_hook` on the OperationEnvoy's
    underlying :class:`OperationAccessor`'s ``post_hooks`` list.
    """

    @wraps(fn)
    def wrapper(self: OperationEnvoy, *args, **kwargs):
        mediator = self.interleaver.current
        iteration = (
            mediator.iteration
            if mediator.iteration is not None
            else mediator.iteration_tracker[f"{self.path}.output"]
        )
        requester = f"{self.path}.output.i{iteration}"

        if self.interleaver.batcher.current_provider != requester:
            operation_output_hook(mediator, self.accessor)

        return fn(self, *args, **kwargs)

    return wrapper


def requires_operation_input(fn):
    """Decorator for OperationEnvoy eproperty stubs that need an input hook.

    Equivalent to :func:`requires_input` but registers an operation-level
    hook via :func:`operation_input_hook` on the OperationEnvoy's
    underlying :class:`OperationAccessor`'s ``pre_hooks`` list.
    """

    @wraps(fn)
    def wrapper(self: OperationEnvoy, *args, **kwargs):
        mediator = self.interleaver.current
        iteration = (
            mediator.iteration
            if mediator.iteration is not None
            else mediator.iteration_tracker[f"{self.path}.input"]
        )
        requester = f"{self.path}.input.i{iteration}"

        if self.interleaver.batcher.current_provider != requester:
            operation_input_hook(mediator, self.accessor)

        return fn(self, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Operation hooks (for source tracing)
# ---------------------------------------------------------------------------


def operation_output_hook(mediator: Mediator, op_accessor: OperationAccessor):
    """Register a one-shot output hook on an :class:`OperationAccessor`.

    Appends a hook to ``op_accessor.post_hooks`` and returns an
    :class:`OperationHookHandle` which is also tracked on
    ``mediator.hooks`` for unified cleanup in :meth:`Mediator.remove_hooks`.
    When the operation's wrapper (created by :func:`wrap_operation`) runs,
    it iterates ``post_hooks`` and calls each with the operation's output
    value.

    The iteration-matching protocol mirrors the module-level
    :func:`output_hook`: target iteration is captured at registration
    from either ``mediator.iteration`` or ``iteration_tracker[path]``,
    and the hook only fires once the tracker advances to match.  After
    firing with a non-zero target, ``mediator.iteration`` is cleared.

    The tracker for operation paths is bumped by the same persistent
    iter hooks registered by :class:`IteratorTracer` — those hooks fire
    on the parent module's output and increment the tracker for all
    provider paths under that module, including operation-level ones.

    Args:
        mediator: The mediator requesting the value.
        op_accessor: The :class:`OperationAccessor` to hook.

    Returns:
        An :class:`OperationHookHandle` whose ``.remove()`` pops the hook
        from ``op_accessor.post_hooks``.
    """
    path = f"{op_accessor.path}.output"
    iteration = (
        mediator.iteration
        if mediator.iteration is not None
        else mediator.iteration_tracker[path]
    )

    handle = None

    def hook(value: Any) -> Any:
        if mediator.iteration_tracker[path] != iteration:
            return value

        if iteration != 0:
            mediator.iteration = None

        handle.remove()
        return mediator.handle(f"{path}.i{iteration}", value)

    op_accessor.post_hooks.append(hook)
    handle = OperationHookHandle(op_accessor.post_hooks, hook)
    mediator.hooks.append(handle)
    return handle


def operation_input_hook(mediator: Mediator, op_accessor: OperationAccessor):
    """Register a one-shot input hook on an :class:`OperationAccessor`.

    Like :func:`operation_output_hook` but appended to
    ``op_accessor.pre_hooks``.  The wrapper calls pre-hooks with the
    operation's ``(args, kwargs)`` tuple before invoking the function.

    Args:
        mediator: The mediator requesting the value.
        op_accessor: The :class:`OperationAccessor` to hook.

    Returns:
        An :class:`OperationHookHandle` whose ``.remove()`` pops the hook
        from ``op_accessor.pre_hooks``.
    """
    path = f"{op_accessor.path}.input"
    iteration = (
        mediator.iteration
        if mediator.iteration is not None
        else mediator.iteration_tracker[path]
    )

    handle = None

    def hook(inputs: Any) -> Any:
        if mediator.iteration_tracker[path] != iteration:
            return inputs

        if iteration != 0:
            mediator.iteration = None

        handle.remove()
        return mediator.handle(f"{path}.i{iteration}", inputs)

    op_accessor.pre_hooks.append(hook)
    handle = OperationHookHandle(op_accessor.pre_hooks, hook)
    mediator.hooks.append(handle)
    return handle


def operation_fn_hook(mediator: Mediator, op_accessor: OperationAccessor):
    """Register a one-shot fn hook for recursive source tracing.

    Appended to ``op_accessor.fn_hooks``.  Unlike input/output hooks,
    fn hooks are **not** iteration-aware — they fire on the first call
    to the operation wrapper after registration, deliver the function to
    the worker thread (which injects it with nested ``wrap`` calls), and
    receive the injected replacement via a SWAP event processed inside
    the same :meth:`Mediator.handle` call.  The hook returns the injected
    function so the wrapper uses it for the actual invocation.

    After firing, the injected function is installed (one-shot) as
    ``op_accessor.fn_replacement`` by :attr:`OperationEnvoy.source` and
    cleared again by :func:`wrap_operation` once it executes — re-accessing
    ``.source`` on an OperationEnvoy reinstalls it from the cached nested
    accessor.

    Args:
        mediator: The mediator requesting the function.
        op_accessor: The :class:`OperationAccessor` to hook.

    Returns:
        An :class:`OperationHookHandle` whose ``.remove()`` pops the hook
        from ``op_accessor.fn_hooks``.
    """

    handle = None

    def hook(fn: Callable) -> Callable:
        handle.remove()
        return mediator.handle(f"{op_accessor.path}.fn", fn)

    op_accessor.fn_hooks.append(hook)
    handle = OperationHookHandle(op_accessor.fn_hooks, hook)
    mediator.hooks.append(handle)
    return handle
