"""Dynamic one-shot hook registration for lazy hook execution.

Instead of permanently registering hooks on every module (which incurs overhead
on every forward pass regardless of whether any intervention is active), hooks
are registered **on-demand** by each mediator and **self-remove** after firing.

This module provides:

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
"""

from functools import partial, wraps
from typing import Any, Callable, List, Optional, TYPE_CHECKING
from .interleaver import Mediator
import torch
from torch.utils.hooks import RemovableHandle
from ..util import apply

if TYPE_CHECKING:
    from nnsight.intervention.envoy import Envoy
else:
    Envoy = Any


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

    The hook captures the mediator's current ``iteration`` at registration time.
    On each forward call it increments ``iteration_tracker[path]``.  If the
    tracker hasn't reached the target iteration yet, the hook returns early
    (no-op).  Once the correct iteration fires, the hook **self-removes** via
    ``handle.remove()`` and delegates to ``mediator.handle()`` to deliver or
    modify the input value.

    For iteration 0 the hook always fires on the first call (the
    ``iteration != 0`` guard lets it through).  For higher iterations, the hook
    stays registered across forward passes until the tracker matches.

    Each mediator maintains its own ``iteration_tracker``, so hooks from
    different mediators on the same module count independently.

    Args:
        mediator: The mediator requesting this hook.
        module: The PyTorch module to hook.
        path: The provider path prefix (e.g. ``"model.layer.0.input"``).

    Returns:
        A :class:`~torch.utils.hooks.RemovableHandle` for the registered hook.
    """

    handle = None
    iteration = mediator.iteration

    def hook(module: torch.nn.Module, args: Any, kwargs: Any) -> Any:

        mediator.iteration_tracker[path] += 1

        nonlocal iteration

        if iteration != 0 and mediator.iteration_tracker[path] - 1 != iteration:
            return args, kwargs

        nonlocal handle
        handle.remove()

        args, kwargs = mediator.handle(f"{path}.i{iteration}", (args, kwargs))

        return args, kwargs

    hook.mediator_idx = mediator.idx

    handle = add_ordered_hook(module, hook, "input")

    return handle


def output_hook(mediator: Mediator, module: torch.nn.Module, path: str) -> Any:
    """Register a one-shot forward hook for a mediator on a module.

    Behaves identically to :func:`input_hook` but intercepts the module's
    output rather than its input.  See :func:`input_hook` for details on
    iteration tracking and self-removal.

    Args:
        mediator: The mediator requesting this hook.
        module: The PyTorch module to hook.
        path: The provider path prefix (e.g. ``"model.layer.0.output"``).

    Returns:
        A :class:`~torch.utils.hooks.RemovableHandle` for the registered hook.
    """

    handle = None
    iteration = mediator.iteration

    def hook(module: torch.nn.Module, _, output: Any) -> Any:

        mediator.iteration_tracker[path] += 1

        nonlocal iteration

        if iteration != 0 and mediator.iteration_tracker[path] - 1 != iteration:
            return output

        nonlocal handle
        handle.remove()

        output = mediator.handle(f"{path}.i{iteration}", output)

        return output

    hook.mediator_idx = mediator.idx

    handle = add_ordered_hook(module, hook, "output")

    return handle


def requires_output(fn):
    """Decorator that ensures an output hook is registered before the wrapped function runs.

    Used on ``eproperty`` stub functions (e.g. ``Envoy.output``) to lazily
    register a one-shot output hook when a mediator requests the value.

    If ``batcher.current_provider`` already matches the requester string for
    this module and iteration, the hook is skipped — the value is already being
    provided (e.g. when ``output`` and another eproperty sharing the same key
    are accessed back-to-back).
    """

    @wraps(fn)
    def wrapper(self: Envoy, *args, **kwargs):

        interleaver = self._interleaver

        mediator = interleaver.current

        requester = f"{self.path}.output.i{mediator.iteration}"

        if interleaver.batcher.current_provider != requester:

            output_hook(mediator, self._module, f"{self.path}.output")

        return fn(self, *args, **kwargs)

    return wrapper


def requires_input(fn):
    """Decorator that ensures an input hook is registered before the wrapped function runs.

    Used on ``eproperty`` stub functions (e.g. ``Envoy.inputs``, ``Envoy.input``,
    ``Envoy.skip``) to lazily register a one-shot input hook when a mediator
    requests or modifies the value.

    See :func:`requires_output` for the ``current_provider`` skip logic.
    """

    @wraps(fn)
    def wrapper(self: Envoy, *args, **kwargs):

        interleaver = self._interleaver

        mediator = interleaver.current

        requester = f"{self.path}.input.i{mediator.iteration}"

        if interleaver.batcher.current_provider != requester:

            input_hook(mediator, self._module, f"{self.path}.input")

        return fn(self, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Persistent cache hooks
# ---------------------------------------------------------------------------


def cache_output_hook(cache, module: torch.nn.Module, path: str, batcher, batch_group) -> RemovableHandle:
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
        batch_group: The mediator's batch group for narrowing, or ``None``.

    Returns:
        A :class:`~torch.utils.hooks.RemovableHandle` for the registered hook.
    """

    def hook(module: torch.nn.Module, input: Any, output: Any) -> None:
        value = output
        if batcher.needs_batching and batch_group is not None and batch_group[0] != -1:
            value = apply(value, partial(batcher._narrow, batch_group), torch.Tensor)
        cache.add(path, "output", value)

    hook.mediator_idx = float("inf")

    handle = add_ordered_hook(module, hook, "output")
    cache.hook_handles.append(handle)
    return handle


def cache_input_hook(cache, module: torch.nn.Module, path: str, batcher, batch_group) -> RemovableHandle:
    """Register a persistent input hook that records values into a Cache.

    Behaves like :func:`cache_output_hook` but intercepts the module's input
    (as ``(args, kwargs)``).

    Args:
        cache: The :class:`Cache` object to record values into.
        module: The PyTorch module to hook.
        path: The module's envoy path (e.g. ``"model.transformer.h.0"``).
        batcher: The :class:`Batcher` instance for narrowing batched values.
        batch_group: The mediator's batch group for narrowing, or ``None``.

    Returns:
        A :class:`~torch.utils.hooks.RemovableHandle` for the registered hook.
    """

    def hook(module: torch.nn.Module, args: Any, kwargs: Any) -> None:
        value = (args, kwargs)
        if batcher.needs_batching and batch_group is not None and batch_group[0] != -1:
            value = apply(value, partial(batcher._narrow, batch_group), torch.Tensor)
        cache.add(path, "inputs", value)

    hook.mediator_idx = float("inf")

    handle = add_ordered_hook(module, hook, "input")
    cache.hook_handles.append(handle)
    return handle


# ---------------------------------------------------------------------------
# Operation hooks (for source tracing)
# ---------------------------------------------------------------------------


def operation_output_hook(mediator, op_envoy):
    """Register a one-shot output hook on an OperationEnvoy.

    The hook is appended to ``op_envoy.post_hooks``.  When the operation's
    wrapper fires, it iterates post hooks and calls each one with the output
    value.  The hook tracks iterations, waits for the correct one,
    self-removes, then delegates to ``mediator.handle()``.

    Args:
        mediator: The mediator requesting the value.
        op_envoy: The :class:`OperationEnvoy` to hook.
    """
    path = f"{op_envoy.name}.output"
    iteration = mediator.iteration

    def hook(value):
        mediator.iteration_tracker[path] += 1

        if iteration != 0 and mediator.iteration_tracker[path] - 1 != iteration:
            return value

        op_envoy.post_hooks.remove(hook)
        return mediator.handle(f"{path}.i{iteration}", value)

    op_envoy.post_hooks.append(hook)


def operation_input_hook(mediator, op_envoy):
    """Register a one-shot input hook on an OperationEnvoy.

    Like :func:`operation_output_hook` but appended to ``op_envoy.pre_hooks``.
    The hook receives ``(args, kwargs)`` and returns the (potentially modified) tuple.

    Args:
        mediator: The mediator requesting the value.
        op_envoy: The :class:`OperationEnvoy` to hook.
    """
    path = f"{op_envoy.name}.input"
    iteration = mediator.iteration

    def hook(inputs):
        mediator.iteration_tracker[path] += 1

        if iteration != 0 and mediator.iteration_tracker[path] - 1 != iteration:
            return inputs

        op_envoy.pre_hooks.remove(hook)
        return mediator.handle(f"{path}.i{iteration}", inputs)

    op_envoy.pre_hooks.append(hook)


def operation_fn_hook(mediator, op_envoy):
    """Register a one-shot fn hook for recursive source tracing.

    Appended to ``op_envoy.fn_hooks``.  When the operation wrapper fires, it
    passes the original function through each fn-hook.  The hook calls
    ``mediator.handle()`` to deliver the function to the worker thread (which
    injects it) and receives the injected replacement via a SWAP event
    processed in the same handle call.

    Args:
        mediator: The mediator requesting the function.
        op_envoy: The :class:`OperationEnvoy` to hook.
    """

    def hook(fn):
        op_envoy.fn_hooks.remove(hook)
        return mediator.handle(f"{op_envoy.name}.fn", fn)

    op_envoy.fn_hooks.append(hook)
