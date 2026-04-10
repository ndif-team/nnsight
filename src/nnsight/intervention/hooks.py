from functools import wraps
from typing import Any, Callable, TYPE_CHECKING
from .interleaver import Mediator
import torch
from torch.utils.hooks import RemovableHandle

if TYPE_CHECKING:
    from nnsight.intervention.envoy import Envoy
else:
    Envoy = Any


def add_ordered_hook(module: torch.nn.Module, hook: Callable, type: str) -> Any:

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

    # Insert the hook into the dict at the position corresponding to its mediator_idx

    hook_mediator_idx = getattr(hook, "mediator_idx", float("-inf"))

    # Find where to insert (the dict is already sorted by .mediator_idx)
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
        # insert at end
        new_items.append((handle.id, hook))
    hook_dict.clear()
    hook_dict.update(new_items)

    return handle


def input_hook(mediator: Mediator, module: torch.nn.Module, path: str) -> Any:

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

    @wraps(fn)
    def wrapper(self: Envoy, *args, **kwargs):

        interleaver = self._interleaver

        mediator = interleaver.current

        requester = f"{self.path}.output.i{mediator.iteration}"

        print(
            interleaver.batcher.current_provider != requester,
            requester,
            interleaver.batcher.current_provider,
        )

        if interleaver.batcher.current_provider != requester:

            output_hook(mediator, self._module, f"{self.path}.output")

        return fn(self, *args, **kwargs)

    return wrapper


def requires_input(fn):

    @wraps(fn)
    def wrapper(self: Envoy, *args, **kwargs):

        interleaver = self._interleaver

        mediator = interleaver.current

        requester = f"{self.path}.input.i{mediator.iteration}"

        if interleaver.batcher.current_provider != requester:

            input_hook(mediator, self._module, f"{self.path}.input")

        return fn(self, *args, **kwargs)

    return wrapper
