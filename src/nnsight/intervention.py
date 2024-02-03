"""This module contains logic to interleave a computation graph (an intervention graph) with the computation graph of a model.

The :class:`InterventionProxy <nnsight.intervention.InterventionProxy>` class extends the functionality of a base nnsight.fx.Proxy.Proxy object and makes it easier for users to interact with.

:func:`intervene() <nnsight.intervention.intervene>` is the entry hook into the models computation graph in order to interleave an intervention graph.

The :class:`HookModel <nnsight.intervention.HookModel>` provides a context manager for adding input and output hooks to modules and removing them upon context exit.
"""

from __future__ import annotations

import inspect
from contextlib import AbstractContextManager
from typing import Any, Callable, Collection, List, Tuple, Union

import torch
from torch.utils.hooks import RemovableHandle

from . import util
from .tracing.Graph import Graph
from .tracing.Node import Node
from .tracing.Proxy import Proxy


class InterventionProxy(Proxy):
    """Sub-class for Proxy that adds additional user functionality to proxies.

    Examples:

        Saving a proxy so it is not deleted at the completion of it's listeners is enabled with ``.save()``:

        .. code-block:: python

            with runner.invoke('The Eiffel Tower is in the city of') as invoker:
                hidden_states = model.lm_head.input.save()
                logits = model.lm_head.output.save()

            print(hidden_states.value)
            print(logits.value)

        This works and would output the inputs and outputs to the model.lm_head module.
        Had you not called .save(), calling .value would have been None.

        Indexing by token of hidden states can easily done using ``.token[<idx>]`` or ``.t[<idx>]``

        .. code-block:: python

            with runner.invoke('The Eiffel Tower is in the city of') as invoker:
                logits = model.lm_head.output.t[0].save()

            print(logits.value)

        This would save only the first token of the output for this module.
        This should be used when using multiple invokes as the batching and padding of multiple inputs could mean the indices for tokens shifts around and this take care of that.

        Calling ``.shape`` on an InterventionProxy returns the shape or collection of shapes for the tensors traced through this module.

        Calling ``.value`` on an InterventionProxy returns the actual populated values, updated during actual execution of the model.

    """

    def __init__(self, node: Node) -> None:
        super().__init__(node)

        self._grad: InterventionProxy = None

    def save(self) -> InterventionProxy:
        """Method when called, indicates to the intervention graph to not delete the tensor values of the result.

        Returns:
            InterventionProxy: Save proxy.
        """

        # Add a 'null' node with the save proxy as an argument to ensure the values are never deleted.
        # This is because 'null' nodes never actually get set and therefore there will always be a
        # dependency for the save proxy.
        self.node.graph.add(
            value=None,
            target="null",
            args=[self.node],
        )

        return self

    @property
    def grad(self) -> InterventionProxy:
        """
        Calling denotes the user wishes to get the grad of proxy tensor and therefore we create a Proxy of that request.
        Only generates a proxy the first time it is references otherwise return the already set one.

        Returns:
            Proxy: Grad proxy.
        """
        if self._grad is None:
            self._grad = self.node.graph.add(
                value=self.node.proxy_value, target="grad", args=[self.node]
            )

        return self._grad

    @grad.setter
    def grad(self, value: Union[InterventionProxy, Any]) -> None:
        """
        Calling denotes the user wishes to set the grad of this proxy tensor and therefore we create a Proxy of that request.

        Args:
            value (Union[InterventionProxy, Any]): Value to set output to.
        """

        self.node.graph.add(target="swp", args=[self.grad.node, value], value=True)

        self._grad = None

    @property
    def shape(self) -> Union[torch.Size, Collection[torch.Size]]:
        """Property to retrieve the shape of the traced proxy value.

        Returns:
            Union[torch.Size,Collection[torch.Size]]: Proxy value shape or collection of shapes.
        """
        return util.apply(self.node.proxy_value, lambda x: x.shape, torch.Tensor)

    @property
    def value(self) -> Any:
        """Property to return the value of this proxy's node.

        Returns:
            Any: The stored value of the proxy, populated during execution of the model.
        """

        if not self.node.done():
            raise ValueError("Accessing Proxy value before it's been set.")

        return self.node.value


def concat(
    activations: Any,
    value: Any,
    batch_start: int,
    batch_size: int,
    total_batch_size: int,
):
    def _concat(values):
        if isinstance(values[0], torch.Tensor):
            orig_size = values[-1]
            new_size = sum([value.shape[0] for value in values[:-1]])
            if new_size == orig_size:
                return torch.concatenate(values[:-1])
            return values[0]
        elif isinstance(values[0], list):
            return [
                _concat([value[value_idx] for value in values])
                for value_idx in range(len(values[0]))
            ]
        elif isinstance(values[0], tuple):
            return tuple(
                [
                    _concat([value[value_idx] for value in values])
                    for value_idx in range(len(values[0]))
                ]
            )
        elif isinstance(values[0], dict):
            return {
                key: _concat([value[key] for value in values])
                for key in values[0].keys()
            }
        return values[0]

    def narrow1(acts: torch.Tensor):
        if total_batch_size == acts.shape[0]:
            return acts.narrow(0, 0, batch_start)

        return acts

    pre = util.apply(activations, narrow1, torch.Tensor)

    post_batch_start = batch_start + batch_size

    def narrow2(acts: torch.Tensor):
        if total_batch_size == acts.shape[0]:
            return acts.narrow(0, post_batch_start, acts.shape[0] - post_batch_start)

        return acts

    post = util.apply(
        activations,
        narrow2,
        torch.Tensor,
    )

    orig_sizes = util.apply(activations, lambda x: x.shape[0], torch.Tensor)

    return _concat([pre, value, post, orig_sizes])


def intervene(
    activations: Any, module_path: str, graph: Graph, key: str, total_batch_size: int
):

    module_path = f"{module_path}.{key}"

    if module_path in graph.argument_node_names:
        argument_node_names = graph.argument_node_names[module_path]

        for argument_node_name in argument_node_names:
            node = graph.nodes[argument_node_name]

            _, batch_size, batch_start, call_iter = node.args

            narrowed = False

            def narrow(acts: torch.Tensor):

                nonlocal narrowed

                if batch_size != total_batch_size and total_batch_size == acts.shape[0]:
                    narrowed = True
                    return acts.narrow(0, batch_start, batch_size)

                return acts

            value = util.apply(
                activations,
                narrow,
                torch.Tensor,
            )

            node.set_value(value)

            value = graph.get_swap(value)

            if narrowed:

                activations = concat(
                    activations, value, batch_start, batch_size, total_batch_size
                )

            else:

                activations = value

    return activations


class HookModel(AbstractContextManager):
    """Context manager that applies input and/or output hooks to modules in a model.

    Registers provided hooks on __enter__ and removes them on __exit__.

    Attributes:
        model (torch.nn.Module): Root model to access modules and apply hooks to.
        modules (List[Tuple[torch.nn.Module, str]]): List of modules to apply hooks to along with their module_path.
        input_hook (Callable): Function to apply to inputs of designated modules.
            Should have signature of [inputs(Any), module_path(str)] -> inputs(Any)
        output_hook (Callable): Function to apply to outputs of designated modules.
            Should have signature of [outputs(Any), module_path(str)] -> outputs(Any)
        handles (List[RemovableHandle]): Handles returned from registering hooks as to be used when removing hooks on __exit__.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        module_keys: List[str],
        input_hook: Callable = None,
        output_hook: Callable = None,
    ) -> None:
        self.model = model
        self.module_keys = module_keys

        self.input_hook = input_hook
        self.output_hook = output_hook

        self.handles: List[RemovableHandle] = []

    def __enter__(self) -> HookModel:
        """Registers input and output hooks to modules if they are defined.

        Returns:
            HookModel: HookModel object.
        """

        for module_key in self.module_keys:
            *module_atoms, hook_type = module_key.split(".")[:-1]
            module_path = ".".join(module_atoms)

            module: torch.nn.Module = util.fetch_attr(self.model, module_path)

            if hook_type == "input":

                def input_hook(module, input, kwargs, module_path=module_path):
                    return self.input_hook((input, kwargs), module_path)

                self.handles.append(
                    module.register_forward_pre_hook(input_hook, with_kwargs=True)
                )

            elif hook_type == "output":

                def output_hook(module, input, output, module_path=module_path):
                    return self.output_hook(output, module_path)

                self.handles.append(module.register_forward_hook(output_hook))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Removes all handles added during __enter__."""
        for handle in self.handles:
            handle.remove()
