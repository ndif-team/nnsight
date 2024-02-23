"""This module contains logic to interleave a computation graph (an intervention graph) with the computation graph of a model.

The :class:`InterventionProxy <nnsight.intervention.InterventionProxy>` class extends the functionality of a base nnsight.tracing.Proxy.Proxy object and makes it easier for users to interact with.

:func:`intervene() <nnsight.intervention.intervene>` is the entry hook into the models computation graph in order to interleave an intervention graph.

The :class:`HookModel <nnsight.intervention.HookModel>` provides a context manager for adding input and output hooks to modules and removing them upon context exit.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Callable, Collection, Dict, List, Tuple, Union

import torch
from torch.utils.hooks import RemovableHandle
from typing_extensions import Self

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

        Calling ``.shape`` on an InterventionProxy returns the shape or collection of shapes for the tensors traced through this module.

        Calling ``.value`` on an InterventionProxy returns the actual populated values, updated during actual execution of the model.

    """

    def __init__(self, node: Node) -> None:
        super().__init__(node)

        self.__dict__["_grad"] = None

        self._grad: InterventionProxy

    def save(self) -> InterventionProxy:
        """Method when called, indicates to the intervention graph to not delete the tensor values of the result.

        Returns:
            InterventionProxy: Save proxy.
        """

        # Add a 'null' node with the save proxy as an argument to ensure the values are never deleted.
        # This is because 'null' nodes never actually get set and therefore there will always be a
        # dependency for the save proxy.
        self.node.add(
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

            # We track how many times backward is called via an attribute on the Graph
            if not hasattr(self.node.graph, 'n_backward_calls'):

                setattr(self.node.graph, 'n_backward_calls', 0)

            self.__dict__["_grad"] = self.node.add(
                value=self.node.proxy_value, target="grad", args=[self.node,self.node.graph.n_backward_calls]
            )

        return self._grad

    @grad.setter
    def grad(self, value: Union[InterventionProxy, Any]) -> None:
        """
        Calling denotes the user wishes to set the grad of this proxy tensor and therefore we create a Proxy of that request.

        Args:
            value (Union[InterventionProxy, Any]): Value to set output to.
        """
        self.node.add(target="swap", args=[self.grad.node, value], value=True)

    def __call__(self, *args, **kwargs) -> Self:


        # We don't want to call backward on fake tensors
        if (
            self.node.target is util.fetch_attr
            and isinstance(self.node.args[1], str)
            and self.node.args[1] == "backward"
        ):
            # We track how many times backward is called via an attribute on the Graph
            if not hasattr(self.node.graph, 'n_backward_calls'):

                setattr(self.node.graph, 'n_backward_calls', 0)

            # Clear all .grad proxies 
            for node in self.node.graph.nodes.values():

                try:

                    if node.proxy._grad is not None:

                        node.proxy.__dict__['_grad'] = None

                except ReferenceError:
                    pass

            self.node.graph.n_backward_calls += 1

            return self.node.add(
                value=None,
                target=Proxy.proxy_call,
                args=[self.node] + list(args),
                kwargs=kwargs,
            )


        return super().__call__(*args, **kwargs)

    def __setattr__(
        self, key: Union[InterventionProxy, Any], value: Union[Self, Any]
    ) -> None:

        if key == "grad":
            getattr(self.__class__, key).fset(self, value)

        return super().__setattr__(key, value)

    @property
    def shape(self) -> Collection[torch.Size]:
        """Property to retrieve the shape of the traced proxy value or real value.

        Returns:
            Union[torch.Size,Collection[torch.Size]]: Proxy value shape or collection of shapes.
        """

        if self.node.is_graph_dereferenced():

            return util.apply(self.value, lambda x: x.shape, torch.Tensor)

        return util.apply(self.node.proxy_value, lambda x: x.shape, torch.Tensor)

    @property
    def device(self) -> Collection[torch.device]:
        """Property to retrieve the device of the traced proxy value or real value.

        Returns:
            Union[torch.Size,Collection[torch.device]]: Proxy value shape or collection of shapes.
        """

        if self.node.is_graph_dereferenced():

            return util.apply(self.value, lambda x: x.device, torch.Tensor)

        return util.apply(self.node.proxy_value, lambda x: x.device, torch.Tensor)


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
    activations: Any,
    module_path: str,
    key: str,
    intervention_handler: InterventionHandler,
):
    """Entry to intervention graph. This should be hooked to all modules involved in the intervention graph.

    Forms the current module_path key in the form of <module path>.<output/input>
    Checks the graphs argument_node_names attribute for this key.
    If exists, value is a list of node names to iterate through.
    Node args for argument type nodes should be ``[module_path, batch_size, batch_start, call_iter]``.
    Checks and updates the counter for the given argument node. If counter is not ready yet continue.
    Using batch_size and batch_start, apply torch.narrow to tensors in activations to select
    only batch indexed tensors relevant to this intervention node. Sets the value of a node
    using the indexed values. Using torch.narrow returns a view of the tensors as opposed to a copy allowing
    subsequent downstream nodes to make edits to the values only in the relevant tensors, and have it update the original
    tensors. This both prevents interventions from effecting bathes outside their preview and allows edits
    to the output from downstream intervention nodes in the graph.

    Args:
        activations (Any): Either the inputs or outputs of a torch module.
        module_path (str): Module path of the current relevant module relative to the root model.
        key (str): Key denoting either "input" or "output" of module.
        intervention_handler (InterventionHandler): Handler object that stores the intervention graph and keeps track of module call count.

    Returns:
        Any: The activations, potentially modified by the intervention graph.
    """

    # Key to module activation argument nodes has format: <module path>.<output/input>
    module_path = f"{module_path}.{key}"

    if module_path in intervention_handler.graph.argument_node_names:
        argument_node_names = intervention_handler.graph.argument_node_names[
            module_path
        ]

        # Multiple argument nodes can have same module_path if there are multiple invocations.
        for argument_node_name in argument_node_names:

            node = intervention_handler.graph.nodes[argument_node_name]

            # Args for argument nodes are (module_path, batch_size, batch_start, call_iter).
            _, batch_size, batch_start, call_iter = node.args

            # Updates the count of argument node calls.
            # If count matches call_iter, time to inject value into node.
            if call_iter != intervention_handler.count(argument_node_name):

                continue

            # Narrow tensor values in activations only to relevant batch idxs.
            # Only narrow if batch_size != total batch size ( no need to narrow when its the entire batch )
            # and if the first dim == total_batch_size ( otherwise must not be a batched tensor ).
            # Checks to see if anything was narrowed. If not, no need to concat later.
            narrowed = False

            def narrow(acts: torch.Tensor):

                nonlocal narrowed

                if (
                    batch_size != intervention_handler.total_batch_size
                    and intervention_handler.total_batch_size == acts.shape[0]
                ):
                    narrowed = True
                    return acts.narrow(0, batch_start, batch_size)

                return acts

            value = util.apply(
                activations,
                narrow,
                torch.Tensor,
            )

            # Value injection.
            node.set_value(value)

            # Check if through the previous value injection, there was a 'swap' intervention.
            # This would mean we want to replace activations for this batch with some other ones.
            value = intervention_handler.graph.get_swap(value)

            # If we narrowed any data, we need to concat it with data before and after it.
            if narrowed:

                activations = concat(
                    activations,
                    value,
                    batch_start,
                    batch_size,
                    intervention_handler.total_batch_size,
                )
            # Otherwise just return the whole value as the activations.
            else:

                activations = value

    return activations


class HookHandler(AbstractContextManager):
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

    def __enter__(self) -> HookHandler:
        """Registers input and output hooks to modules if they are defined.

        Returns:
            HookModel: HookModel object.
        """

        for module_key in self.module_keys:

            module_atoms = module_key.split(".")

            if len(module_atoms) == 1:
                continue

            *module_atoms, hook_type = module_atoms

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

        if isinstance(exc_val, Exception):
            raise exc_val


class InterventionHandler:

    def __init__(self, graph: Graph, total_batch_size: int) -> None:

        self.graph = graph
        self.total_batch_size = total_batch_size
        self.call_counter: Dict[str, int] = {}

    def count(self, name: str):

        if name not in self.call_counter:

            self.call_counter[name] = 0

        else:

            self.call_counter[name] += 1

        return self.call_counter[name]
