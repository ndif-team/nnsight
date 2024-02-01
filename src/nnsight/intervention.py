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
    def token(self) -> TokenIndexer:
        """Property used to do token based indexing on a proxy.
        Directly indexes the second dimension of tensors.
        Makes positive indices negative as tokens are padded on the left.

        Example:

            .. code-block:: python

                model.transformer.h[0].mlp.output.token[0]

            Is equivalent to:

            .. code-block:: python

                model.transformer.h[0].mlp.output.token[:,-3]

            For a proxy tensor with 3 tokens.

        Returns:
            TokenIndexer: Object to do token based indexing.
        """
        return TokenIndexer(self)

    @property
    def t(self) -> TokenIndexer:
        """Property as alias for InterventionProxy.token"""
        return self.token

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
            # For same reason as we do total_batch_size
            # TODO
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

    # As interventions are scoped only to their relevant batch, if we want to swap in values for this batch
    # we need to concatenate the batches before and after the relevant batch with the new values.
    # Getting batch data before.

    def narrow1(acts: torch.Tensor):
        if total_batch_size == acts.shape[0]:
            return acts.narrow(0, 0, batch_start)

        return acts

    def narrow2(acts: torch.Tensor):
        if total_batch_size == acts.shape[0]:
            return acts.narrow(0, post_batch_start, acts.shape[0] - post_batch_start)

        return acts

    pre = util.apply(activations, lambda x: narrow1(x), torch.Tensor)
    post_batch_start = batch_start + batch_size
    # Getting batch data after.
    post = util.apply(
        activations,
        lambda x: narrow2(x),
        torch.Tensor,
    )

    # For same reason as we do total_batch_size
    # TODO
    orig_sizes = util.apply(activations, lambda x: x.shape[0], torch.Tensor)

    # Concatenate
    return _concat([pre, value, post, orig_sizes])


def intervene(activations: Any, module_path: str, graph: Graph, key: str):
    """Entry to intervention graph. This should be hooked to all modules involved in the intervention graph.

    Forms the current module_path key in the form of <module path>.<output/input>.<graph generation index>
    Checks the graphs argument_node_names attribute for this key.
    If exists, value is a list of node names to iterate through.
    Node args for argument type nodes should be ``[module_path, batch_size, batch_start]``.
    Using batch_size and batch_start, apply torch.narrow to tensors in activations to select
    only batch indexed tensors relevant to this intervention node. Sets the value of a node
    using the indexed values. Using torch.narrow returns a view of the tensors as opposed to a copy allowing
    subsequent downstream nodes to make edits to the values only in the relevant tensors, and have it update the original
    tensors. This both prevents interventions from effecting bathes outside their preview and allows edits
    to the output from downstream intervention nodes in the graph.

    Args:
        activations (Any): Either the inputs or outputs of a torch module.
        module_path (str): Module path of the current relevant module relative to the root model.
        graph (Graph): Intervention graph to interleave with the computation "graph" of the model.
        key (str): Key denoting either "input" or "output" of module.

    Returns:
        Any: The activations, potentially modified by the intervention graph.
    """

    # Key to module activation argument nodes has format: <module path>.<output/input>.<generation index>
    module_path = f"{module_path}.{key}.{graph.generation_idx}"

    if module_path in graph.argument_node_names:
        argument_node_names = graph.argument_node_names[module_path]

        # multiple argument nodes can have same module_path if there are multiple invocations.
        for argument_node_name in argument_node_names:
            node = graph.nodes[argument_node_name]

            # args for argument nodes are (module_path, batch_size, batch_start)
            _, batch_size, batch_start = node.args

            # We set its result to the activations, indexed by only the relevant batch idxs.

            # We find the max size of all shapes[0] and assume that is the total batch size.
            # We then use this to NOT narrow tensors that does not have this size as their first dim.
            # TODO maybe this isnt the right way to handle this. Maybe just check if multi invokes happen and if not, dont narrow.
            total_batch_size = None

            def narrow(acts: torch.Tensor):
                nonlocal total_batch_size

                _batch_size = acts.shape[0]

                if total_batch_size is None or _batch_size > total_batch_size:
                    total_batch_size = _batch_size

                if total_batch_size == _batch_size:
                    return acts.narrow(0, batch_start, batch_size)

                return acts

            value = util.apply(
                activations,
                lambda x: narrow(x),
                torch.Tensor,
            )

            node.set_value(value)

            # Check if through the previous value injection, there was a 'swp' intervention.
            # This would mean we want to replace activations for this batch with some other ones.
            value = graph.get_swap(value)

            activations = concat(
                activations, value, batch_start, batch_size, total_batch_size
            )

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
        backward_input_hook: Callable = None,
        backward_output_hook: Callable = None,
    ) -> None:
        self.model = model
        self.module_keys = module_keys

        self.input_hook = input_hook
        self.output_hook = output_hook
        self.backward_input_hook = backward_input_hook
        self.backward_output_hook = backward_output_hook

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

            elif hook_type == "backward_input":

                def backward_input_hook(module, input, output, module_path=module_path):
                    return self.backward_input_hook(input, module_path)

                self.handles.append(
                    module.register_full_backward_hook(backward_input_hook)
                )

            elif hook_type == "backward_output":

                def backward_output_hook(module, output, module_path=module_path):
                    return self.backward_output_hook(output, module_path)

                self.handles.append(
                    module.register_full_backward_pre_hook(backward_output_hook)
                )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Removes all handles added during __enter__."""
        for handle in self.handles:
            handle.remove()


class TokenIndexer:
    """Helper class to directly access token indices of hidden states.
    Directly indexes the second dimension of tensors.
    Makes positive indices negative as tokens are padded on the left.

    Args:
        proxy (InterventionProxy): Proxy to aid in token indexing.
    """

    def __init__(self, proxy: InterventionProxy) -> None:
        self.proxy = proxy

    def convert_idx(self, idx: int):
        if idx >= 0:
            n_tokens = self.proxy.node.proxy_value.shape[1]
            idx = -(n_tokens - idx)

        return idx

    def __getitem__(self, key: int) -> Proxy:
        key = self.convert_idx(key)

        return self.proxy[:, key]

    def __setitem__(self, key: int, value: Union[Proxy, Any]) -> None:
        key = self.convert_idx(key)

        self.proxy[:, key] = value
