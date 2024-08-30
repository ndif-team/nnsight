"""This module contains logic to interleave a computation graph (an intervention graph) with the computation graph of a model.

The :class:`InterventionProxy <nnsight.intervention.InterventionProxy>` class extends the functionality of a base nnsight.tracing.Proxy.Proxy object and makes it easier for users to interact with.

:func:`intervene() <nnsight.intervention.InterventionProtocol.intervene>` is the entry hook into the models computation graph in order to interleave an intervention graph.

The :class:`HookModel <nnsight.intervention.HookModel>` provides a context manager for adding input and output hooks to modules and removing them upon context exit.
"""

from __future__ import annotations

import inspect
from collections import defaultdict
from contextlib import AbstractContextManager
from typing import Any, Callable, Collection, Dict, List, Tuple, Union

import torch
from torch.utils.hooks import RemovableHandle
from typing_extensions import Self

from . import util
from .contexts.Conditional import Conditional
from .tracing import protocols
from .tracing.Graph import Graph
from .tracing.Node import Node
from .tracing.protocols import Protocol
from .tracing.Proxy import Proxy


class InterventionProxy(Proxy):
    """Sub-class for Proxy that adds additional user functionality to proxies.

    Examples:

        Saving a proxy so it is not deleted at the completion of it's listeners is enabled with ``.save()``:

        .. code-block:: python

            with model.trace('The Eiffel Tower is in the city of'):
                hidden_states = model.lm_head.input.save()
                logits = model.lm_head.output.save()

            print(hidden_states)
            print(logits)
    """

    def __init__(self, node: Node) -> None:
        super().__init__(node)

        self.__dict__["_grad"] = None

        self._grad: InterventionProxy

    def save(self) -> InterventionProxy:
        """Method when called, indicates to the intervention graph to not delete the tensor values of the result.

        Returns:
            InterventionProxy: Proxy.
        """

        # Add a 'lock' node with the save proxy as an argument to ensure the values are never deleted.
        # This is because 'lock' nodes never actually get set and therefore there will always be a
        # dependency for the save proxy.

        protocols.LockProtocol.add(self.node)

        return self

    def stop(self) -> InterventionProxy:
        """Method when called, indicates to the intervention graph to stop the execution of the model after this Proxy/Node is completed..

        Returns:
            InterventionProxy: Proxy.
        """

        protocols.EarlyStopProtocol.add(self.node.graph, self.node)

        return self

    def update(self, value: Union[Node, Any]) -> InterventionProxy:
        """Updates the value of the Proxy via the creation of the UpdateProtocol node.

        Args:
            - value (Union[Node, Any]): New proxy value.

        Returns:
            InterventionProxy: Proxy.

        .. codeb-block:: python
            with model.trace(input) as tracer:
                num = tracer.apply(int, 0)
                num.update(5)
        """

        return protocols.UpdateProtocol.add(self.node, value)

    @property
    def grad(self) -> InterventionProxy:
        """
        Calling denotes the user wishes to get the grad of proxy tensor and therefore we create a Proxy of that request.
        Only generates a proxy the first time it is references otherwise return the already set one.

        Returns:
            Proxy: Grad proxy.
        """
        if self._grad is None:

            self.__dict__["_grad"] = protocols.GradProtocol.add(self.node)

        return self._grad

    @grad.setter
    def grad(self, value: Union[InterventionProxy, Any]) -> None:
        """
        Calling denotes the user wishes to set the grad of this proxy tensor and therefore we create a Proxy of that request via a SwapProtocol.

        Args:
            value (Union[InterventionProxy, Any]): Value to set output to.
        """
        protocols.SwapProtocol.add(self.grad.node, value)

        self.__dict__["_grad"] = None

    def __call__(self, *args, **kwargs) -> Self:

        # We don't want to call backward on fake tensors.
        # We also want to track the number of times .backward() has been called so .grad on a Proxy refers to the right backward pass.
        if (
            self.node.target is util.fetch_attr
            and isinstance(self.node.args[1], str)
            and self.node.args[1] == "backward"
        ):

            # Clear all .grad proxies so allow users to get the ,.grad of the next backward pass.
            for node in self.node.graph.nodes.values():

                try:

                    if node.proxy._grad is not None:

                        node.proxy.__dict__["_grad"] = None

                except ReferenceError:
                    pass

            # Use GradProtocol to increment the tracking of the number of times .backward() has been called.
            protocols.GradProtocol.increment(self.node.graph)

            return self.node.create(
                proxy_value=None,
                target=Proxy.proxy_call,
                args=[self.node] + list(args),
                kwargs=kwargs,
            )

        return super().__call__(*args, **kwargs)

    def __setattr__(
        self, key: Union[InterventionProxy, Any], value: Union[Self, Any]
    ) -> None:

        # We catch setting .grad as that is a special Protocol vs. setting attributes generally.
        if key == "grad":
            return getattr(self.__class__, key).fset(self, value)

        return super().__setattr__(key, value)

    @property
    def shape(self) -> Collection[torch.Size]:
        """Property to retrieve the shape of the traced proxy value or real value.

        Returns:
            Union[torch.Size,Collection[torch.Size]]: Proxy value shape or collection of shapes.
        """

        if not self.node.attached():

            return util.apply(self.value, lambda x: x.shape, torch.Tensor)

        # If we haven't scanned in a proxy_value, just return a proxy to get the attribute.
        if self.node.proxy_value is inspect._empty:

            return super().__getattr__("shape")

        return util.apply(
            self.node.proxy_value, lambda x: x.shape, torch.Tensor
        )

    @property
    def device(self) -> Collection[torch.device]:
        """Property to retrieve the device of the traced proxy value or real value.

        Returns:
            Union[torch.Size,Collection[torch.device]]: Proxy value device or collection of devices.
        """

        if not self.node.attached():

            return util.apply(self.value, lambda x: x.device, torch.Tensor)

        # If we haven't scanned in a proxy_value, just return a proxy to get the attribute.
        if self.node.proxy_value is inspect._empty:

            return super().__getattr__("device")

        return util.apply(
            self.node.proxy_value, lambda x: x.device, torch.Tensor
        )

    @property
    def dtype(self) -> Collection[torch.device]:
        """Property to retrieve the dtype of the traced proxy value or real value.

        Returns:
            Union[torch.Size,Collection[torch.dtype]]: Proxy value dtype or collection of dtypes.
        """

        if not self.node.attached():

            return util.apply(self.value, lambda x: x.dtype, torch.Tensor)

        # If we haven't scanned in a proxy_value, just return a proxy to get the attribute.
        if self.node.proxy_value is inspect._empty:

            return super().__getattr__("dtype")

        return util.apply(
            self.node.proxy_value, lambda x: x.dtype, torch.Tensor
        )


class InterventionProtocol(Protocol):
    """Primary Protocol that handles tracking and injecting inputs and outputs from a torch model into the overall intervention Graph.
    Uses an attachment on the Graph to store the names of nodes that need to be injected with data from inputs or outputs of modules.
    """

    attachment_name = "nnsight_module_nodes"
    condition: bool = False

    @classmethod
    def add(
        cls,
        graph: "Graph",
        proxy_value: Any,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
    ) -> Proxy:
        """Adds an InterventionProtocol Node to a Graph.

        Args:
            graph (Graph): Graph to add to.
            module_path (str): Module path of data this Node depends on (ex. model.module1.module2.output)
            proxy_value (Any): Proxy value.
            args (List[Any], optional): Args. Defaults to None.
            kwargs (Dict[str, Any], optional): Kwargs. Defaults to None.

        Returns:
            Proxy: _description_
        """

        # Creates the InterventionProtocol Node.
        proxy = graph.create(
            proxy_value=proxy_value, target=cls, args=args, kwargs=kwargs
        )

        cls.compile(proxy.node)

        return proxy

    @classmethod
    def compile(cls, node: Node) -> None:

        graph = node.graph

        module_path, *_ = node.args

        # Add attachment if it does not exist.
        if cls.attachment_name not in graph.attachments:

            graph.attachments[cls.attachment_name] = dict()

        # More than one Node can depend on a given input or output, therefore we store a list of node names.
        arguments = graph.attachments[cls.attachment_name]

        if module_path not in arguments:
            arguments[module_path] = []

        # Append the newly created nodes name.
        arguments[module_path].append(node.name)

    @classmethod
    def get_interventions(cls, graph: "Graph") -> Dict:
        """Returns mapping from module_paths to  InterventionNode names added to the given Graph.

        Args:
            graph (Graph): Graph.

        Returns:
            Dict: Interventions.
        """

        return graph.attachments.get(cls.attachment_name, dict())

    @classmethod
    def concat(
        cls,
        activations: Any,
        value: Any,
        batch_start: int,
        batch_size: int,
        total_batch_size: int,
    ):
        def _concat(values):

            data_type = type(values[0])

            if data_type == torch.Tensor:
                orig_size = values[-1]
                new_size = sum([value.shape[0] for value in values[:-1]])
                if new_size == orig_size:
                    return torch.concatenate(values[:-1])

                return values[0]
            elif data_type == list:
                return [
                    _concat([value[value_idx] for value in values])
                    for value_idx in range(len(values[0]))
                ]
            elif data_type == tuple:
                return tuple(
                    [
                        _concat([value[value_idx] for value in values])
                        for value_idx in range(len(values[0]))
                    ]
                )
            elif data_type == dict:
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
                return acts.narrow(
                    0, post_batch_start, acts.shape[0] - post_batch_start
                )

            return acts

        post = util.apply(
            activations,
            narrow2,
            torch.Tensor,
        )

        orig_sizes = util.apply(activations, lambda x: x.shape[0], torch.Tensor)

        return _concat([pre, value, post, orig_sizes])

    @classmethod
    def intervene(
        cls,
        activations: Any,
        module_path: str,
        key: str,
        intervention_handler: InterventionHandler,
    ):
        """Entry to intervention graph. This should be hooked to all modules involved in the intervention graph.

        Forms the current module_path key in the form of <module path>.<output/input>
        Checks the graphs InterventionProtocol attachment attribute for this key.
        If exists, value is a list of node names to iterate through.
        Node args for intervention type nodes should be ``[module_path, batch_size, batch_start, call_iter]``.
        Checks and updates the counter for the given intervention node. If counter is not ready yet continue.
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

        # Key to module activation intervention nodes has format: <module path>.<output/input>
        module_path = f"{module_path}.{key}"

        interventions = cls.get_interventions(intervention_handler.graph)

        if module_path in interventions:
            intervention_node_names = interventions[module_path]

            # Multiple intervention nodes can have same module_path if there are multiple invocations.
            for intervention_node_name in intervention_node_names:

                node = intervention_handler.graph.nodes[intervention_node_name]

                # Args for intervention nodes are (module_path, batch_group_idx, call_iter).
                _, batch_group_idx, call_iter = node.args

                batch_start, batch_size = intervention_handler.batch_groups[
                    batch_group_idx
                ]

                # Updates the count of intervention node calls.
                # If count matches call_iter, time to inject value into node.
                if call_iter != intervention_handler.count(
                    intervention_node_name
                ):

                    continue

                value = activations

                narrowed = False

                if len(intervention_handler.batch_groups) > 1:

                    def narrow(acts: torch.Tensor):

                        if acts.shape[0] == intervention_handler.batch_size:

                            nonlocal narrowed

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
                value = protocols.SwapProtocol.get_swap(
                    intervention_handler.graph, value
                )

                # If we narrowed any data, we need to concat it with data before and after it.
                if narrowed:

                    activations = cls.concat(
                        activations,
                        value,
                        batch_start,
                        batch_size,
                        intervention_handler.batch_size,
                    )
                # Otherwise just return the whole value as the activations.
                else:

                    activations = value

        return activations

    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        return {
            "node": {"color": "green4", "shape": "box"},  # Node display
            "label": cls.__name__,
            "arg": defaultdict(
                lambda: {"color": "gray", "shape": "box"}
            ),  # Non-node argument display
            "arg_kname": defaultdict(
                lambda: None, {0: "key", 1: "batch_size", 2: "batch_start"}
            ),  # Argument label key word
            "edge": defaultdict(lambda: "solid"),
        }  # Argument Edge display


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
                    module.register_forward_pre_hook(
                        input_hook, with_kwargs=True, prepend=True
                    )
                )

            elif hook_type == "output":

                def output_hook(module, input, output, module_path=module_path):
                    return self.output_hook(output, module_path)

                self.handles.append(
                    module.register_forward_hook(output_hook, prepend=True)
                )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Removes all handles added during __enter__."""

        for handle in self.handles:
            handle.remove()

        if isinstance(exc_val, Exception):
            raise exc_val


class InterventionHandler:
    """Object passed to InterventionProtocol.intervene to store information about the current interleaving execution run.

    Like the Intervention Graph, the total batch size that is being executed, and a counter for how many times an Intervention node has been attempted to be executed.
    """

    def __init__(
        self, graph: Graph, batch_groups: List[Tuple[int, int]], batch_size: int
    ) -> None:

        self.graph = graph
        self.batch_groups = batch_groups
        self.batch_size = batch_size
        self.call_counter: Dict[str, int] = {}

    def count(self, name: str) -> int:
        """Increments the count of times a given Intervention Node has tried to be executed and returns the count.

        Args:
            name (str): Name of intervention node to return count for.

        Returns:
            int: Count.
        """

        if name not in self.call_counter:

            self.call_counter[name] = 0

        else:

            self.call_counter[name] += 1

        return self.call_counter[name]
