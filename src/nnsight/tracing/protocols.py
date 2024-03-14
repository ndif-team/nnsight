import inspect
from typing import TYPE_CHECKING, Any, Dict, List

import torch
from torch._subclasses.fake_tensor import FakeCopyMode, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from .. import util

if TYPE_CHECKING:
    from ..intervention import InterventionProxy
    from .Bridge import Bridge
    from .Graph import Graph
    from .Node import Node


class Protocol:
    """A `Protocol` represents some complex action a user might want to create a `Node` and `Proxy` for as well as add to a `Graph`.
    Unlike normal `Node` target execution, these have access to the `Node` itself and therefore the `Graph`, enabling more powerful functionality than with just functions and methods.
    """

    name: str

    @classmethod
    def add(cls, *args, **kwargs) -> "InterventionProxy":
        """Class method to be implemented in order to add a Node of this Protocol to a Graph."""

        raise NotImplementedError()

    @classmethod
    def execute(cls, node: "Node"):
        """Class method to be implemented which contains the actual execution logic of the Protocol. By default, does nothing

        Args:
            node (Node): Node to execute using this Protocols execution logic.
        """
        pass

    @classmethod
    def compile(cls, node: "Node") -> None:
        pass


class ApplyModuleProtocol(Protocol):
    """Protocol that references some root model, and calls its .forward() method given some input.
    Using .forward() vs .__call__() means it wont trigger hooks.
    Uses an attachment to the Graph to store the model.
    """

    attachment_name = "nnsight_root_module"

    @classmethod
    def add(
        cls, graph: "Graph", module_path: str, *args, **kwargs
    ) -> "InterventionProxy":
        """Creates and adds an ApplyModuleProtocol to the Graph.
        Assumes the attachment has already been added via ApplyModuleProtocol.set_module().

        Args:
            graph (Graph): Graph to add the Protocol to.
            module_path (str): Module path (model.module1.module2 etc), of module to apply from the root module.

        Returns:
            InterventionProxy: ApplyModule Proxy.
        """

        value = inspect._empty

        # If the Graph is validating, we need to compute the proxy_value for this node.
        if graph.validate:

            # If the module has parameters, get its device to move input tensors to.
            try:

                module = cls.get_module(graph)
                device = next(module.parameters()).device

            except:

                device = None

            # Enter FakeMode for proxy_value computing.
            with FakeTensorMode(
                allow_non_fake_inputs=True,
                shape_env=ShapeEnv(assume_static_by_default=True),
            ) as fake_mode:
                with FakeCopyMode(fake_mode):

                    value = cls.call_module(
                        graph,
                        module_path,
                        *Node.prepare_proxy_values(args, device=device),
                        **Node.prepare_proxy_values(kwargs, device=device),
                    )

        # Create and attach Node.
        return graph.create(
            target=cls,
            proxy_value=value,
            args=[module_path] + list(args),
            kwargs=kwargs,
        )

    @classmethod
    def execute(cls, node: "Node") -> None:
        """Executes the ApplyModuleProtocol on Node.

        Args:
            node (Node): ApplyModule Node.
        """

        args, kwargs = node.prepare_inputs()

        module_path, *args = args

        output = cls.call_module(node.graph, module_path, *args, **kwargs)

        node.set_value(output)

    @classmethod
    def call_module(cls, graph: "Graph", module_path: str, *args, **kwargs) -> Any:
        """Given a Graph (with the nnsight root module attached) and module_path, get the root module from the Graph,
        get the submodule using the module_path, and call its .forward() method using the given args.

        Args:
            graph (Graph): Graph.
            module_path (str): Module path (model.module1.module2 etc).

        Returns:
            Any: Output of module's .forward() call.
        """

        module: torch.nn.Module = util.fetch_attr(cls.get_module(graph), module_path)

        return module.forward(*args, **kwargs)

    @classmethod
    def set_module(cls, graph: "Graph", module: torch.nn.Module) -> None:
        """Sets the nnsight root module as an attachment on a Graph.

        Args:
            graph (Graph): Graph.
            module (torch.nn.Module): Root module.
        """

        graph.attachments[cls.attachment_name] = module

    @classmethod
    def get_module(cls, graph: "Graph") -> torch.nn.Module:
        """Returns the nnsight root module from an attachment on a Graph.

        Args:
            graph (Graph): Graph

        Returns:
            torch.nn.Module: Root Module.
        """

        return graph.attachments[cls.attachment_name]


class LockProtocol(Protocol):
    """Simple Protocol who's .execute() method does nothing. This means not calling .set_value() on the Node, therefore  the Node won't be destroyed."""

    @classmethod
    def add(cls, node: "Node") -> "InterventionProxy":

        return node.create(
            proxy_value=None,
            target=cls,
            args=[node],
        )


class GradProtocol(Protocol):
    """Protocol which adds a backwards hook via .register_hook() to a Tensor. The hook injects the gradients into the node's value on hook execution.
    Nodes created via this protocol are relative to the next time .backward() was called during tracing allowing separate .grads to reference separate backwards passes:

    .. code-block:: python
        with model.trace(...):

            grad1 = model.module.output.grad.save()

            model.output.sum().backward(retain_graph=True)

            grad2 = model.module.output.grad.save()

            model.output.sum().backward()

    Uses an attachment to store number of times .backward() has been called during tracing so a given .grad hook is only value injected at the appropriate backwards pass.
    """

    attachment_name = "nnsight_backward_idx"

    @classmethod
    def add(cls, node: "Node") -> "InterventionProxy":

        # Get number of times .backward() was called during tracing from an attachment. Use as Node argument.
        backward_idx = node.graph.attachments.get(cls.attachment_name, 0)

        return node.create(
            proxy_value=node.proxy_value,
            target=cls,
            args=[node, backward_idx],
        )

    @classmethod
    def execute(cls, node: "Node") -> None:

        args, kwargs = node.prepare_inputs()

        # First arg is the Tensor to add hook to.
        tensor: torch.Tensor = args[0]
        # Second is which backward pass this Node refers to.
        backward_idx: int = args[1]

        # Hook to remove when hook is executed at the appropriate backward pass.
        hook = None

        def grad(value):

            nonlocal backward_idx

            # If backward_idx == 0, this is the correct backward pass and we should actually execute.
            if backward_idx == 0:

                # Set the value of the Node.
                node.set_value(value)

                if node.attached():

                    # There may be a swap Protocol executed during the resolution of this part of the graph.
                    # If so get it and replace value with it.
                    value = SwapProtocol.get_swap(node.graph, value)

                # Don't execute this hook again.
                backward_idx = -1

                # Remove hook (if this is not done memory issues occur)
                hook.remove()

                return value

            # Otherwise decrement backward_idx
            else:

                backward_idx -= 1

                return None

        # Register hook.
        hook = tensor.register_hook(grad)

    @classmethod
    def increment(cls, graph: "Graph"):
        """Increments the backward_idx attachment to track the number of times .backward() is called in tracing for this Graph.

        Args:
            graph (Graph): Graph.
        """

        backward_idx = graph.attachments.get(cls.attachment_name, 0)

        graph.attachments[cls.attachment_name] = backward_idx + 1


class SwapProtocol(Protocol):
    """Protocol which adds an attachment to the Graph which can store some value. Used to replace ('swap') a value with another value."""

    attachment_name = "nnsight_swap"

    @classmethod
    def add(cls, node: "Node", value: Any) -> "InterventionProxy":

        return node.create(target=cls, args=[node, value], proxy_value=True)

    @classmethod
    def execute(cls, node: "Node") -> None:

        # In case there is already a swap, get it from attachments.
        swap: "Node" = node.graph.attachments.get(cls.attachment_name, None)

        # And set it to False to destroy it.
        if swap is not None:
            swap.set_value(False)

        # Set the swap to this Node.
        node.graph.attachments[cls.attachment_name] = node

    @classmethod
    def get_swap(cls, graph: "Graph", value: Any) -> Any:
        """Checks if a swap exists on a Graph. If so get and return it, otherwise return the given value.

        Args:
            graph (Graph): Graph
            value (Any): Default value.

        Returns:
            Any: Default value or swap value.
        """

        # Tries to get the swap.
        swap: "Node" = graph.attachments.get(cls.attachment_name, None)

        # If there was one:
        if swap is not None:

            device = None

            def _device(value: torch.Tensor):
                nonlocal device

                device = value.device

            # Get device of default value.
            util.apply(value, _device, torch.Tensor)

            # Get swap Node's value.
            value = util.apply(swap.args[1], lambda x: x.value, type(swap))

            if device is not None:

                def _to(value: torch.Tensor):
                    return value.to(device)

                # Move swap values to default value's device.
                value = util.apply(value, _to, torch.Tensor)

            # Set value of 'swap' node so it destroys itself and listeners.
            swap.set_value(True)

            # Un-set swap.
            graph.attachments[cls.attachment_name] = None

        return value


class BridgeProtocol(Protocol):
    """Protocol to connect two Graphs by grabbing a value from one and injecting it into another.
    Uses an attachment to store a Bridge object which references all relevant Graphs and their ordering.
    """

    attachment_name = "nnsight_bridge"

    @classmethod
    def add(cls, from_node: "Node", to_node: "Node") -> "InterventionProxy":

        # Adds a Lock Node. One, so the value from_node isn't destroyed until the to_nodes are done with it,
        # and two acts as an easy reference to the from_node to get its value from the lock Node args.
        lock_node = LockProtocol.add(from_node).node

        # Args for a Bridge Node are the id of the Graph and node name of the Lock Node.
        return to_node.create(
            target=cls,
            proxy_value=from_node.proxy_value,
            args=[from_node.graph.id, lock_node.name],
        )

    @classmethod
    def execute(cls, node: "Node") -> None:

        # Gets Bridge object from the Node's Graph.
        bridge = cls.get_bridge(node.graph)

        # Args are Graph's id and name of the Lock Node on it.
        from_graph_id, lock_node_name = node.args

        # Gets the from_node's Graph via its id with the Bridge and get the Lock Node.
        lock_node = bridge.get_graph(from_graph_id).nodes[lock_node_name]

        # Value node is Lock Node's only arg
        value_node: "Node" = lock_node.args[0]

        # Set value to that of the value Node.
        node.set_value(value_node.value)

        # Bridge.release tells this Protocol when to release all Lock Nodes as we no longer need the data (useful when running a Graph in a loop, only release on last iteration)
        if bridge.release:

            lock_node.set_value(None)

    @classmethod
    def set_bridge(cls, graph: "Graph", bridge: "Bridge") -> None:
        """Sets Bridge object as an attachment on a Graph.

        Args:
            graph (Graph): Graph.
            bridge (Bridge): Bridge.
        """

        graph.attachments[cls.attachment_name] = bridge

    @classmethod
    def get_bridge(cls, graph: "Graph") -> "Bridge":
        """Gets Brudge object from a Graph. Assumes Bridge has been set as an attachment on this Graph via BridgeProtocol.set_bridge().

        Args:
            graph (Graph): Graph.

        Returns:
            Bridge: Bridge.
        """

        if not cls.has_bridge(graph):
            # TODO error
            pass

        return graph.attachments[cls.attachment_name]

    @classmethod
    def has_bridge(cls, graph: "Graph") -> bool:
        """Checks to see if a Bridge was added as an attachment on this Graph via BridgeProtocol.set_bridge().

        Args:
            graph (Graph): Graph

        Returns:
            bool: If Graph has Bridge attachment.
        """

        return cls.attachment_name in graph.attachments
