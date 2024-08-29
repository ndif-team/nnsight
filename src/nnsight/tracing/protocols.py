import inspect
import weakref
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
from torch._subclasses.fake_tensor import FakeCopyMode, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from nnsight.tracing.Node import Node

from .. import util
from ..contexts.Conditional import ConditionalManager
from .util import validate

if TYPE_CHECKING:
    from ..contexts.backends.LocalBackend import LocalMixin
    from ..contexts.Conditional import Conditional
    from ..intervention import InterventionProxy
    from .Bridge import Bridge
    from .Graph import Graph
    from .Node import Node


class Protocol:
    """A `Protocol` represents some complex action a user might want to create a `Node` and `Proxy` for as well as add to a `Graph`.
    Unlike normal `Node` target execution, these have access to the `Node` itself and therefore the `Graph`, enabling more powerful functionality than with just functions and methods.
    """

    redirect: bool = True
    condition: bool = True

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

    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        return {
            "node": {"color": "black", "shape": "ellipse"},  # Node display
            "label": cls.__name__,
            "arg": defaultdict(
                lambda: {"color": "gray", "shape": "box"}
            ),  # Non-node argument display
            "arg_kname": defaultdict(lambda: None),  # Argument label key word
            "edge": defaultdict(lambda: "solid"),
        }  # Argument edge display


class ApplyModuleProtocol(Protocol):
    """Protocol that references some root model, and calls its .forward() method given some input.
    Using .forward() vs .__call__() means it wont trigger hooks.
    Uses an attachment to the Graph to store the model.
    """

    attachment_name = "nnsight_root_module"

    @classmethod
    def add(
        cls, graph: "Graph", module_path: str, *args, hook=False, **kwargs
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

            from .Node import Node

            # If the module has parameters, get its device to move input tensors to.
            module: torch.nn.Module = util.fetch_attr(
                cls.get_module(graph), module_path
            )

            try:
                device = next(module.parameters()).device
            except:
                device = None

            # Enter FakeMode for proxy_value computing.
            value = validate(module.forward, *args, **kwargs)

        kwargs["hook"] = hook

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

        module: torch.nn.Module = util.fetch_attr(
            cls.get_module(node.graph), node.args[0]
        )

        try:
            device = next(module.parameters()).device
        except:
            device = None

        args, kwargs = node.prepare_inputs(
            (node.args, node.kwargs), device=device
        )

        module_path, *args = args

        hook = kwargs.pop("hook")

        if hook:
            output = module(*args, **kwargs)
        else:
            output = module.forward(*args, **kwargs)

        node.set_value(output)

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

    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        return {
            "node": {
                "color": "blue",
                "shape": "polygon",
                "sides": 6,
            },  # Node display
            "label": cls.__name__,
            "arg": defaultdict(
                lambda: {"color": "gray", "shape": "box"}
            ),  # Non-node argument display
            "arg_kname": defaultdict(lambda: None),  # Argument label word
            "edge": defaultdict(lambda: "solid"),
        }  # Argument edge display


class LockProtocol(Protocol):
    """Simple Protocol who's .execute() method does nothing. This means not calling .set_value() on the Node, therefore  the Node won't be destroyed."""

    redirect: bool = False

    @classmethod
    def add(cls, node: "Node") -> "InterventionProxy":

        return node.create(
            proxy_value=None,
            target=cls,
            args=[node],
        )

    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        return {
            "node": {"color": "brown", "shape": "ellipse"},  # Node display
            "label": cls.__name__,
            "arg": defaultdict(
                lambda: {"color": "gray", "shape": "box"}
            ),  # Non-node argument display
            "arg_kname": defaultdict(lambda: None),  # Argument lable key word
            "edge": defaultdict(lambda: "solid"),
        }  # Argument edge display


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

        args, kwargs = node.prepare_inputs((node.args, node.kwargs))

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
            "arg_kname": defaultdict(lambda: None),  # Argument label key word
            "edge": defaultdict(lambda: "solid"),
        }  # Argument edge display


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

    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        return {
            "node": {"color": "green4", "shape": "ellipse"},  # Node display
            "label": cls.__name__,
            "arg": defaultdict(
                lambda: {"color": "gray", "shape": "box"}
            ),  # Non-node argument display
            "arg_kname": defaultdict(lambda: None),  # Argument label key word
            "edge": defaultdict(lambda: "solid"),
        }  # Argument edge key word


class BridgeProtocol(Protocol):
    """Protocol to connect two Graphs by grabbing a value from one and injecting it into another.
    Uses an attachment to store a Bridge object which references all relevant Graphs and their ordering.
    """

    attachment_name = "nnsight_bridge"
    condition: bool = False

    class BridgeException(Exception):
        def __init__(self):
            super.__init__(
                "Must define a Session context to make use of the Bridge"
            )

    @classmethod
    def add(cls, node: "Node") -> "InterventionProxy":

        bridge = cls.get_bridge(node.graph)
        curr_graph = bridge.peek_graph()
        bridge_proxy = bridge.get_bridge_proxy(
            node, curr_graph.id
        )  # a bridged node has a unique bridge node proxy per graph reference

        # if the bridge node does not exist, create one
        if bridge_proxy is None:
            # Adds a Lock Node. One, so the value from_node isn't destroyed until the to_nodes are done with it,
            # and two acts as an easy reference to the from_node to get its value from the lock Node args.
            lock_node = LockProtocol.add(node).node

            # Args for a Bridge Node are the id of the Graph and node name of the Lock Node.
            bridge_proxy = node.create(
                target=cls,
                proxy_value=node.proxy_value,
                args=[node.graph.id, lock_node.name],
            )
            bridge.add_bridge_proxy(node, bridge_proxy)

        return bridge_proxy

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
        
        if value_node.done():

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

        graph.attachments[cls.attachment_name] = weakref.proxy(bridge)

    @classmethod
    def get_bridge(cls, graph: "Graph") -> "Bridge":
        """Gets Bridge object from a Graph. Assumes Bridge has been set as an attachment on this Graph via BridgeProtocol.set_bridge().

        Args:
            graph (Graph): Graph.

        Returns:
            Bridge: Bridge.
        """

        if not cls.has_bridge(graph):
            raise cls.BridgeException()

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

    @classmethod
    def peek_graph(cls, graph: "Graph") -> "Graph":
        """Returns current Intervention Graph.

        Args:
            - graph (Graph): Graph.

        Returns:
            Graph: Graph.
        """

        if not BridgeProtocol.has_bridge(graph):
            return graph
        else:
            bridge = BridgeProtocol.get_bridge(graph)
            return bridge.peek_graph()

    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        return {
            "node": {"color": "brown", "shape": "box"},  # Node display
            "label": cls.__name__,
            "arg": defaultdict(
                lambda: {
                    "color": "gray",
                    "shape": "box",
                },  # Non-node argument display
                {0: {"color": "gray", "shape": "box", "style": "dashed"}},
            ),
            "arg_kname": defaultdict(
                lambda: None, {0: "graph_id"}
            ),  # Arugment label key word
            "edge": defaultdict(lambda: "solid", {0: "dashed"}),
        }  # Argument edge display


class EarlyStopProtocol(Protocol):
    """Protocol to stop the execution of a model early."""

    class EarlyStopException(Exception):
        pass

    @classmethod
    def add(
        cls, graph: "Graph", stop_point_node: Optional["Node"] = None
    ) -> "InterventionProxy":
        return graph.create(
            target=cls,
            proxy_value=None,
            args=([stop_point_node] if stop_point_node is not None else []),
        )

    @classmethod
    def execute(cls, node: "Node") -> None:

        node.set_value(True)

        raise cls.EarlyStopException()

    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        return {
            "node": {
                "color": "red",
                "shape": "polygon",
                "sides": 6,
            },  # Node display
            "label": cls.__name__,
            "arg": defaultdict(
                lambda: {"color": "gray", "shape": "box"}
            ),  # Non-node argument display
            "arg_kname": defaultdict(lambda: None),  # Argument label key word
            "edge": defaultdict(lambda: "solid"),
        }  # Argument edge display


class LocalBackendExecuteProtocol(Protocol):

    @classmethod
    def add(cls, object: "LocalMixin", graph: "Graph") -> "InterventionProxy":

        return graph.create(target=cls, proxy_value=None, args=[object])

    @classmethod
    def execute(cls, node: Node) -> None:

        object: "LocalMixin" = node.args[0]

        object.local_backend_execute()

        node.set_value(None)

    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        return {
            "node": {
                "color": "purple",
                "shape": "polygon",
                "sides": 6,
            },  # Node display
            "label": "ExecuteProtocol",
            "arg": defaultdict(
                lambda: {"color": "gray", "shape": "box"}
            ),  # Non-node argument display
            "arg_kname": defaultdict(lambda: None),  # Argument label key word
            "edge": defaultdict(lambda: "solid"),
        }  # Argument edge display


class ValueProtocol(Protocol):

    @classmethod
    def add(cls, graph: "Graph", default: Any = None) -> "InterventionProxy":

        return graph.create(target=cls, proxy_value=default, args=[default])

    @classmethod
    def execute(cls, node: Node) -> None:
        
        node.set_value(node.args[0])

    @classmethod
    def set(cls, node: Node, value: Any) -> None:

        node.args[0] = value

    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        return {
            "node": {"color": "blue", "shape": "box"},  # Node display
            "label": cls.__name__,
            "arg": defaultdict(
                lambda: {"color": "gray", "shape": "box"}
            ),  # Non-node argument
            "arg_kname": defaultdict(lambda: None),  # Argument label key word
            "edge": defaultdict(lambda: "solid"),
        }  # Argument edge display


class ConditionalProtocol(Protocol):
    """Protocol operating as a conditional statement.
    Uses the ConditionalManager attachment to handle all visited Conditional contexts within a single Intervention Graph.
    Evaluates the condition value of the Conditional as a boolean.

    Example:

        Setup:
            .. code-block:: python
                import torch
                from collections import OrderedDict

                input_size = 5
                hidden_dims = 10
                output_size = 2

                model = nn.Sequential(OrderedDict([
                    ('layer1', torch.nn.Linear(input_size, hidden_dims)),
                    ('layer2', torch.nn.Linear(hidden_dims, output_size)),
                ]))

                input = torch.rand((1, input_size))    å

        Ex 1: The .save() on the model output will only be executed if the condition (x > 0) is evaluated to True.

        .. code-block:: python
            with model.trace(input) as tracer:
                num = 5
                with tracer.cond(x > 0):
                    out = model.output.save()

        Ex 2: The condition is a tensor boolean operation on the Envoy's output InterventionProxy.

        .. code-block:: python
            with model.trace(input) as tracer:
                l1_out = model.layer1.output
                with tracer.cond(l1_out[:, 0] > 0):
                    out = model.output.save()
    """

    attachment_name = "nnsight_conditional_manager"

    @classmethod
    def add(
        cls, graph: "Graph", condition: Union["Node", Any]
    ) -> "InterventionProxy":

        return graph.create(target=cls, proxy_value=True, args=[condition])

    @classmethod
    def execute(cls, node: "Node") -> None:
        """Evaluate the node condition to a boolean.

        Args:
            node (Node): ConditionalProtocol node.
        """

        cond_value = Node.prepare_inputs(node.args[0])
        if cond_value:
            # cond_value is True
            node.set_value(True)
            return

        def update_conditioned_nodes(conditioned_node: "Node") -> None:
            """Recursively decrement the remaining listeners count of all the dependencies of conditioned nodes.

            Args:
                - conditioned_node (Node): Conditioned Node
            """
            for listener in conditioned_node.listeners:
                for listener_arg in listener.arg_dependencies:
                    listener_arg.remaining_listeners -= 1
                    if listener_arg.done() and listener_arg.redundant():
                        listener_arg.destroy()
                update_conditioned_nodes(listener)

        # If the condition value is ignore or evaluated to False, update conditioned nodes
        update_conditioned_nodes(node)

    @classmethod
    def has_conditional(cls, graph: "Graph") -> bool:
        """Checks if the Intervention Graph has a ConditionalManager attached to it.

        Args:
            graph (Graph): Intervention Graph.

        Returns:
            bool: If graph has a ConditionalManager attachement.
        """
        return cls.attachment_name in graph.attachments.keys()

    @classmethod
    def get_conditional(
        cls, graph: "Graph", cond_node_name: str
    ) -> "Conditional":
        """Gets the ConditionalProtocol node by its name.

        Args:
            graph (Graph): Intervention Graph.
            cond_node_name (str): ConditionalProtocol name.

        Returns:
            Node: ConditionalProtocol Node.
        """
        return graph.attachments[cls.attachment_name].get(cond_node_name)

    @classmethod
    def push_conditional(cls, node: "Node") -> None:
        """Attaches a Conditional context to its graph.

        Args:
            node (Node): ConditionalProtocol of the current Conditional context.
        """

        # All ConditionalProtocols associated with a graph are stored and managed by the ConditionalManager.
        # Create a ConditionalManager attachement to the graph if this the first Conditional context to be entered.
        if cls.attachment_name not in node.graph.attachments.keys():
            node.graph.attachments[cls.attachment_name] = ConditionalManager()

        # Push the ConditionalProtocol node to the ConditionalManager
        node.graph.attachments[cls.attachment_name].push(node)

    @classmethod
    def pop_conditional(cls, graph: "Graph") -> None:
        """Pops latest ConditionalProtocol from the ConditionalManager attached to the graph.

        Args:
            graph (Graph): Intervention Graph.
        """
        graph.attachments[cls.attachment_name].pop()

    @classmethod
    def peek_conditional(cls, graph: "Graph") -> "Node":
        """Gets the ConditionalProtocol node of the current Conditional context.

        Args:
            - graph (Graph): Graph.

        Returns:
            Node: ConditionalProtocol of the current Conditional context.
        """
        return graph.attachments[cls.attachment_name].peek()

    @classmethod
    def add_conditioned_node(cls, node: "Node") -> None:
        """Adds a conditioned Node the ConditionalManager attached to its graph.

        Args:
            - node (Node): Conditioned Node.
        """

        node.graph.attachments[cls.attachment_name].add_conditioned_node(node)

    @classmethod
    def is_node_conditioned(cls, node: "Node") -> bool:
        """Checks if the Node is conditoned by the current Conditional context.

        Args:
            - node (Node): Conditioned Node.

        Returns:
            bool: Whether the Node is conditioned.
        """

        return node.graph.attachments[cls.attachment_name].is_node_conditioned(
            node
        )

    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        return {
            "node": {
                "color": "#FF8C00",
                "shape": "polygon",
                "sides": 6,
            },  # Node display
            "label": cls.__name__,
            "arg": defaultdict(
                lambda: {"color": "gray", "shape": "box"}
            ),  # Non-node argument
            "arg_kname": defaultdict(lambda: None),  # Argument label key word
            "edge": defaultdict(lambda: "solid"),
        }  # Argument edge display


class UpdateProtocol(Protocol):
    """Protocol to update the value of an InterventionProxy node.

    .. codeb-block:: python
        with model.trace(input) as tracer:
            num = tracer.apply(int, 0)
            num.update(5)
    """

    @classmethod
    def add(
        cls, node: "Node", new_value: Union[Node, Any]
    ) -> "InterventionProxy":
        """Creates an UpdateProtocol node.

        Args:
            node (Node): Original node.
            new_value (Union[Node, Any]): The update value.

        Returns:
            InterventionProxy: proxy.
        """

        return node.create(
            target=cls,
            proxy_value=node.proxy_value,
            args=[
                node,
                new_value,
            ],
        )

    @classmethod
    def execute(cls, node: "Node") -> None:
        """Sets the value of the original node to the new value.
            If the original is defined outside the context, it uses the bridge to get the node.

        Args:
            node (Node): UpdateProtocol node.
        """

        value_node, new_value = node.args
        new_value = Node.prepare_inputs(new_value)

        if value_node.target == BridgeProtocol:
            value_node._value = new_value
            bridge = BridgeProtocol.get_bridge(value_node.graph)
            lock_node = bridge.id_to_graph[value_node.args[0]].nodes[
                value_node.args[1]
            ]
            value_node = lock_node.args[0]

        value_node._value = new_value

        node.set_value(new_value)

    @classmethod
    def style(cls) -> Dict[str, Any]:
        """Visualization style for this protocol node.

        Returns:
            - Dict: dictionary style.
        """

        return {
            "node": {"color": "blue", "shape": "ellipse"},  # Node display
            "label": cls.__name__,
            "arg": defaultdict(
                lambda: {"color": "gray", "shape": "box"}
            ),  # Non-node argument
            "arg_kname": defaultdict(lambda: None),  # Argument label key word
            "edge": defaultdict(lambda: "solid"),
        }  # Argument edge display
