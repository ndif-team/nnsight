from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch.fx

from . import util


class Proxy(torch.futures.Future, torch.fx.Proxy):
    """
    We extend the Proxy class here for a few reasons.
    We add the .save() method to denote were creating a save Node for a future Intervention.
    We add torch.future.Future as a super class so we can set the value after Inference.
    We add add save proxies to Proxy.save_proxies to refernce and set their value after Inference.
    """

    save_proxies: Dict[str, Proxy] = dict()

    @staticmethod
    def reset() -> None:
        """Reset promises"""
        Proxy.save_proxies.clear()

    @staticmethod
    def proxy_set(
        module_path: str,
        activation_node: torch.fx.node.Node,
        value: Union[torch.fx.node.Node, Any],
    ) -> None:
        """
        Shell function to capture when were setting the value of a module output.
        activation_node is not used but it is an argument here so it is added to its dependencies
        (dont want to set the module output until weve actually arrived at the module).

        Args:
            module_path (str): _description_
            activation_node (torch.fx.node.Node): _description_
            value (Union[torch.fx.node.Node, Any]): _description_
        """
        pass

    @staticmethod
    def proxy_save(node: torch.fx.node.Node) -> None:
        """Shell function to capture when were saving a value during Inference.

        Args:
            node (torch.fx.node.Node): _description_
        """
        pass

    def __init__(self, *args, **kwargs):
        torch.fx.Proxy.__init__(self, *args, **kwargs)
        torch.futures.Future.__init__(self)

    def save(self) -> Proxy:
        """Creates a save proxy and adds it to Proxy.save_proxies."""
        proxy = Proxy(
            self.node.graph.call_function(Proxy.proxy_save, args=(self.node,)),
            self.tracer,
        )
        Proxy.save_proxies[proxy.node.name] = proxy
        return proxy
    
    @property
    def shape(self) -> torch.Size:
        return self.node.shape


class Node(torch.fx.node.Node):
    """
    We extend the Node class here for two reasons.
    One, so we can keep track of the shape of data as it flows through the graph so users can see the shape.
    Two, we call get_shape which actually runs the node command on meta data to compute the shape and validate the command
    in terms of shapes and data types.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.shape = self.get_shape()

    def get_meta(self) -> torch.Tensor:
        """Return a meta tensor with shape of this nodes computed output shape.

        Returns:
            torch.Tensor: _description_
        """
        return (
            torch.empty(self.shape, device="meta") if self.shape is not None else None
        )

    def prepare_inputs(self) -> Tuple[List[Any], Dict[str, Any]]:
        """Preprocess this nodes input to be ran by its command.

        Returns:
            Tuple[List[Any], Dict[str, Any]]: _description_
        """

        def _prepare(node: Node):
            return node.get_meta()

        def _to(value: torch.Tensor):
            return value.to("meta")

        # Turn nodes into their meta tensors
        args = util.apply(self.args, _prepare, Node)
        kwargs = util.apply(self.kwargs, _prepare, Node)
        # Move tensors to meta device
        args = util.apply(args, _to, torch.Tensor)
        kwargs = util.apply(kwargs, _to, torch.Tensor)

        return args, kwargs

    def get_shape(self) -> torch.Size:
        """Runs this nodes comannd with this nodes inputs and return the shape of the output.

        Raises:
            ValueError: _description_

        Returns:
            torch.Size: _description_
        """
        args, kwargs = self.prepare_inputs()

        # A placeholder in our context is the output from a module during inference.
        if self.op == "placeholder":
            # Just get the meta tensor of the module output shape. We use the default_value kwarg to store this shape.
            result = torch.empty(args[0], device="meta")
        elif self.op == "get_attr":
            # TODO ?
            result = None
        elif self.op == "call_function":
            # If were saving a value, the saved value will be the same shape as the old.
            if self.target == Proxy.proxy_save:
                result = args[0]
            # Were setting the value of this Node to the value of the last arg which is the value. So use it's shape.
            elif self.target == Proxy.proxy_set:
                result = args[-1]
            elif self.target == getattr:
                # I believe somewhere above the node creation calls something catches AttributeError and tries to create the node again.
                # For us leading to maximum recursion error so we catch and raise a ValueError
                try:
                    result = self.target(*args, **kwargs)
                except AttributeError as e:
                    raise ValueError(
                        f"'{args[0].__class__.__name__}' object has no attribute '{args[1]}'"
                    )
            else:
                # Just call the function with args.
                result = self.target(*args, **kwargs)
        elif self.op == "call_method":
            # TODO
            self_obj, *args = load_arg(self.args)
            kwargs = load_arg(self.kwargs)
            result = getattr(self_obj, self.target)(*args, **kwargs)
        elif self.op == "call_module":
            # graph.owning_module should be the model. fetch_attr return the submodule specified by the module_path.
            # Then all the nodule with args.
            result = util.fetch_attr(self.graph.owning_module, self.target)(
                *args, **kwargs
            )
        else:
            result = None

        # Get shape of all meta tensors
        return util.apply(result, lambda x: x.shape, torch.Tensor)


# Set torch classes to our classes to account for other ways classes are created
torch.fx.node.Node = Node
torch.fx.graph.Node = Node
torch.fx.proxy.Proxy = Proxy
# When we use graph.eliminate_dead_code(), we want proxy_set and proxy_save and their dependencies to not be removed
torch.fx.node._side_effectful_functions.update(set([Proxy.proxy_set, Proxy.proxy_save]))
