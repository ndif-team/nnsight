from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import torch.fx
from torch.fx.graph import Graph

from . import logger, util


class Proxy(torch.futures.Future, torch.fx.Proxy):
    """
    We extend the Proxy class here for a couple reasons.
    We add the .save() method to denote were creating a save Node for a future Intervention and save a reference to it.
    We add torch.future.Future as a super class so we can set the value after Inference.
    """

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
            self.tracer.create_node(
                "call_function", Proxy.proxy_save, (self.node,), {}
            ),
            self.tracer,
        )
        self.tracer.save_proxies[proxy.node.name] = proxy
        return proxy

    @property
    def shape(self) -> torch.Size:
        """_summary_

        Returns:
            torch.Size: _description_
        """
        return self.tracer.node_name_to_shape[self.node.name]
    

    def token(self, idx:int) -> Proxy:

        if idx >= 0:
            n_tokens = self.shape[1]
            idx = -(n_tokens - idx)

        return self[:,idx]
    
    def t(self, idx:int) -> Proxy:

        return self.token(idx)


class Tracer(torch.fx.proxy.GraphAppendingTracer):
    """
    We extend the base Tracer class used by Graph and Node creating to keep track of the output shape
    for each node as well as track all Proxy.proxy_save function calls so we can set the value of them later.

    Args:
        torch (_type_): _description_
    """

    def __init__(self, graph: Graph):
        super().__init__(graph)

        self.node_name_to_shape = dict()
        self.save_proxies = dict()

    def create_node(self, *args, **kwargs) -> torch.fx.Node:
        """
        Overrides Tracer.create_node so everytime a node is created, we log the node
        and run get_shape on the node ot both run the command with meta tensors defined
        by the node to see if it actually works as well as save the shape of the result.

        Returns:
            torch.fx.Node: _description_
        """
        node = super().create_node(*args, **kwargs)

        if node.op != "root":
            logger.debug(f"=> Proxy({node.name})")

            self.node_name_to_shape[node.name] = self.get_shape(node)

        return node

    def get_meta(self, node: torch.fx.node.Node) -> torch.Tensor:
        """Return a meta tensor with shape of this nodes computed output shape.

        Returns:
            torch.Tensor: _description_
        """
        shape = self.node_name_to_shape.get(node.name, None)
        return torch.empty(shape, device="meta") if shape is not None else None

    def prepare_inputs(
        self, node: torch.fx.node.Node
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Preprocess this nodes input to be ran by its command

        Args:
            node (torch.fx.node.Node): _description_

        Returns:
            Tuple[List[Any], Dict[str, Any]]: _description_
        """

        def _prepare(node: torch.fx.Node):
            return self.get_meta(node)

        def _to(value: torch.Tensor):
            return value.to("meta")

        # Turn nodes into their meta tensors
        args = util.apply(node.args, _prepare, torch.fx.Node)
        kwargs = util.apply(node.kwargs, _prepare, torch.fx.Node)
        # Move tensors to meta device
        args = util.apply(args, _to, torch.Tensor)
        kwargs = util.apply(kwargs, _to, torch.Tensor)

        return args, kwargs

    def get_shape(self, node: torch.fx.node.Node) -> torch.Size:
        """
        Runs this nodes comannd with this nodes inputs and return the shape of the output

        Args:
            node (torch.fx.node.Node): _description_

        Raises:
            ValueError: _description_

        Returns:
            torch.Size: _description_
        """
        args, kwargs = self.prepare_inputs(node)

        # A placeholder in our context is the output from a module during inference.
        if node.op == "placeholder":
            # Just get the meta tensor of the module output shape. We use the default_value kwarg to store this shape.
            result = torch.empty(args[0], device="meta")
        elif node.op == "get_attr":
            # TODO ?
            result = None
        elif node.op == "call_function":
            # If were saving a value, the saved value will be the same shape as the old.
            if node.target == Proxy.proxy_save:
                result = args[0]
            # Were setting the value of this Node to the value of the last arg which is the value. So use it's shape.
            elif node.target == Proxy.proxy_set:
                result = args[-1]
            elif node.target == getattr:
                # I believe somewhere above the node creation calls something catches AttributeError and tries to create the node again.
                # For us leading to maximum recursion error so we catch and raise a ValueError
                try:
                    result = node.target(*args, **kwargs)
                except AttributeError as e:
                    raise ValueError(
                        f"'{args[0].__class__.__name__}' object has no attribute '{args[1]}'"
                    )
            else:
                # Just call the function with args.
                result = node.target(*args, **kwargs)
        elif node.op == "call_method":
            # TODO
            self_obj, *args = load_arg(self.args)
            kwargs = load_arg(self.kwargs)
            result = getattr(self_obj, self.target)(*args, **kwargs)
        elif node.op == "call_module":
            # graph.owning_module should be the model. fetch_attr return the submodule specified by the module_path.
            # Then all the nodule with args.
            result = util.fetch_attr(self.graph.owning_module, node.target)(
                *args, **kwargs
            )
        else:
            result = None

        # Get shape of all meta tensors
        return util.apply(result, lambda x: x.shape, torch.Tensor)


# Set torch classes to our classes to account for other ways classes are created
torch.fx.proxy.Proxy = Proxy
# When we use graph.eliminate_dead_code(), we want proxy_set and proxy_save and their dependencies to not be removed
torch.fx.node._side_effectful_functions.update(set([Proxy.proxy_set, Proxy.proxy_save]))
