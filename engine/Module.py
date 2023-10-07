from __future__ import annotations

from typing import Any, Type, Union

import torch

from . import util
from .contexts.Generator import Generator
from .fx.Graph import Graph
from .fx.Node import Node
from .intervention import InterventionProxy


class Module(torch.nn.Module):
    """_summary_

    Attributes:
        generator (Generator): _description_
        module_path (str): _description_
        output_shape (torch.Size): _description_
        output_type (Type): _description_
        _output (Proxy): _description_
    """

    def __init__(self) -> None:
        self.module_path: str = None
        self.input_shape: torch.Size = None
        self.input_type: Type = None
        self.output_shape: torch.Size = None
        self.output_type: Type = None

        self._output: InterventionProxy = None
        self._input: InterventionProxy = None
        self._graph: Graph = None

        self.generator: Generator = None

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Override __call__ to check for InterventionProxy arguments. If there are any, we should return an
        InterventionProxy denoting we want to call the given module with arguments.

        Returns:
            Any: _description_
        """
        proxy = any(
            isinstance(x, InterventionProxy) for x in list(args) + list(kwds.values())
        )

        if proxy:
            module_proxy = getattr(self.generator.graph.module_proxy, self.module_path)

            return module_proxy.forward(*args, **kwds)

        return super().__call__(*args, **kwds)

    @property
    def output(self) -> InterventionProxy:
        """
        Calling denotes the user wishes to get the output of this module and therefore we create a Proxy of that request.
        Only generates a proxy the first time it is references otherwise return the already set one.

        Returns:
            Proxy: _description_
        """
        if self._output is None:
            self._output = self.generator.graph.add(
                graph=self.generator.graph,
                value=util.apply(
                    self.output_shape,
                    lambda x: torch.empty(x, device="meta"),
                    torch.Size,
                ),
                target="argument",
                args=[
                    f"{self.module_path}.output.{self.generator.generation_idx}",
                    self.generator.batch_size,
                    len(self.generator.prompts) - self.generator.batch_size,
                ],
            )

        return self._output

    @output.setter
    def output(self, value: Union[InterventionProxy, Any]) -> None:
        """
        Calling denotes the user wishes to set the output of this module and therefore we create a Proxy of that request.

        Args:
            value (Union[Proxy, Any]): _description_
        """

        Node.update(
            self.output.node.proxy_value, self.output.node.prepare_proxy_values(value)
        )

        self.output.node.graph.add(
            graph=self.output.node.graph,
            value=self.output.node.proxy_value,
            target=Node.update,
            args=[self.output.node, value],
        )
    
    @property
    def input(self) -> InterventionProxy:
        """
        Calling denotes the user wishes to get the input of this module and therefore we create a Proxy of that request.
        Only generates a proxy the first time it is references otherwise return the already set one.

        Returns:
            Proxy: _description_
        """
        if self._input is None:
            self._input = self.generator.graph.add(
                graph=self.generator.graph,
                value=util.apply(
                    self.input_shape,
                    lambda x: torch.empty(x, device="meta"),
                    torch.Size,
                ),
                target="argument",
                args=[
                    f"{self.module_path}.input.{self.generator.generation_idx}",
                    self.generator.batch_size,
                    len(self.generator.prompts) - self.generator.batch_size,
                ],
            )

        return self._input

    @input.setter
    def input(self, value: Union[InterventionProxy, Any]) -> None:
        """
        Calling denotes the user wishes to set the input of this module and therefore we create a Proxy of that request.

        Args:
            value (Union[Proxy, Any]): _description_
        """

        Node.update(
            self.input.node.proxy_value, self.input.node.prepare_proxy_values(value)
        )

        self.input.node.graph.add(
            graph=self.input.node.graph,
            value=self.input.node.proxy_value,
            target=Node.update,
            args=[self.input.node, value],
        )

    @property
    def graph(self) -> Graph:
        if self._graph is None:
            self._graph = Graph.trace(
                self,
                *util.apply(
                    self.input_shape,
                    lambda x: torch.empty(x, device="meta"),
                    torch.Size,
                ),
            )

        return self._graph

    @staticmethod
    def wrap(module: torch.nn.Module) -> Module:
        """Wraps the torch Module with our Module

        Args:
            module (torch.nn.Module): _description_

        Returns:
            Module: _description_
        """

        def hook(module: Module, input: Any, output: Any):
            module._output = None
            module._input = None
            module.output_shape = util.apply(output, lambda x: x.shape, torch.Tensor)
            module.input_shape = util.apply(input, lambda x: x.shape, torch.Tensor)
            module.output_type = util.apply(output, lambda x: x.dtype, torch.Tensor)
            module.input_type = util.apply(input, lambda x: x.dtype, torch.Tensor)

        for name, _module in module.named_children():
            setattr(module, name, Module.wrap(_module))

        if isinstance(module, (Module, torch.nn.ModuleList)):
            return module

        util.wrap(module, Module)

        module.register_forward_hook(hook)

        return module
