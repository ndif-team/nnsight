from __future__ import annotations

from typing import Any, Type, Union

import torch

from . import util
from .contexts.Generator import Generator
from .intervention import InterventionProxy
from .fx.Graph import Graph


class Module:
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
        self._graph: Graph = None

        self.generator: Generator = None

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """We override the __call__ function of modules because we may want to do a ModuleIntervention
        i.e call a submodule of the model at a certain point during inference. In the graph model, this function checks to see if any
        of it's arguments are Proxies. If so it thinks were in an Invocation context and should then instead of call the module,
        create a new proxy denoting we eant to call the model with the given input.

        Returns:
            Any: _description_
        """
        proxy = any(
            isinstance(x, InterventionProxy) for x in list(args) + list(kwds.values())
        )

        if proxy:
            module_proxy = getattr(self.generator.graph.module_proxy, self.module_path)

            return module_proxy(*args, **kwds)

        return super().__call__(*args, **kwds)

    @property
    def output(self) -> InterventionProxy:
        """
        Calling denotes the user wishes to get the output of this module and therefore we create a Proxy of that request.
        Only generates a proxy the first time it is references otherwise returnh the already set one.

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
                    f"{self.module_path}.output.{self.generator.generation_idx}.{self.generator.batch_idx}"
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
        self.output.set(value)

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
            hook (Callable): _description_
            invoker_state (InvokerState): _description_

        Returns:
            Module: _description_
        """

        def hook(module: Module, input: Any, output: Any):
            module._output = None
            module.output_shape = util.apply(output, lambda x: x.shape, torch.Tensor)
            module.input_shape = util.apply(input, lambda x: x.shape, torch.Tensor)
            module.output_type = util.apply(output, lambda x: x.dtype, torch.Tensor)
            module.input_type = util.apply(input, lambda x: x.dtype, torch.Tensor)

        for name, _module in module.named_children():
            setattr(module, name, Module.wrap(_module))

        if isinstance(module, Module):
            return module

        util.wrap(module, Module)

        module.register_forward_hook(hook)

        return module
    
