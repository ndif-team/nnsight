from __future__ import annotations
import inspect

from typing import Any, Type, Union

import torch
import torch.fx
from . import util
from .contexts.Generator import Generator
from .fx.Proxy import InterventionProxy


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
        self._graph: torch.fx.Graph = None

        self.generator: Generator = None

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """We override the __call__ function of modules because we may want to do a ModuleIntervention
        i.e call a submodule of the model at a certain point during inference. In the graph model, this function checks to see if any
        of it's arguments are Proxies. If so it thinks were in an Invocation context and should then instead of call the module,
        create a new proxy denoting we eant to call the model with the given input.

        Returns:
            Any: _description_
        """
        adhoc = any(
            isinstance(x, InterventionProxy) for x in list(args) + list(kwds.values())
        )

        if adhoc:
            _get_node = lambda x: x.node

            args = util.apply(args, _get_node, InterventionProxy)
            kwds = util.apply(kwds, _get_node, InterventionProxy)

            return self.generator.tracer.proxy(
                self.generator.tracer.create_node(
                    "call_module", self.module_path, args, kwds
                )
            )

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
            self._output = self.generator.tracer.proxy(
                self.generator.tracer.create_node(
                    "placeholder",
                    f"{self.module_path}.output.{self.generator.generation_idx}.{self.generator.batch_idx}",
                    (
                        util.apply(
                            self.output_shape,
                            lambda x: torch.empty(x, device="meta"),
                            torch.Size,
                        ),
                    ),
                    {},
                )
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
    def graph(self) -> torch.fx.graph.Graph:
        if self._graph is None:
            
            tracer = ModuleTracer()
            # input_proxy = tracer.proxy(
            #     tracer.create_node(
            #         "placeholder",
            #         "input",
                    
            #             util.apply(
            #                 self.input_shape,
            #                 lambda x: torch.empty(x, device="meta"),
            #                 torch.Size,
            #             ),
                    
            #         {},
            #     )
            # )
            # self(input_proxy)
            signature = inspect.signature(self.forward)
            concrete_args =  {
                k: v.default if (i+1) > len(self.input_shape) else torch.empty(self.input_shape[i], device="meta")
                for i, (k, v) in enumerate(signature.parameters.items())
            }
            breakpoint()
            self._graph = tracer.trace(self, concrete_args=concrete_args)
        return self._graph
    
    def get_node(self, name:str) -> torch.fx.node.Node:

        for node in self.graph.nodes:

            if node.name == name:
                return node
            
        return None
    
    def modulize(self, node_name:str, name:str):

        class WrapperModule(Module, torch.nn.Module):

            def __init__(self) -> None:
                torch.nn.Module.__init__(self)
                Module.__init__(self)

            def forward(self, x):

                return x

        node = self.get_node(node_name)

        setattr(self, name, WrapperModule())

        with self.graph.inserting_after(node):

            self.graph.call_module(name, args=(node,))

        util.wrap(self, torch.fx.graph_module.GraphModule, self, self.graph)

        breakpoint()

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
