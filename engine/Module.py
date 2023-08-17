from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Type, Union

import torch.fx

from . import util
from .fx import Proxy
from.Invoker import InvokerState

class Module:
    """_summary_

    Attributes:
        invoker_state (InvokerState): _description_
        module_path (str): _description_
        output_shape (torch.Size): _description_
        output_type (Type): _description_
        _output (Proxy): _description_
    """

    def __init__(self, invoker_state: "InvokerState") -> None:
        self.invoker_state = invoker_state

        self.module_path: str = None
        self.output_shape: torch.Size = None
        self.output_type: Type = torch.Tensor

        self._output: Proxy = None

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """We override the __call__ function of modules because we may want to do a ModuleIntervention
        i.e call a submodule of the model at a certain point during inference. In the graph model, this function checks to see if any
        of it's arguments are Proxies. If so it thinks were in an Invocation context and should then instead of call the module,
        create a new proxy denoting we eant to call the model with the given input.

        Returns:
            Any: _description_
        """
        adhoc = any(isinstance(x, Proxy) for x in list(args) + list(kwds.values()))

        if adhoc:
            _get_node = lambda x: x.node

            args = util.apply(args, _get_node, Proxy)
            kwds = util.apply(kwds, _get_node, Proxy)

            return Proxy(
                self.invoker_state.tracer.create_node(
                    "call_module", self.module_path, args, kwds
                ),
                self.invoker_state.tracer
            )

        return super().__call__(*args, **kwds)

    @property
    def output(self) -> Proxy:
        """
        Calling denotes the user wishes to get the output of this module and therefore we create a Proxy of that request.
        Only generates a proxy the first time it is references otherwise returnh the already set one.

        Returns:
            Proxy: _description_
        """
        if self._output is None:
            self._output = Proxy(
                self.invoker_state.tracer.create_node(
                    "placeholder",
                    f"{self.module_path}.output.{self.invoker_state.generation_idx}.{self.invoker_state.batch_idx}",
                    (util.apply(self.output_shape, lambda x : torch.empty(x, device="meta"), torch.Size),),
                    {},
                ),
                self.invoker_state.tracer,
            )

        return self._output

    @output.setter
    def output(self, value: Union[Proxy, Any]) -> None:
        """Calling denotes the user wishes to set the output of this module and therefore we create a Proxy of that request.
        _output is then set to be the new Proxy.

        Args:
            value (Union[Proxy, Any]): _description_
        """

        self._output = Proxy(
            self.invoker_state.tracer.create_node(
                "call_function",
                Proxy.proxy_set,
                (
                    f"{self.module_path}.output.{self.invoker_state.generation_idx}.{self.invoker_state.batch_idx}",
                    self.output.node,
                    value.node,
                ),
                {},
            ),
            self.invoker_state.tracer,
        )

    @staticmethod
    def wrap(module: torch.nn.Module, invoker_state: "InvokerState") -> Module:
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

        for name, _module in module.named_children():
            setattr(module, name, Module.wrap(_module, invoker_state))

        if isinstance(module, Module):
            return module

        util.wrap(module, Module, invoker_state)

        module.register_forward_hook(hook)

        return module
