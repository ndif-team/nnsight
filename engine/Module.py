from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Type, Union

import torch.fx

from . import util
from .fx import Proxy

if TYPE_CHECKING:
    from .Model import Model


@dataclass
class IdxTracker:
    batch_idx: int = 0
    generation_idx: int = 0

    def reset(self):
        self.batch_idx = 0
        self.generation_idx = 0


class Module:
    """_summary_

    Attributes:
        graph (torch.fx.graph.Graph): _description_
        idx_tracker (IdxTracker): _description_
        module_path (str): _description_
        output_shape (torch.Size): _description_
        output_type (Type): _description_
        _output (Proxy): _description_
    """

    def __init__(self, model: "Model") -> None:
        self.model = model

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
                self.model.intervention_graph.call_module(
                    self.module_path, args=args, kwargs=kwds
                )
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
                self.model.intervention_graph.placeholder(
                    f"{self.module_path}.output.{self.model.idx_tracker.generation_idx}.{self.model.idx_tracker.batch_idx}",
                    default_value=self.output_shape,
                )
            )
        return self._output

    @output.setter
    def output(self, value: Union[Proxy, Any]) -> None:
        """Calling denotes the user wishes to set the output of this module and therefore we create a Proxy of that request.
        _output is then set to be the new Proxy.

        Args:
            value (Union[Proxy, Any]): _description_
        """
        node = self.model.intervention_graph.call_function(
            Proxy.proxy_set,
            args=(
                f"{self.module_path}.output.{self.model.idx_tracker.generation_idx}.{self.model.idx_tracker.batch_idx}",
                self.output.node,
                value.node,
            ),
        )
        self._output = Proxy(node, self._output.tracer)

    @staticmethod
    def wrap(module: torch.nn.Module, model: "Model") -> Module:
        """Wraps the torch Module with our Module

        Args:
            module (torch.nn.Module): _description_
            hook (Callable): _description_
            model (Model): _description_

        Returns:
            Module: _description_
        """

        def hook(module: Module, input: Any, output: Any):
            module._output = None
            module.output_shape = util.apply(output, lambda x: x.shape, torch.Tensor)

        for name, _module in module.named_children():
            setattr(module, name, Module.wrap(_module, model))

        if isinstance(module, Module):
            return module

        util.wrap(module, Module, model)

        module.register_forward_hook(hook)

        return module
