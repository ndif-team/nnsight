"""The Module object acts as the entrypoint into the interwoven intervention graph. Through it, one can manipulate the inputs and outputs of Modules as a model is running. Specifically, the ``.output`` and ``.input`` attributes are the root nodes of the intervention graph, and all operations performed on these Proxy objects are the downstream nodes that populate the graph. Requires a ``Tracer`` object to add the correct proxies to the intervention graph. 

    Examples:

        The example below accesses the inputs and outputs of a gpt2 module, saves them during execution, and prints the resulting values:
        
        .. code-block:: python

            with model.generate() as generator:
                with generator.invoke('The Eiffel Tower is in the city of') as invoker:
                    hidden_states = model.lm_head.input.save()
                    logits = model.lm_head.output.save()

            print(hidden_states.value)
            print(logits.value)

        This example shows how to set the output of a gpt2 module to zero:

        .. code-block:: python

            with model.generate() as generator:
                with generator.invoke('The Eiffel Tower is in the city of') as invoker:
                    model.transformer.h[0].output[0] = 0
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import torch

from . import util
from .contexts.Tracer import Tracer
from .intervention import InterventionProxy
from .tracing.Graph import Graph
from .tracing.Proxy import Proxy


class Module(torch.nn.Module):
    """Class that wraps existing torch modules within a model's module tree in order to add nnsight functionality.
    Proxies of it's output and input are accessed by `.output` and `.input` respectively.

    Attributes:
        module_path (str): String representing the attribute path of this module relative the the root model. Separated by '.' e.x ('transformer.h.0.mlp'). Set by AbstractModel on initialization of meta model.
        output (nnsight.intervention.InterventionProxy): Proxy object representing the output of this module. Reset on pass through.
        output_shape (torch.Size): Shape of the tensor outputs to this module. Populated by most recent pass through. Can also be a nested list of torch.Size.
        output_type (torch.dtype): Dtype of the tensor outputs to this module. Populated by most recent pass through. Can also be a nested list of torch.dtype.
        output (nnsight.intervention.InterventionProxy): Proxy object representing the output of this module. Reset on pass through.
        input_shape (torch.Size): Shape of the tensor inputs to this module. Populated by most recent pass through. Can also be a nested list of torch.Size.
        input_type (torch.dtype): Dtype of the tensor inputs to this module. Populated by most recent pass through. Can also be a nested list of torch.dtype.
        tracer (nnsight.context.Tracer.Tracer): Object which adds this module's output and input proxies to an intervention graph. Must be set on Module objects manually.
    """

    def __init__(self) -> None:
        self.module_path: str = None
        self.input_shape: torch.Size = None
        self.input_type: torch.dtype = None
        self.output_shape: torch.Size = None
        self.output_type: torch.dtype = None

        self._output: InterventionProxy = None
        self._input: InterventionProxy = None
        self._backward_output: InterventionProxy = None
        self._backward_input: InterventionProxy = None
        self._graph: Graph = None

        self.tracer: Tracer = None

    def __call__(
        self, *args: List[Any], **kwds: Dict[str, Any]
    ) -> Union[Any, InterventionProxy]:
        """Override __call__ to check for InterventionProxy arguments.
        If there are any, we should return an InterventionProxy denoting we want to call the given module with arguments.

        Args:
            args (List[Any]): Positional arguments.
            kwargs (Dict[str,Any]): Keyword arguments

        Returns:
            Union[Any,InterventionProxy]: Either the output of the module if not tracing or a Proxy if tracing.
        """
        proxy = any(
            isinstance(x, InterventionProxy) for x in list(args) + list(kwds.values())
        )

        if proxy:
            module_proxy = getattr(self.tracer.graph.module_proxy, self.module_path)

            return module_proxy.forward(*args, **kwds)

        return super().__call__(*args, **kwds)

    @property
    def output(self) -> InterventionProxy:
        """
        Calling denotes the user wishes to get the output of this module and therefore we create a Proxy of that request.
        Only generates a proxy the first time it is references otherwise return the already set one.

        Returns:
            Proxy: Output proxy.
        """
        if self._output is None:
            self._output = self.tracer.graph.add(
                value=util.apply(
                    self.output_shape,
                    lambda x: torch.empty(x, device="meta", requires_grad=True).clone(),
                    torch.Size,
                ),
                target="argument",
                args=[
                    f"{self.module_path}.output.{self.tracer.generation_idx}",
                    self.tracer.batch_size,
                    len(self.tracer.batched_input) - self.tracer.batch_size,
                ],
            )

        return self._output

    @output.setter
    def output(self, value: Union[InterventionProxy, Any]) -> None:
        """
        Calling denotes the user wishes to set the output of this module and therefore we create a Proxy of that request.

        Args:
            value (Union[Proxy, Any]): Value to set output to.
        """

        self.output.node.graph.add(
            target=Proxy.proxy_update,
            args=[self.output.node, value],
        )

    @property
    def input(self) -> InterventionProxy:
        """
        Calling denotes the user wishes to get the input of this module and therefore we create a Proxy of that request.
        Only generates a proxy the first time it is references otherwise return the already set one.

        Returns:
            Proxy: Input proxy.
        """
        if self._input is None:
            self._input = self.tracer.graph.add(
                value=util.apply(
                    self.input_shape,
                    lambda x: torch.empty(x, device="meta", requires_grad=True).clone(),
                    torch.Size,
                ),
                target="argument",
                args=[
                    f"{self.module_path}.input.{self.tracer.generation_idx}",
                    self.tracer.batch_size,
                    len(self.tracer.batched_input) - self.tracer.batch_size,
                ],
            )

        return self._input

    @input.setter
    def input(self, value: Union[InterventionProxy, Any]) -> None:
        """
        Calling denotes the user wishes to set the input of this module and therefore we create a Proxy of that request.

        Args:
            value (Union[Proxy, Any]): Value to set input to.
        """

        self.input.node.graph.add(
            target=Proxy.proxy_update,
            args=[self.input.node, value],
        )

    @property
    def backward_output(self) -> InterventionProxy:
        """
        Calling denotes the user wishes to get the backward_output of this module and therefore we create a Proxy of that request.
        Only generates a proxy the first time it is references otherwise return the already set one.

        Returns:
            Proxy: backward_output proxy.
        """
        if self._backward_output is None:
            self._backward_output = self.tracer.graph.add(
                value=util.apply(
                    self.output_shape,
                    lambda x: torch.empty(x, device="meta"),
                    torch.Size,
                ),
                target="argument",
                args=[
                    f"{self.module_path}.backward_output.{self.tracer.generation_idx}",
                    self.tracer.batch_size,
                    len(self.tracer.batched_input) - self.tracer.batch_size,
                ],
            )

        return self._backward_output

    @backward_output.setter
    def backward_output(self, value: Union[InterventionProxy, Any]) -> None:
        """
        Calling denotes the user wishes to set the backward_output of this module and therefore we create a Proxy of that request.

        Args:
            value (Union[Proxy, Any]): Value to set backward_output to.
        """

        self.backward_output.node.graph.add(
            target=Proxy.proxy_update,
            args=[self.backward_output.node, value],
        )

    @property
    def backward_input(self) -> InterventionProxy:
        """
        Calling denotes the user wishes to get the backward_input of this module and therefore we create a Proxy of that request.
        Only generates a proxy the first time it is references otherwise return the already set one.

        Returns:
            Proxy: backward_input proxy.
        """
        if self._backward_input is None:
            self._backward_input = self.tracer.graph.add(
                value=util.apply(
                    self.input_shape,
                    lambda x: torch.empty(x, device="meta"),
                    torch.Size,
                ),
                target="argument",
                args=[
                    f"{self.module_path}.backward_input.{self.tracer.generation_idx}",
                    self.tracer.batch_size,
                    len(self.tracer.batched_input) - self.tracer.batch_size,
                ],
            )

        return self._backward_input

    @backward_input.setter
    def backward_input(self, value: Union[InterventionProxy, Any]) -> None:
        """
        Calling denotes the user wishes to set the backward_input of this module and therefore we create a Proxy of that request.

        Args:
            value (Union[Proxy, Any]): Value to set backward_input to.
        """

        self.backward_input.node.graph.add(
            target=Proxy.proxy_update,
            args=[self.backward_input.node, value],
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
        """Wraps the torch Module with our Module class. Adds a forward hook to the module to populate output and input data. Does not wrap torch.nn.ModuleList.

        Args:
            module (torch.nn.Module): The torch Module to wrap.

        Returns:
            Module: The wrapped Module.
        """

        def hook(module: Module, input: Any, output: Any):
            module._output = None
            module._input = None
            module._backward_output = None
            module._backward_input = None
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
