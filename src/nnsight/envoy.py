from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from torch._guards import detect_fake_mode

from .contexts.Tracer import Tracer
from .intervention import InterventionProxy


class Envoy:
    """Envoy object act as proxies for torch modules within a model's module tree in order to add nnsight functionality.
    Proxies of the underlying module's output and input are accessed by `.output` and `.input` respectively.

    Attributes:
        _module_path (str): String representing the attribute path of this Envoy's module relative the the root model. Separated by '.' e.x ('transformer.h.0.mlp'). Set by NNsight on initialization of meta model.
        _fake_outputs (List[torch.Tensor]): List of 'meta' tensors built from the outputs most recent _scan. Is list as there can be multiple shapes for a module called more than once.
        _fake_inputs (List[torch.Tensor]): List of 'meta' tensors built from the inputs most recent _scan. Is list as there can be multiple shapes for a module called more than once.
        output (nnsight.intervention.InterventionProxy): Proxy object representing the output of this Envoy's module. Reset on forward pass.
        input (nnsight.intervention.InterventionProxy): Proxy object representing the input of this Envoy's module. Reset on forward pass.
        _call_iter (int): Integer representing the current iteration of this Envoy's module's inputs/outputs.
        _tracer (nnsight.context.Tracer.Tracer): Object which adds this Envoy's module's output and input proxies to an intervention graph. Must be set on Envoys objects manually by the Tracer.
    """

    def __init__(self, module: torch.nn.Module, module_path: str = ""):

        self._module_path = module_path

        self._fake_outputs: List[torch.Tensor] = []
        self._fake_inputs: List[torch.Tensor] = []

        self._output: Optional[InterventionProxy] = None
        self._input: Optional[InterventionProxy] = None

        self._call_iter = 0

        self._tracer: Tracer = None

        self._module = module
        self._sub_envoys: List[Envoy] = []

        # Register hook on underlying module to update the _fake_outputs and _fake_inputs on forward pass.
        self._hook_handle = self._module.register_forward_hook(
            self._hook, with_kwargs=True
        )

        if isinstance(module, torch.nn.ModuleList):
            for i, module in enumerate(self._module):

                envoy = Envoy(module, module_path=f"{self._module_path}.{i}")

                self._sub_envoys.append(envoy)

        else:
            for name, module in self._module.named_children():

                envoy = Envoy(module, module_path=f"{self._module_path}.{name}")

                self._sub_envoys.append(envoy)

                setattr(self, name, envoy)

    def _set_tracer(self, tracer: Tracer, propagate=True):
        """Set tracer object on Envoy.

        Args:
            tracer (Tracer): Tracer to set.
            propagate (bool, optional): If to propagate to all sub-modules. Defaults to True.
        """

        self._tracer = tracer

        if propagate:
            for envoy in self._sub_envoys:
                envoy._set_tracer(tracer, propagate=True)

    def _reset_proxies(self, propagate: bool = True) -> None:
        """Sets proxies to None.

        Args:
            propagate (bool, optional): If to propagate to all sub-modules. Defaults to True.
        """

        self._output: InterventionProxy = None
        self._input: InterventionProxy = None

        if propagate:
            for envoy in self._sub_envoys:
                envoy._reset_proxies(propagate=True)

    def _reset(self, propagate: bool = True) -> None:
        """Sets _call_iter to zero. Calls ._reset_proxies as well.

        Args:
            propagate (bool, optional): If to propagate to all sub-modules. Defaults to True.
        """

        self._reset_proxies(propagate=False)

        self._call_iter = 0

        if propagate:
            for envoy in self._sub_envoys:
                envoy._reset(propagate=True)

    def _clear(self, propagate: bool = True) -> None:
        """Clears _fake_outputs and _fake_inputs. Calls ._reset as well.

        Args:
            propagate (bool, optional): If to propagate to all sub-modules. Defaults to True.
        """

        self._reset(propagate=False)

        self._fake_outputs = []
        self._fake_inputs = []

        if propagate:
            for envoy in self._sub_envoys:
                envoy._clear(propagate=True)

    def _hook(
        self, module: torch.nn.Module, input: Any, input_kwargs: Dict, output: Any
    ):

        if detect_fake_mode(input):

            self._reset_proxies(propagate=False)

            input = (input, input_kwargs)

            self._fake_outputs.append(output)
            self._fake_inputs.append(input)

    def next(self, increment: int = 1, propagate: bool = False) -> Envoy:

        self._call_iter += increment

        self._reset_proxies(propagate=False)

        if propagate:
            for envoy in self._sub_envoys:
                envoy.next(increment=increment, propagate=True)

        return self

    def __str__(self) -> str:
        """Wrapper method for underlying module's string representation.

        Returns:
            str: String.
        """

        return str(self._module)

    def __iter__(self) -> Iterator[Envoy]:
        """Wrapper method for underlying ModuleList iterator.

        Returns:
            Iterator[Envoy]: Iterator.
        """

        return iter(self._sub_envoys)

    def __getitem__(self, key: int) -> Envoy:
        """Wrapper method for underlying ModuleList getitem.

        Args:
            key (int): Key.

        Returns:
            Envoy: Envoy.
        """

        return self._sub_envoys[key]

    def __len__(self) -> int:
        """Wrapper method for underlying ModuleList len.

        Returns:
            int: Length.
        """

        return len(self._module)

    def __getattr__(self, key: str) -> Union[Envoy, Any]:
        """Wrapper method for underlying module's attributes.

        Args:
            key (str): Key.

        Returns:
            Any: Attribute.
        """

        return getattr(self._module, key)

    def __call__(self, *args: List[Any], **kwargs: Dict[str, Any]) -> InterventionProxy:
        """Creates a proxy to call the underlying module's forward method with some inputs.

        Returns:
            InterventionProxy: Module call proxy.
        """

        module_proxy = getattr(self._tracer._graph.module_proxy, self._module_path)

        torch.set_default_device(next(self._module.parameters()).device)

        proxy = module_proxy.forward(*args, **kwargs)

        torch._GLOBAL_DEVICE_CONTEXT.__exit__(None, None, None)

        torch._GLOBAL_DEVICE_CONTEXT = None

        return proxy
    
    torch.set_default_device
    @property
    def output(self) -> InterventionProxy:
        """
        Calling denotes the user wishes to get the output of the underlying module and therefore we create a Proxy of that request.
        Only generates a proxy the first time it is references otherwise return the already set one.

        Returns:
            InterventionProxy: Output proxy.
        """
        if self._output is None:

            if isinstance(self._module, torch.nn.ModuleList):

                self._output = [envoy.output for envoy in self._sub_envoys]

                return self._output

            if len(self._fake_outputs) == 0:
                fake_output = None
            elif self._call_iter >= len(self._fake_outputs):
                # TODO warning?
                fake_output = self._fake_outputs[-1]
            else:
                fake_output = self._fake_outputs[self._call_iter]

            self._output = self._tracer._graph.add(
                value=fake_output,
                target="argument",
                args=[
                    f"{self._module_path}.output",
                    self._tracer._batch_size,
                    self._tracer._batch_start,
                    self._call_iter,
                ],
            )

        return self._output

    @output.setter
    def output(self, value: Union[InterventionProxy, Any]) -> None:
        """
        Calling denotes the user wishes to set the output of the underlying module and therefore we create a Proxy of that request.

        Args:
            value (Union[InterventionProxy, Any]): Value to set output to.
        """

        self._tracer._graph.add(
            target="swap", args=[self.output.node, value], value=True
        )

        self._output = None

    @property
    def input(self) -> InterventionProxy:
        """
        Calling denotes the user wishes to get the input of the underlying module and therefore we create a Proxy of that request.
        Only generates a proxy the first time it is references otherwise return the already set one.

        Returns:
            InterventionProxy: Input proxy.
        """
        if self._input is None:

            if isinstance(self._module, torch.nn.ModuleList):

                self._input = [envoy.input for envoy in self._sub_envoys]

                return self._input

            if len(self._fake_inputs) == 0:
                fake_input = None
            elif self._call_iter >= len(self._fake_inputs):
                # TODO warning?
                fake_input = self._fake_inputs[-1]
            else:
                fake_input = self._fake_inputs[self._call_iter]

            self._input = self._tracer._graph.add(
                value=fake_input,
                target="argument",
                args=[
                    f"{self._module_path}.input",
                    self._tracer._batch_size,
                    self._tracer._batch_start,
                    self._call_iter,
                ],
            )

        return self._input

    @input.setter
    def input(self, value: Union[InterventionProxy, Any]) -> None:
        """
        Calling denotes the user wishes to set the input of the underlying module and therefore we create a Proxy of that request.

        Args:
            value (Union[InterventionProxy, Any]): Value to set input to.
        """

        self._tracer._graph.add(
            target="swap", args=[self.input.node, value], value=True
        )

        self._input = None
