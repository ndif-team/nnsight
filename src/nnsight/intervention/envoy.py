from __future__ import annotations

import inspect
import weakref
import warnings
from contextlib import AbstractContextManager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
from typing_extensions import Self

from . import protocols
from .backends import EditingBackend
from .contexts import InterleavingTracer
from .graph import InterventionNodeType, InterventionProxyType


class Envoy(Generic[InterventionProxyType, InterventionNodeType]):
    """Envoy objects act as proxies for torch modules themselves within a model's module tree in order to add nnsight functionality.
    Proxies of the underlying module's output and input are accessed by `.output` and `.input` respectively.

    Attributes:
        path (str): String representing the attribute path of this Envoy's module relative the the root model. Separated by '.' e.x ('.transformer.h.0.mlp').
        output (nnsight.intervention.InterventionProxy): Proxy object representing the output of this Envoy's module. Reset on forward pass.
        inputs (nnsight.intervention.InterventionProxy): Proxy object representing the inputs of this Envoy's module. Proxy is in the form of (Tuple[Tuple[<Positional arg>], Dict[str, <Keyword arg>]])Reset on forward pass.
        input (nnsight.intervention.InterventionProxy): Alias for the first positional Proxy input i.e Envoy.inputs[0][0]
        iter (nnsight.envoy.EnvoyIterator): Iterator object allowing selection of specific .input and .output iterations of this Envoy.
        _module (torch.nn.Module): Underlying torch module.
        _children (List[Envoy]): Immediate Envoy children of this Envoy.
        _fake_outputs (List[torch.Tensor]): List of 'meta' tensors built from the outputs most recent _scan. Is list as there can be multiple shapes for a module called more than once.
        _fake_inputs (List[torch.Tensor]): List of 'meta' tensors built from the inputs most recent _scan. Is list as there can be multiple shapes for a module called more than once.
        _rename (Optional[Dict[str,str]]): Optional mapping of (old name -> new name).
            For example to rename all gpt 'attn' modules to 'attention' you would: rename={r"attn": "attention"}
            Not this does not actually change the underlying module names, just how you access its envoy. Renaming will replace Envoy.path but Envoy._path represents the pre-renamed true attribute path.
        _tracer (nnsight.context.Tracer.Tracer): Object which adds this Envoy's module's output and input proxies to an intervention graph. Must be set on Envoys objects manually by the Tracer.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        module_path: str = "",
        alias_path: Optional[str] = None,
        rename: Optional[Dict[str, str]] = None,
    ):

        self.path = alias_path or module_path
        self._path = module_path

        self._module: torch.nn.Module = weakref.proxy(module)

        self._rename = rename

        self._iteration_stack = [0]

        self._fake_outputs: List[torch.Tensor] = []
        self._fake_inputs: List[torch.Tensor] = []

        self._output_stack: List[Optional[InterventionProxyType]] = [None]
        self._input_stack: List[Optional[InterventionProxyType]] = [None]

        self._tracer: InterleavingTracer = None

        self._children: List[Envoy] = []

        # Register hook on underlying module to update the _fake_outputs and _fake_inputs on forward pass.
        self._hook_handle = self._module.register_forward_hook(
            self._hook, with_kwargs=True
        )

        # Recurse into PyTorch module tree.
        for name, module in list(self._module.named_children()):

            setattr(self, name, module)

    # Public API ################

    def __call__(
        self, *args: List[Any], hook=False, **kwargs: Dict[str, Any]
    ) -> InterventionProxyType:
        """Creates a proxy to call the underlying module's forward method with some inputs.

        Returns:
            InterventionProxy: Module call proxy.
        """

        if not self._tracing() or self._scanning():
            return self._module(*args, **kwargs)

        if isinstance(self._tracer.backend, EditingBackend):
            hook = True

        return protocols.ApplyModuleProtocol.add(
            self._tracer.graph, self._path, *args, hook=hook, **kwargs
        )

    @property
    def output(self) -> InterventionProxyType:
        """
        Calling denotes the user wishes to get the output of the underlying module and therefore we create a Proxy of that request.
        Only generates a proxy the first time it is references otherwise return the already set one.

        Returns:
            InterventionProxy: Output proxy.
        """
        output = self._output_stack.pop()

        if output is None:

            if isinstance(self._module, torch.nn.ModuleList):
                output = self._tracer.apply(list)
                output.extend([envoy.output for envoy in self._children])
            else:

                iteration = self._iteration_stack[-1]

                if len(self._fake_outputs) == 0:
                    fake_output = inspect._empty
                elif iteration >= len(self._fake_outputs):
                    # TODO warning?
                    fake_output = self._fake_outputs[-1]
                else:
                    fake_output = self._fake_outputs[iteration]

                module_path = f"{self._path}.output"

                output = protocols.InterventionProtocol.add(
                    self._tracer.graph,
                    module_path,
                    self._tracer._invoker_group,
                    iteration,
                    fake_value=fake_output,
                )

        self._output_stack.append(output)

        return output

    @output.setter
    def output(self, value: Union[InterventionProxyType, Any]) -> None:
        """
        Calling denotes the user wishes to set the output of the underlying module and therefore we create a Proxy of that request.

        Args:
            value (Union[InterventionProxy, Any]): Value to set output to.
        """

        protocols.SwapProtocol.add(self.output.node.graph, self.output.node, value)

        self._output_stack[-1] = None

    @property
    def inputs(self) -> InterventionProxyType:
        """
        Calling denotes the user wishes to get the input of the underlying module and therefore we create a Proxy of that request.
        Only generates a proxy the first time it is references otherwise return the already set one.

        Returns:
            InterventionProxy: Input proxy.
        """

        input = self._input_stack.pop()

        if input is None:

            if isinstance(self._module, torch.nn.ModuleList):
                input = self._tracer.apply(list)
                input.extend([envoy.inputs for envoy in self._children])
            else:

                iteration = self._iteration_stack[-1]

                if len(self._fake_inputs) == 0:
                    fake_input = inspect._empty
                elif iteration >= len(self._fake_inputs):
                    # TODO warning?
                    fake_input = self._fake_inputs[-1]
                else:
                    fake_input = self._fake_inputs[iteration]

                module_path = f"{self._path}.input"

                input = protocols.InterventionProtocol.add(
                    self._tracer.graph,
                    module_path,
                    self._tracer._invoker_group,
                    iteration,
                    fake_value=fake_input,
                )

        self._input_stack.append(input)

        return input

    @inputs.setter
    def inputs(self, value: Union[InterventionProxyType, Any]) -> None:
        """
        Calling denotes the user wishes to set the input of the underlying module and therefore we create a Proxy of that request.

        Args:
            value (Union[InterventionProxy, Any]): Value to set input to.
        """

        protocols.SwapProtocol.add(self.inputs.node.graph, self.inputs.node, value)

        self._input_stack[-1] = None

    @property
    def input(self) -> InterventionProxyType:
        """Getting the first positional argument input of the model's module.

        Returns:
            InterventionProxy: Input proxy.
        """

        if isinstance(self._module, torch.nn.ModuleList):
            input = self._tracer.apply(list)
            input.extend([envoy.input for envoy in self._children])

            return input

        return self.inputs[0][0]

    @input.setter
    def input(self, value: Union[InterventionProxyType, Any]) -> None:
        """Setting the value of the input's first positional argument in the model's module.

        Args;
            value (Union[InterventionProxy, Any]): Value to set the input to.
        """

        self.inputs = ((value,) + self.inputs[0][1:],) + (self.inputs[1:])

    @property
    def iter(self) -> IterationEnvoy:

        return IterationEnvoy(self)

    @iter.setter
    def iter(self, iteration: Union[int, List[int], slice]) -> None:
        self._iteration_stack.append(iteration)

    def next(self, increment: int = 1) -> Envoy:
        """By default, this modules inputs and outputs only refer to the first time its called. Use `.next()`to select which iteration .input an .output refer to.

        Args:
            increment (int, optional): How many iterations to jump. Defaults to 1.

        Returns:
            Envoy: Self.
        """

        return self.iter[self._iteration_stack[-1] + increment].__enter__()

    def all(self, propagate: bool = True) -> Envoy:
        """By default, this modules inputs and outputs only refer to the first time its called. Use `.all()`to have .input and .output refer to all iterations.

        Returns:
            Envoy: Self.
        """

        return self.iter[:].__enter__()

    def to(self, *args, **kwargs) -> Envoy:
        """Override torch.nn.Module.to so this returns the Envoy, not the underlying module when doing: model = model.to(...)

        Returns:
            Envoy: Envoy.
        """

        self._module = self._module.to(*args, **kwargs)

        return self

    def modules(
        self,
        include_fn: Callable[[Envoy], bool] = None,
        names: bool = False,
        envoys: List = None,
    ) -> List[Envoy]:
        """Returns all Envoys in the Envoy tree.

        Args:
            include_fn (Callable, optional): Optional function to be ran against all Envoys to check if they should be included in the final collection of Envoys. Defaults to None.
            names (bool, optional): If to include the name/module_path of returned Envoys along with the Envoy itself. Defaults to False.

        Returns:
            List[Envoy]: Included Envoys
        """

        if envoys is None:
            envoys = list()

        included = True

        if include_fn is not None:
            included = include_fn(self)

        if included:
            if names:
                envoys.append((self.path, self))
            else:
                envoys.append(self)

        for sub_envoy in self._children:
            sub_envoy.modules(include_fn=include_fn, names=names, envoys=envoys)

        return envoys

    def named_modules(self, *args, **kwargs) -> List[Tuple[str, Envoy]]:
        """Returns all Envoys in the Envoy tree along with their name/module_path.

        Args:
            include_fn (Callable, optional): Optional function to be ran against all Envoys to check if they should be included in the final collection of Envoys. Defaults to None.

        Returns:
            List[Tuple[str, Envoy]]: Included Envoys and their names/module_paths.
        """

        return self.modules(*args, **kwargs, names=True)

    # Private API ###############################

    def _update(self, module: torch.nn.Module) -> None:
        """Updates the ._model attribute using a new model of the same architecture.
        Used when loading the real weights (dispatching) and need to replace the underlying modules.
        """

        self._hook_handle.remove()

        self._hook_handle = module.register_forward_hook(self._hook, with_kwargs=True)

        i = 0

        for i, child in enumerate(module.children()):

            self._children[i]._update(child)

        # Handle extra modules added after initialization: issues/376
        for name, child in list(self._module.named_children())[i + 1 :]:

            setattr(module, name, child)

        self._module = weakref.proxy(module)

    def _add_envoy(self, module: torch.nn.Module, name: str) -> None:
        """Adds a new Envoy for a given torch module under this Envoy.

        Args:
            module (torch.nn.Module): Module to create Envoy for.
            name (str): name of envoy/attribute.
        """

        alias_path = None

        module_path = f"{self._path}.{name}"

        if self._rename is not None and name in self._rename:

            name = self._rename[name]

            alias_path = f"{self.path}.{name}"

        envoy = Envoy(
            module, module_path=module_path, alias_path=alias_path, rename=self._rename
        )

        self._children.append(envoy)

        setattr(self._module, name, module)

        # If the module already has a sub-module named 'input' or 'output',
        # mount the proxy access to 'nns_input' or 'nns_output instead.
        if hasattr(Envoy, name):

            self._handle_overloaded_mount(envoy, name)

        else:

            super().__setattr__(name, envoy)

    def _handle_overloaded_mount(self, envoy: Envoy, mount_point: str) -> None:
        """If a given module already has an attribute of the same name as something nnsight wants to add, we need to rename it.

        Directly edits the underlying class to accomplish this.

        Args:
            envoy (Envoy): Envoy to handle.
            mount_point (str): Overloaded attribute name.
        """

        warnings.warn(
            f"Module of type `{type(self._module)}` has pre-defined a `{mount_point}` attribute. nnsight access for `{mount_point}` will be mounted at `.nns_{mount_point}` instead of `.{mount_point}` for this module only."
        )

        # If we already shifted a mount point dont create another new class.
        if "Preserved" in self.__class__.__name__:

            new_cls = self.__class__

        else:

            new_cls = type(
                f"{Envoy.__name__}.Preserved",
                (Envoy,),
                {},
            )

        # Get the normal proxy mount point
        mount = getattr(new_cls, mount_point)

        # Move it to nns_<mount point>
        setattr(new_cls, f"nns_{mount_point}", mount)
        # Set the sub-module/envoy to the normal mount point on the CLASS itself not the instance.
        setattr(new_cls, mount_point, envoy)

        # Update the class on the instance
        self.__class__ = new_cls

    def _set_tracer(self, tracer: InterleavingTracer, propagate=True):
        """Set tracer object on Envoy.

        Args:
            tracer (Tracer): Tracer to set.
            propagate (bool, optional): If to propagate to all sub-modules. Defaults to True.
        """

        self._tracer = tracer

        if propagate:
            for envoy in self._children:
                envoy._set_tracer(tracer, propagate=True)

    def _tracing(self) -> bool:
        """Whether or not tracing.

        Returns:
            bool: Is tracing.
        """

        try:

            return self._tracer.graph.alive

        except:

            return False

    def _scanning(self) -> bool:
        """Whether or not in scanning mode. Checks the current Tracer's Invoker.

        Returns:
            bool: Is scanning.
        """

        try:

            return self._tracer.invoker.scanning

        except:

            return False

    def _set_iteration(
        self, iteration: Optional[int] = None, propagate: bool = True
    ) -> None:

        if iteration is not None:
            self._iteration_stack.append(iteration)
            self._output_stack.append(None)
            self._input_stack.append(None)
        else:
            self._iteration_stack.pop()
            self._output_stack.pop()
            self._input_stack.pop()

        if propagate:
            for envoy in self._children:
                envoy._set_iteration(iteration, propagate=True)

    def _reset_proxies(self, propagate: bool = True) -> None:
        """Sets proxies to None.

        Args:
            propagate (bool, optional): If to propagate to all sub-modules. Defaults to True.
        """

        self._output_stack = [None]
        self._input_stack = [None]

        if propagate:
            for envoy in self._children:
                envoy._reset_proxies(propagate=True)

    def _reset(self, propagate: bool = True) -> None:
        """Sets _call_iter to zero. Calls ._reset_proxies as well.

        Args:
            propagate (bool, optional): If to propagate to all sub-modules. Defaults to True.
        """

        self._reset_proxies(propagate=False)

        self._iteration_stack = [0]

        if propagate:
            for envoy in self._children:
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
            for envoy in self._children:
                envoy._clear(propagate=True)

    def _hook(
        self,
        module: torch.nn.Module,
        input: Any,
        input_kwargs: Dict,
        output: Any,
    ):

        if self._scanning():

            input = (input, input_kwargs)

            self._fake_outputs.append(output)
            self._fake_inputs.append(input)

    def _repr_module_list(self):

        list_of_reprs = [repr(item) for item in self._children]
        if len(list_of_reprs) == 0:
            return self._module._get_name() + "()"

        start_end_indices = [[0, 0]]
        repeated_blocks = [list_of_reprs[0]]
        for i, r in enumerate(list_of_reprs[1:], 1):
            if r == repeated_blocks[-1]:
                start_end_indices[-1][1] += 1
                continue

            start_end_indices.append([i, i])
            repeated_blocks.append(r)

        lines = []
        main_str = self._module._get_name() + "("
        for (start_id, end_id), b in zip(start_end_indices, repeated_blocks):
            local_repr = f"({start_id}): {b}"  # default repr

            if start_id != end_id:
                n = end_id - start_id + 1
                local_repr = f"({start_id}-{end_id}): {n} x {b}"

            local_repr = torch.nn.modules.module._addindent(local_repr, 2)
            lines.append(local_repr)

        main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    def __repr__(self) -> str:
        """Wrapper method for underlying module's string representation.

        Returns:
            str: String.
        """

        if isinstance(self._module, torch.nn.ModuleList):

            return self._repr_module_list()

        extra_lines = []
        extra_repr = self._module.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for attribute_name, attribute in self.__dict__.items():

            if attribute_name == "_tracer":
                continue

            if isinstance(attribute, Envoy):

                mod_str = repr(attribute)
                mod_str = torch.nn.modules.module._addindent(mod_str, 2)
                child_lines.append("(" + attribute_name + "): " + mod_str)

        lines = extra_lines + child_lines

        main_str = self._module._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"

        return main_str

    def __iter__(self) -> Iterator[Envoy[InterventionProxyType, InterventionNodeType]]:
        """Wrapper method for underlying ModuleList iterator.

        Returns:
            Iterator[Envoy]: Iterator.
        """

        return iter(self._children)

    def __getitem__(
        self, key: int
    ) -> Envoy[InterventionProxyType, InterventionNodeType]:
        """Wrapper method for underlying ModuleList getitem.

        Args:
            key (int): Key.

        Returns:
            Envoy: Envoy.
        """

        return self._children[key]

    def __len__(self) -> int:
        """Wrapper method for underlying ModuleList len.

        Returns:
            int: Length.
        """

        return len(self._module)

    def __getattr__(
        self, key: str
    ) -> Union[
        Envoy[InterventionProxyType, InterventionNodeType], InterventionProxyType, Any
    ]:
        """Wrapper method for underlying module's attributes.
        If the attribute is a tensor (e.g. weights or bias) and accessed during tracing, then an InterventionProxy is created.

        Args:
            key (str): Key.

        Returns:
            Union[InterventionProxyType, Any]: Attribute.
        """

        attr = getattr(self._module, key)

        if self._tracing() and isinstance(attr, torch.Tensor):
            attr_proxy = protocols.ParameterProtocol.add(
                self._tracer.graph, self._path, key
            )

            return attr_proxy

        return attr

    def __setattr__(self, key: Any, value: Any) -> None:
        """Overload setattr to create and set an Envoy when trying to set a torch Module."""

        if key != "_module" and isinstance(value, torch.nn.Module):

            self._add_envoy(value, key)

        else:

            super().__setattr__(key, value)


class IterationEnvoy(Envoy, AbstractContextManager):

    def __init__(self, envoy: Envoy) -> None:

        self.__dict__.update(envoy.__dict__)

        self._iteration = self._iteration_stack[-1]

        self._open_context = False

    @property
    def output(self) -> InterventionProxyType:

        self._output_stack.append(None)
        self._iteration_stack.append(self._iteration)

        output = super().output

        self._output_stack.pop()
        self._iteration_stack.pop()

        return output

    @property
    def input(self) -> InterventionProxyType:

        self._input_stack.append(None)
        self._iteration_stack.append(self._iteration)

        input = super().input

        self._input_stack.pop()
        self._iteration_stack.pop()

        return input

    def __getitem__(self, key: Union[int, List[int], slice]) -> Self:

        # TODO: Error if not valid key type

        if isinstance(key, tuple):

            key = list(key)

        self._iteration = key

        return self

    def __enter__(self) -> IterationEnvoy:

        if not self._open_context:

            self._set_iteration(self._iteration)

        self._open_context = True

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:

        self._set_iteration()

        self._open_context = False

        if isinstance(exc_val, BaseException):
            raise exc_val
