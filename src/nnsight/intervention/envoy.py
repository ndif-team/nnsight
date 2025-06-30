from __future__ import annotations

import inspect
import os
import warnings
from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, MethodType
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.nn.modules.module import _addindent

from .. import CONFIG, base_deprecation_message, deprecated, util
from ..util import apply
from . import serialization
from .batching import Batchable
from .inject import convert as inject
from .tracing.base import Tracer, WithBlockNotFoundError
from .tracing.editing import EditingTracer
from .tracing.globals import Object
from .tracing.iterator import IteratorProxy
from .tracing.tracer import InterleavingTracer, ScanningTracer

if TYPE_CHECKING:
    from .interleaver import Interleaver
else:
    Interleaver = Any


class Envoy(Batchable):
    """
    A proxy class that wraps a PyTorch module to enable intervention during execution.

    This class provides access to module inputs and outputs during forward passes,
    and allows for modification of these values through an interleaving mechanism.
    It serves as the primary interface for inspecting and modifying the behavior
    of neural network modules during execution.

    Attributes:
        path (str): The module's location in the model hierarchy.
            Example: "model.encoder.layer1" indicates this module is the first layer of the encoder in the model.
        _module (torch.nn.Module): The underlying PyTorch module
        _source (Optional[EnvoySource]): Source code representation of the module
        _interleaver (Optional[Interleaver]): Interleaver for managing execution flow
        _default_mediators (List[List[str]]): List of default mediators created with .edit
        _children (List[Envoy]): List of child Envoys
        _alias (Aliaser): Aliaser object for managing aliases
    """

    def __init__(
        self,
        module: torch.nn.Module,
        interleaver: Optional[Interleaver] = None,
        path: Optional[str] = "model",
        rename: Optional[Dict[str, Union[str, List[str]]]] = None,
    ) -> None:
        """
        Initialize an Envoy for a PyTorch module.

        Args:
            module (torch.nn.Module): The PyTorch module to wrap
            interleaver (Optional[Interleaver]): Optional interleaver for managing execution flow
            path (Optional[str]): Optional path string representing the module's location in the model hierarchy
            rename (Optional[Dict[str, Union[str, List[str]]]]): Optional dictionary mapping module names to alias names.
                Example: {"layer1": "first_layer", "layer2": "second_layer"}
                Example: {".model.layers": ".layers"} <-- Mounts .layers to the root model.
                Example: {".transformer": ["model", "mdl"]} <-- Allows access of .transformer as .model or .mdl

        """
        self.path = path

        self._module = module
        self._module.__path__ = path

        self._source = None

        self._interleaver = interleaver

        self._default_mediators: List[List[str]] = []

        self._children: List[Envoy] = []

        self._fake_inputs = inspect._empty
        self._fake_output = inspect._empty

        if rename is not None:
            self._alias = Aliaser(rename)
        else:
            self._alias = None

        for name, module in list(self._module.named_children()):
            setattr(self, name, module)

        if rename is not None:
            self._alias.build(self)

    def __getitem__(self, key: str) -> Envoy:
        """
        Access a child Envoy by index for Module Lists.

        Args:
            key: The index of the child Envoy to retrieve

        Returns:
            The child Envoy at the specified index
        """
        return self._children[key]

    @property
    def interleaving(self) -> bool:
        """
        Check if the Envoy is currently nterleaving.

        Returns:
            True if the Envoy is interleaving, False otherwise
        """
        return self._interleaver is not None

    #### Properties ####

    @property
    def output(self) -> Object:
        """
        Get the output of the module's forward pass.

        This property allows access to the return values produced by the module
        during the forward pass.

        Example:
            >>> model = LanguageModel("gpt2", device_map='auto', dispatch=True)
            >>> with model.trace("Hello World"):
            >>>     attn = model.transformer.h[0].attn.output[0].save()
            >>> print(attn)

        Returns:
            The module's output values
        """

        if self.interleaving:

            return self._interleaver.current.request(
                self._interleaver.current.iterate(f"{self.path}.output")
            )
        elif self._fake_output is not inspect._empty:
            return self._fake_output
        else:
            raise ValueError(
                "Cannot return output of Envoy that is not interleaving nor has a fake output set."
            )

    @output.setter
    def output(self, value: Any):
        """
        Set new values for the module's output.

        This allows for intervention by replacing the module's output with
        custom values during execution.

        Example:
            >>> model = LanguageModel("gpt2", device_map='auto', dispatch=True)
            >>> with model.trace("Hello World"):
            >>>     model.transformer.h[0].attn.output[0] *= 2

        Args:
            value: The new output value to use.
        """
        if self.interleaving:
            self._interleaver.current.swap(
                self._interleaver.current.iterate(f"{self.path}.output"), value
            )

        else:
            raise ValueError("Cannot set output of Envoy that is not interleaving.")

    @property
    def inputs(self) -> Tuple[Tuple[Object], Dict[str, Object]]:
        """
        Get the inputs to the module's forward pass.

        This property provides access to all input values passed to the module
        during the forward pass.

        Example:
            >>> model = LanguageModel("gpt2", device_map='auto', dispatch=True)
            >>> with model.trace("Hello World"):
            >>>     args, kwargs = model.transformer.h[0].attn.inputs

        Returns:
            The module's input values as a tuple of positional and keyword arguments. i.e (args, kwargs)

        """
        if self.interleaving:
            return self._interleaver.current.request(
                self._interleaver.current.iterate(f"{self.path}.input")
            )
        elif self._fake_inputs is not inspect._empty:
            return self._fake_inputs
        else:
            raise ValueError(
                "Cannot return inputs of Envoy that is not interleaving nor has a fake inputs set."
            )

    @inputs.setter
    def inputs(self, value: Any):
        """
        Set new values for the module's inputs.

        This allows for intervention by replacing the module's inputs with
        custom values during execution.

        Example:
            >>> model = LanguageModel("gpt2", device_map='auto', dispatch=True)
            >>> with model.trace("Hello World"):
            >>>     model.transformer.h[0].attn.inputs = (torch.randn(1, 1024, 1024), {})

        Args:
            value: The new input value(s) to use, structured as a tuple of (args, kwargs)
        """
        if self.interleaving:
            self._interleaver.current.swap(
                self._interleaver.current.iterate(f"{self.path}.input"), value
            )
        else:
            raise ValueError("Cannot set inputs of Envoy that is not interleaving.")

    @property
    def input(self) -> Object:
        """
        Get the first input to the module's forward pass.

        This is a convenience property that returns just the first input value
        from all inputs passed to the module. So first positional argument, or first keyword argumetn if there are no positional arguments.

        Example:
            >>> model = LanguageModel("gpt2", device_map='auto', dispatch=True)
            >>> with model.trace("Hello World"):
            >>>     hidden_states = model.transformer.h[0].attn.input.save()
            >>> print(hidden_states)

        Returns:
            The first input value
        """

        inputs = self.inputs

        return [*inputs[0], *inputs[1].values()][0]

    @input.setter
    def input(self, value: Any):
        """
        Set a new value for the module's first input.

        This is a convenience method that replaces just the first input value
        while preserving all other inputs.

        Example:
            >>> model = LanguageModel("gpt2", device_map='auto', dispatch=True)
            >>> with model.trace("Hello World"):
            >>>     model.transformer.h[0].attn.input = torch.randn(1, 1024, 1024)

        Args:
            value: The new value for the first input
        """
        inputs = self.inputs

        value = (value, *inputs[0][1:]), inputs[1]

        self.inputs = value

    @property
    def source(self) -> EnvoySource:
        """
        Get the source code representation of the module.

        This property provides access to the module's source code with operations
        highlighted, allowing for inspection and intervention at specific points.

        Example:

            >>> model = LanguageModel("gpt2", device_map='auto', dispatch=True)

            >>> # We can print to see the formward method of the module and names associated with the operations within.
            >>> print(model.transformer.h[0].attn.source)

                                                   60
                                                   61     if using_eager and self.reorder_and_upcast_attn:
              self__upcast_and_reordered_attn_0 -> 62         attn_output, attn_weights = self._upcast_and_reordered_attn(
                                                   63             query_states, key_states, value_states, attention_mask, head_mask
                                                   64         )
                                                   65     else:
              attention_interface_0             -> 66         attn_output, attn_weights = attention_interface(
                                                   67             self,
                                                   68             query_states,
                                                   69             key_states,
                                                   70             value_states,
                                                   71             attention_mask,
                                                   72             head_mask=head_mask,
                                                   73             dropout=self.attn_dropout.p if self.training else 0.0,
                                                   74             is_causal=is_causal,
                                                   75             **kwargs,
                                                   76         )
                                                   77
              attn_output_reshape_0             -> 78     attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
              contiguous_0                      ->  +     ...
              self_c_proj_0                     -> 79     attn_output = self.c_proj(attn_output)
              self_resid_dropout_0              -> 80     attn_output = self.resid_dropout(attn_output)
                                                   81
                                                   82     return attn_output, attn_weights
                                                   83

            >>> # We can print out one of these to see the only the operation and a few operations before and after.
            >>> print(model.transformer.h[0].attn.source.attention_interface_0)

            .transformer.h.0.attn.attention_interface_0:

                 ....

                     if using_eager and self.reorder_and_upcast_attn:
                         attn_output, attn_weights = self._upcast_and_reordered_attn(
                             query_states, key_states, value_states, attention_mask, head_mask
                         )
                     else:
                 -->     attn_output, attn_weights = attention_interface( <--
                             self,
                             query_states,
                             key_states,
                             value_states,
                             attention_mask,
                             head_mask=head_mask,
                 ....

            >>> with model.trace("Hello World"):
            >>>     # Now we can access it like we would any other Envoy with .input or .output to grab the intermediate value.
            >>>     attn = model.transformer.h[0].attn.source.attention_interface_0.output.save()

            >>> print(attn)


        Returns:
            An EnvoySource object containing the module's source code and operations
        """
        if self._source is None:

            def wrap(fn: Callable, **kwargs):

                bound_obj = (
                    fn.__self__
                    if inspect.ismethod(fn) and fn.__name__ != "forward"
                    else None
                )

                if self.interleaving:
                    return self._interleaver.wrap_operation(
                        fn, **kwargs, bound_obj=bound_obj
                    )
                else:
                    return fn

            source, line_numbers, forward = inject(
                self._module.forward, wrap, self._module.__path__
            )
            self._module.forward = MethodType(forward, self._module)

            self._source = EnvoySource(self._module.__path__, source, line_numbers)
            self._source._set_interleaver(self._interleaver)

        return self._source

    def __call__(self, *args, hook: bool = False, **kwargs):
        return (
            self._module.forward(*args, **kwargs)
            if self.interleaving and not hook
            else self._module(*args, **kwargs)
        )

    #### Public methods ####

    def trace(self, *args, fn: Optional[Callable] = None, trace: bool = None, **kwargs):
        """
        Create a tracer for this module.

        This method returns a tracer that can be used to capture and modify
        the execution of the module.

        Example:
            >>> model = LanguageModel("gpt2", device_map='auto', dispatch=True)
            >>> with model.trace("Hello World"):
            >>>     model.transformer.h[0].attn.output[0][:] = 0

            >>>     output = model.output.save()
            >>> print(output)

        Args:
            *args: Arguments to pass to the tracer
            **kwargs: Keyword arguments to pass to the tracer

        Returns:
            An InterleavingTracer for this module
        """

        # TODO trace= is Legacy
        if trace is not None:
            deprecation_message = f"The `trace` argument {base_deprecation_message}\nJust call the method without a with context instead."
            warnings.warn(deprecation_message)

        if fn is None:
            fn = self.__call__
            kwargs["hook"] = True

        return InterleavingTracer(fn, self, *args, **kwargs)

    def scan(self, *args, **kwargs):
        """
        Just like .trace() but runs the model in fake tensor mode to validate operations and inspect tensor shapes.

        This method returns a tracer that runs the model in fake tensor mode to validate operations
        and inspect tensor shapes without performing actual computation. This is useful for:
        - Validating that operations will work with given input shapes
        - Inspecting the shapes and types of tensors that would flow through the model
        - Debugging shape mismatches or other tensor-related issues.

        Note this will not dispatch the model if not dispatched.

        Example:
            >>> model = LanguageModel("gpt2", device_map='auto', dispatch=True)
            >>> # Value error as the fake inputs and outputs have not been scanned in.
            >>> print(model.transformer.h[0].mlp.output.shape)
            >>> # Scan the model to validate operations and inspect shapes
            >>> with model.scan("Hello World"):
            >>>     # Access fake inputs/outputs to inspect shapes
            >>>     attn_input = model.transformer.h[0].attn.input.save()
            >>>     attn_output = model.transformer.h[0].attn.output[0].save()
            >>> print(f"Attention input shape: {attn_input.shape}")
            >>> print(f"Attention output shape: {attn_output.shape}")
            >>> print(model.transformer.h[0].mlp.output.shape)

        Args:
            *args: Arguments to pass to the tracer
            **kwargs: Keyword arguments to pass to the tracer

        Returns:
            A ScanningTracer for this module
        """
        return ScanningTracer(self.__call__, self, *args, hook=True, **kwargs)

    def edit(self, *, inplace: bool = False):
        """
        Create an editing tracer for this module. Allows for setting default interventions.
        This means this tracer won't execute the module, but will instead set default interventions that are applied on all future executions.

        Edits can be cleared with `Envoy.clear_edits()`.

        Example:
            >>> model = LanguageModel("gpt2", device_map='auto', dispatch=True)
            >>> # Now the first layer attention output will always be 0.
            >>> with model.edit() as edited_model:
            >>>     edited_model.transformer.h[0].attn.output[:] = 0


            >>> with model.trace("Hello World"):
            >>>     output = model.output.save()
            >>> # The orignal model will have the default output.
            >>> print(output)

            >>> with edited_model.trace("Hello World"):
            >>>     edited_output = edited_model.output.save()
            >>> # The edited model will have the output after our intervention.
            >>> print(edited_output)


        Args:
            inplace (bool, optional): Whether to edit in place. Defaults to False.

        Returns:
            (EditingTracer): An EditingTracer for this module
        """

        return EditingTracer(self.__call__, self, inplace=inplace)

    def clear_edits(self):
        """
        Clear all edits for this Envoy.
        """
        self._default_mediators = []

    def export_edits(
        self, name: str, export_dir: Optional[str] = None, variant: str = "__default__"
    ):
        """TODO

        Args:
            name (str): _description_
            export_dir (Optional[str], optional): _description_. Defaults to None.
            variant (str, optional): _description_. Defaults to '__default__'.

        Raises:
            ValueError: _description_
        """

        if len(self._default_mediators) == 0:
            raise ValueError("Cannot export an Envoy before calling .edit().")

        if export_dir is None:

            export_dir = os.path.join(CONFIG.APP.CACHE_DIR, "exports")

        export_dir = os.path.expanduser(os.path.join(export_dir, name))

        os.makedirs(export_dir, exist_ok=True)

        serialization.save(
            self._default_mediators, os.path.join(export_dir, f"{variant}.dill")
        )

    def import_edits(
        self, name: str, export_dir: Optional[str] = None, variant: str = "__default__"
    ):
        """TODO

        Args:
            name (str): _description_
            export_dir (Optional[str], optional): _description_. Defaults to None.
            variant (str, optional): _description_. Defaults to '__default__'.
        """

        if export_dir is None:

            export_dir = os.path.join(CONFIG.APP.CACHE_DIR, "exports")

        export_dir = os.path.expanduser(os.path.join(export_dir, name))

        imported_mediators = serialization.load(
            os.path.join(export_dir, f"{variant}.dill"), self
        )

        self._default_mediators.extend(imported_mediators)

    # TODO legacy
    def session(self, *args, **kwargs):
        return Tracer(*args, **kwargs)

    # TODO legacy
    @property
    @deprecated(message="Use `tracer.iter` instead.")
    def iter(self):
        return IteratorProxy(self._interleaver)

    # TODO legacy
    @deprecated(message="Use `tracer.all()` instead.")
    def all(self):
        return self.iter[:]

    def skip(self, replacement: Any):
        """Skips the execution of this module duting execution / interleaving.
        Behavior is the module will not be executed and will return a replacement value instead.

        Example:
            >>> model = LanguageModel("gpt2", device_map='auto', dispatch=True)
            >>> with model.trace("Hello World"):
            >>>     # Skip the first layer and replace it with the input to the layer.
            >>>     model.transformer.h[0].skip((model.transformer.h[0].input, None))
            >>>     output = model.output.save()
            >>> print(output)

        Args:
            replacement (Any): The replacement value to replace the module's output with.
        """

        requester = self._interleaver.current.iterate(f"{self.path}.input")

        self._interleaver.current.skip(requester, replacement)

    def wait_for_input(self):
        """
        Wait for the input to the module to be available.
        """
        self.inputs

    def wait_for_output(self):
        """
        Wait for the output to the module to be available.
        """
        self.output

    def to(self, device: torch.device):
        """
        Move the module to a specific device.

        This method moves the underlying PyTorch module to the specified device.

        Args:
            device: The device to move the module to

        Returns:
            Self, for method chaining
        """
        self._module.to(device)

        return self

    def cpu(self, *args, **kwargs):
        """
        Move the module to the CPU.
        """
        self._module.cpu(*args, **kwargs)
        return self

    def cuda(self, *args, **kwargs):
        """
        Move the module to the GPU.
        """
        self._module.cuda(*args, **kwargs)
        return self

    @property
    def device(self) -> Optional[torch.device]:
        """
        Get the device the module is on. Finds the first parameter and return its device.
        """
        try:
            return next(self._module.parameters()).device
        except:
            return None

    def modules(
        self,
        include_fn: Callable[[Envoy], bool] = None,
        names: bool = False,
    ) -> List[Envoy]:
        """
        Get all modules in the Envoy tree.

        This method returns all Envoys in the tree, optionally filtered by
        an inclusion function.

        Args:
            include_fn: Optional function to filter modules
            names: Whether to include module names in the result

        Returns:
            A list of Envoys or (name, Envoy) tuples
        """
        result = []

        for envoy in self._children:

            result.extend(envoy.modules(include_fn=include_fn, names=names))

        if include_fn is None or include_fn(self):

            if names:
                result.append((self.path, self))
            else:
                result.append(self)

        return result

    def named_modules(self, *args, **kwargs) -> List[Tuple[str, Envoy]]:
        """
        Returns all Envoys in the Envoy tree along with their name/module_path.

        This is a convenience method that calls modules() with names=True.

        Args:
            include_fn (Callable, optional): Optional function to be ran against all Envoys to check if they should be included in the final collection of Envoys. Defaults to None.
            *args, **kwargs: Additional arguments to pass to modules()

        Returns:
            List[Tuple[str, Envoy]]: Included Envoys and their names/module_paths.
        """

        return self.modules(*args, **kwargs, names=True)

    def get(self, path: str) -> Object:
        """Gets the Envoy/Proxy via its path.

        e.x:
            model = nnsight.LanguageModel("openai-community/gpt2")

            module = model.get('transformer.h.0.mlp')

            with model.trace("Hello"):
                value = model.get('transformer.h.0.mlp.output').save()

        Args:
            path (str): '.' separated path.

        Returns:
            Union[Envoy, InterventionProxyType]: Fetched Envoy/Proxy
        """
        return util.fetch_attr(self, path)

    def interleave(self, interleaver: Interleaver, fn: Callable, *args, **kwargs):

        self._set_interleaver(interleaver)

        try:
            device = self.device

            (args, kwargs) = apply(
                (args, kwargs), lambda tensor: tensor.to(device), torch.Tensor
            )

            with interleaver:

                interleaver(fn, *args, **kwargs)

        finally:
            self._set_interleaver(None)

    #### Private methods ####

    def _add_envoy(self, module: torch.nn.Module, name: str) -> None:
        """
        Adds a new Envoy for a given torch module under this Envoy.

        This method creates a new Envoy for a child module and adds it to
        this Envoy's children.

        Args:
            module: Module to create Envoy for.
            name: Name of envoy/attribute.
        """
        module_path = f"{self.path}.{name}"

        envoy = Envoy(
            module,
            path=module_path,
            rename=self._alias.rename if self._alias is not None else None,
        )

        self._children.append(envoy)

        setattr(self._module, name, module)

        # If the module already has a sub-module named 'input' or 'output',
        # mount the proxy access to 'nns_input' or 'nns_output instead.
        if hasattr(Envoy, name):
            self._handle_overloaded_mount(envoy, name)
        else:
            super().__setattr__(name, envoy)

    def _update(self, module: torch.nn.Module) -> None:
        """Updates the ._model attribute using a new model of the same architecture.
        Used when loading the real weights (dispatching) and need to replace the underlying modules.
        """

        i = 0

        for i, child in enumerate(module.children()):

            self._children[i]._update(child)

        # Handle extra modules added after initialization: issues/376
        for name, child in list(self._module.named_children())[i + 1 :]:

            setattr(module, name, child)

        self._module = module
        self._module.__path__ = self.path

        if self._source is not None:
            def wrap(fn: Callable, **kwargs):

                bound_obj = fn.__self__ if inspect.ismethod(fn) and fn.__name__ != "forward" else None

                if self.interleaving:
                    return self._interleaver.wrap_operation(fn, **kwargs, bound_obj=bound_obj)
                else:
                    return fn

            source, line_numbers, forward = inject(
                self._module.forward, wrap, self._module.__path__
            )
            self._module.forward = MethodType(forward, self._module)

    def _update_alias(self, alias: Dict[str, str]):
        """
        Update the alias for this Envoy and its children.
        """
        self._alias = alias

        for envoy in self._children:
            envoy._update_alias(alias)

    def _set_interleaver(self, interleaver: Interleaver):
        """
        Set the interleaver for this Envoy and all its children.

        This method recursively sets the interleaver for this Envoy and all
        its children.

        Args:
            interleaver: The interleaver to set
        """
        self._interleaver = interleaver

        for envoy in self._children:
            envoy._set_interleaver(interleaver)

        if self._source is not None:
            self._source._set_interleaver(interleaver)

    def _clear(self):
        """
        Clear all cached values and references.

        This method removes all cached values and references to the interleaver,
        preparing the Envoy for garbage collection.
        """
        self._interleaver = None

        if self._source is not None:
            self._source._clear()

    def _shallow_copy(self) -> Envoy:
        """Creates a new instance copy of the same class with the all the attributes of the original instance.

        Returns:
            Self: NNsightModel
        """
        copy = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            copy.__dict__[key] = value

        return copy

    #### Dunder methods ####

    def __len__(self):
        """
        Get the length of the Envoy.
        """
        return len(self._module)

    def __iter__(self):
        """
        Iterate over the Envoy.
        """
        return iter(self._children)

    def __str__(self):
        """
        String representation of the Envoy.

        Returns:
            A string representation of the Envoy showing its path
        """
        return self.__repr__()

    def __reprlist__(self):

        list_of_reprs = [repr(item) for item in self]
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

            local_repr = _addindent(local_repr, 2)
            lines.append(local_repr)

        main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    def __repr__(self):
        """
        Representation of the Envoy.

        Returns:
            The string representation of the Envoy
        """

        if isinstance(self._module, torch.nn.ModuleList):
            return self.__reprlist__()

        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self._module.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for envoy in self._children:
            key = envoy.path.split(".")[-1]
            mod_str = repr(envoy)
            mod_str = _addindent(mod_str, 2)
            if self._alias is not None and key in self._alias.name_to_aliases:
                key = "/".join([*self._alias.name_to_aliases[key], key])
            child_lines.append("(" + key + "): " + mod_str)

        if self._alias is not None:
            for extra in self._alias.extras:

                key = "/".join(self._alias.name_to_aliases[extra])
                envoy = self.get(extra)
                mod_str = repr(envoy)
                mod_str = _addindent(mod_str, 2)
                child_lines.append("(" + key + "): " + mod_str)

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

    def __getattr__(self, name: str) -> Union[torch.nn.Module, Envoy, Any]:
        """
        Get an attribute from the underlying module.

        If the attribute is callable, it will be wrapped in a tracer to enable
        intervention during execution.

        Args:
            name: The name of the attribute to get

        Returns:
            The attribute value, possibly wrapped in a tracer

        Raises:
            AttributeError: If the attribute doesn't exist
        """

        if self._alias is not None and name in self._alias.alias_to_name:
            return util.fetch_attr(self, self._alias.alias_to_name[name])

        if hasattr(self._module, name):
            value = getattr(self._module, name)

            # It's a method bound to the module, create an interleaver for it
            if isinstance(
                value,
                (FunctionType, MethodType, BuiltinFunctionType, BuiltinMethodType),
            ):

                # If the Envoy defines a method with __nnsight_{name}__, use it instead to override
                value = getattr(self, f"__nnsight_{name}__", value)

                def trace(*args, **kwargs):
                    try:
                        return self.trace(*args, fn=value, **kwargs)

                    except WithBlockNotFoundError as e:

                        return value(*args, **kwargs)

                return trace
            else:
                return value
        else:
            raise AttributeError(f"{self} has no attribute {name}")

    def __setattr__(self, key: Any, value: Any) -> None:
        """
        Set an attribute on the Envoy.

        If the value is a PyTorch module, it will be wrapped in an Envoy to enable
        intervention during execution.

        Args:
            key: The attribute name
            value: The attribute value
        """
        if key != "_module" and isinstance(value, torch.nn.Module):
            self._add_envoy(value, key)
        else:
            super().__setattr__(key, value)


# TODO extend Envoy
class OperationEnvoy:
    """
    Represents a specific operation within a module's forward pass.

    This class provides access to the inputs and outputs of individual
    operations within a module's execution, allowing for fine-grained
    inspection and intervention at the operation level.
    """

    def __init__(
        self,
        name: str,
        source: str,
        line_number: int,
        interleaver: Optional[Interleaver] = None,
    ):
        """
        Initialize an OperationEnvoy.

        Args:
            name: The fully qualified name of the operation
            source: The source code of the module containing the operation
            line_number: The line number of the operation in the source
            interleaver: Optional interleaver for managing execution flow
        """
        self.name = name
        self.source_code = source
        self.line_number = line_number

        self._interleaver = interleaver

        self._source: EnvoySource = None
        self._fn: Callable = None

    def __str__(self):
        """
        String representation showing the operation in context.

        This method returns a formatted string showing the operation's source code
        with surrounding context lines and highlighting the operation line.

        Returns:
            A formatted string showing the operation's source code with context
        """
        source_lines = self.source_code.split("\n")
        start_idx = max(0, self.line_number - 5)
        end_idx = min(len(source_lines) - 1, self.line_number + 8)

        highlighted_lines = [self.name + ":\n"]

        if start_idx != 0:
            highlighted_lines.append("    ....")

        for i in range(start_idx, end_idx):
            line = source_lines[i]
            if i == self.line_number + 1:
                highlighted_lines.append(f"    --> {line[4:]} <--")
            else:
                highlighted_lines.append("    " + line)

        if end_idx != len(source_lines) - 1:
            highlighted_lines.append("    ....")

        return "\n".join(highlighted_lines)

    @property
    def output(self) -> Union[Any, torch.Tensor]:
        """
        Get the output of this operation.

        This property provides access to the return value(s) produced by the operation
        during execution.

        Returns:
            The operation's output value(s)
        """

        return self._interleaver.current.request(f"{self.name}.output")

    @output.setter
    def output(self, value: Any) -> None:
        """
        Set a new value for the operation's output.

        This allows for intervention by replacing the operation's output with
        a custom value during execution.

        Args:
            value: The new output value
        """
        self._interleaver.current.swap(f"{self.name}.output", value)

    @property
    def inputs(
        self,
    ) -> Tuple[Tuple[Any, torch.Tensor], Dict[str, Union[torch.Tensor, Any]]]:
        """
        Get the inputs to this operation.

        This property provides access to all input value(s) passed to the operation
        during execution, structured as a tuple of positional and keyword arguments.

        Returns:
            The operation's input value(s)
        """
        return self._interleaver.current.request(f"{self.name}.input")

    @inputs.setter
    def inputs(self, value: Any) -> None:
        """
        Set new values for the operation's inputs.

        This allows for intervention by replacing the operation's inputs with
        custom values during execution.

        Args:
            value: The new input value(s)
        """
        self._interleaver.current.swap(f"{self.name}.input", value)

    @inputs.deleter
    def inputs(self):
        """
        Clear the cached input value.

        This removes any stored input values, forcing them to be recomputed
        on the next access.
        """
        self._input = None

    @property
    def input(self) -> Union[Any, torch.Tensor]:
        """
        Get the first input to the operation.

        This is a convenience property that returns just the first input value
        from all inputs passed to the operation.

        Returns:
            The first input value
        """

        inputs = self.inputs

        return [*inputs[0], *inputs[1].values()][0]

    @input.setter
    def input(self, value: Any) -> None:
        """
        Set a new value for the operation's first input.

        This is a convenience method that replaces just the first positional input
        while preserving all other inputs.

        Args:
            value: The new value for the first input
        """
        inputs = self.inputs

        value = (value, *inputs[0][1:]), inputs[1]

        self.inputs = value

    @property
    def source(self) -> EnvoySource:
        """
        Get the source code of the operation.

        This property provides access to the operation's source code with nested
        operations highlighted, allowing for inspection and intervention at specific points.

        Returns:
            An EnvoySource object containing the operation's source code and nested operations
        """

        if self._source is None:
            fn = self._interleaver.current.request(f"{self.name}.fn")

            def wrap(fn: Callable, **kwargs):

                bound_obj = (
                    fn.__self__
                    if fn.__name__ != "forward" and inspect.ismethod(fn)
                    else None
                )

                return self._interleaver.wrap_operation(
                    fn, **kwargs, bound_obj=bound_obj
                )

            source, line_numbers, fn = inject(fn, wrap, self.name)

            self._source = EnvoySource(
                self.name, source, line_numbers, interleaver=self._interleaver
            )

            self._fn = fn

        self._interleaver.current.swap(f"{self.name}.fn", self._fn)

        return self._source

    # @input.setter
    # def input(self, value: Any):
    #     #TODO would need await...
    #     inputs = self._input
    #     self._input = ((value, *inputs[0]), inputs[1])
    #     self.interleaver.set_swap(self._input, (self.module, self.name), Events.INPUT)

    # @input.deleter
    # def input(self):
    #     self._input = None

    def _set_interleaver(self, interleaver: Interleaver):
        """
        Set the interleaver for this operation.

        Args:
            interleaver: The interleaver to use for managing execution flow
        """
        self._interleaver = interleaver

        if self._source is not None:
            self._source._set_interleaver(interleaver)

    def _clear(self):
        """
        Clear all cached values and references.

        This method removes all cached values and references to the interleaver,
        preparing the OperationEnvoy for garbage collection.
        """
        self._interleaver = None


class EnvoySource:
    """
    Represents the source code of a module with operations highlighted.

    This class provides access to the individual operations within a module's
    source code, allowing for inspection and intervention at specific points
    in the code. It serves as a bridge between the source code representation
    and the runtime execution of operations.
    """

    def __init__(
        self,
        name: str,
        source: str,
        line_numbers: dict,
        interleaver: Optional[Interleaver] = None,
    ):
        """
        Initialize an EnvoySource.

        Args:
            name: The fully qualified name of the module or operation
            source: The source code string
            line_numbers: A dictionary mapping operation names to line numbers
            interleaver: Optional interleaver for managing execution flow
        """
        self.source = source
        self.line_numbers = line_numbers

        self.operations: List[OperationEnvoy] = []

        for _name, line_number in line_numbers.items():
            operation = OperationEnvoy(
                f"{name}.{_name}", source, line_number, interleaver=interleaver
            )
            setattr(self, _name, operation)
            self.operations.append(operation)

    def __str__(self):
        """
        String representation showing the source code with operations highlighted.

        This method returns a formatted string showing the source code with
        operation names and line numbers, making it easy to identify intervention points.

        Returns:
            A formatted string showing the source code with operation names and line numbers
        """
        # Find the longest name for proper alignment
        max_name_length = (
            max(len(name) for name in self.line_numbers.keys())
            if self.line_numbers
            else 0
        )

        source_lines = self.source.split("\n")
        formatted_lines = [
            " " * (max_name_length + 6) + "* " + source_lines[0]
        ]  # Keep the function definition unchanged

        # Group operations by line number
        operations_by_line = {}
        for name, line_number in self.line_numbers.items():
            if line_number not in operations_by_line:
                operations_by_line[line_number] = []
            operations_by_line[line_number].append(name)

        for i, line in enumerate(source_lines[1:]):
            line_number = i

            # Check if this line has operations
            if line_number in operations_by_line:
                # Handle multiple operations on the same line
                operations = operations_by_line[line_number]

                # First operation gets the line number
                first_op = operations[0]
                line_prefix = f" {first_op:{max_name_length}} ->{line_number:3d} "
                formatted_lines.append(f"{line_prefix}{line}")

                # For nested operations, unwrap them onto separate lines
                if len(operations) > 1:
                    for op in operations[1:]:
                        continuation_prefix = f" {op:{max_name_length}} ->  + "
                        # Instead of just showing a vertical line, show the operation on its own line
                        formatted_lines.append(
                            f"{continuation_prefix}{' ' * (len(line) - len(line.lstrip()))}..."
                        )
            else:
                # Regular line with no operations
                line_prefix = " " * (max_name_length + 4) + f"{line_number:3d} "
                formatted_lines.append(f"{line_prefix}{line}")

        source = "\n".join(formatted_lines)

        return source

    def _set_interleaver(self, interleaver: Interleaver):
        """
        Set the interleaver for all operations.

        This method recursively sets the interleaver for all operations
        in this source code representation.

        Args:
            interleaver: The interleaver to use for managing execution flow
        """
        for operation in self.operations:
            operation._set_interleaver(interleaver)

    def _clear(self):
        """
        Clear all cached values in all operations.

        This method recursively clears all cached values and references
        in all operations, preparing them for garbage collection.
        """
        for operation in self.operations:
            operation._clear()

            if operation._source is not None:
                operation._source._clear()

    def __getattr__(self, name: str) -> Union[OperationEnvoy]:

        return super().__getattr__(name)


class Aliaser:

    def __init__(self, rename: Dict[str, Union[str, List[str]]]):
        """
        Initialize an Aliaser.

        Args:
            rename (Dict[str, Union[str, List[str]]]): Dictionary mapping module names to alias names.
                Example: {"layer1": "first_layer", "layer2": "second_layer"}
                Example: {".model.layers": ".layers"} <-- Mounts .layers to the root model.
                Example: {".transformer": ["model", "mdl"]} <-- Allows access of .transformer as .model or .mdl

        Attributes:
            rename (Dict[str, Union[str, List[str]]]): Dictionary mapping module names to alias names.
            alias_to_name (Dict[str, str]): Dictionary mapping alias names to module names.
            name_to_aliases (Dict[str, List[str]]): Dictionary mapping module names to list of alias names.
            extras (Dict[str, List[str]]): Dictionary mapping attribute paths (.transformer.h) to list of alias names.
                Used to show dot seperated attributes in the string representation of the Envoy.


        """

        self.rename = rename

        self.alias_to_name = {}
        self.name_to_aliases = {}
        self.extras = {}

    def build(self, envoy: Envoy):

        for name, aliases in self.rename.items():

            try:
                util.fetch_attr(envoy, name)
            except:
                continue

            if isinstance(aliases, str):
                aliases = [aliases]

            name = name.removeprefix(".")

            if "." in name:

                self.extras[name] = aliases

            self.name_to_aliases[name] = aliases

            for alias in aliases:
                self.alias_to_name[alias] = name
