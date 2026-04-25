from __future__ import annotations

import inspect
import os
import warnings
from functools import wraps
from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, MethodType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import torch
from torch.nn.modules.module import _addindent

from .. import CONFIG, util
from ..util import apply

from .batching import Batchable
from .source import (
    SourceEnvoy,
    resolve_true_forward,
    get_or_create_source_accessor,
)
from .tracing.base import Tracer, WithBlockNotFoundError
from .tracing.editing import EditingTracer
from .tracing.globals import Object
from .tracing.iterator import IteratorProxy
from .tracing.tracer import InterleavingTracer, ScanningTracer
from .interleaver import Interleaver, Mediator, IEnvoy, eproperty
from .hooks import requires_input, requires_output


def trace_only(fn: Callable):

    @wraps(fn)
    def wrapper(self: Envoy, *args, **kwargs):

        if self.interleaver is None:
            raise ValueError(f"Must be within a trace to use `.{fn.__name__}(...)`")

        return fn(self, *args, **kwargs)

    return wrapper


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
        interleaver (Optional[Interleaver]): Interleaver for managing execution flow
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
        envoys: Optional[
            Union[Type["Envoy"], Dict[Type[torch.nn.Module], Type["Envoy"]]]
        ] = None,
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
            envoys (Optional[Union[Type[Envoy], Dict]]):
                Controls which Envoy class wraps descendant modules. Propagates down the envoy tree.
                - None (default): all descendants are wrapped with the base Envoy class.
                - A class: all descendants are wrapped with that class.
                - A dict whose values are ``Envoy`` subclasses. Keys may be:
                    * A ``torch.nn.Module`` subclass — matches when the class appears in the
                      descendant's MRO. Example: ``{torch.nn.Linear: MyLinearEnvoy}``.
                    * A string — matches when the descendant's envoy path ends with the key
                      treated as a dotted suffix (component-wise). With a rename dict in play,
                      each component also matches via single-component aliases — so
                      ``{"attn": MyAttnEnvoy}`` matches a path ending in ``self_attn`` when
                      the user passed ``rename={"self_attn": "attn"}``.
                  Type keys are tried first; string keys are a fallback. Descendants without
                  a match fall back to the base Envoy class.
                Example: {torch.nn.Linear: MyLinearEnvoy, "self_attn": MyAttnEnvoy}

        """
        self.path = path

        self._module = module
        self._module.__path__ = path

        self._source = None

        self.interleaver = interleaver if interleaver is not None else Interleaver()
        self.interleaver.wrap_module(module)

        self._default_mediators: List[Mediator] = []

        self._envoys = envoys

        if rename is not None:
            self._alias = Aliaser(rename)
        else:
            self._alias = None

        for name, module in list(self._module.named_children()):
            setattr(self, name, module)

        if rename is not None:
            self._alias.build(self)

    @property
    def _children(self) -> List[Envoy]:
        """
        Get the children of the Envoy.
        """
        return [envoy for envoy in self.__dict__.values() if isinstance(envoy, Envoy)]

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
        return self.interleaver is not None and self.interleaver.interleaving

    #### Properties ####

    @eproperty()
    @requires_output
    def output(self) -> Object:
        """Get the output of the module's forward pass.

        Examples:
            >>> with model.trace("Hello World"):
            ...     attn = model.transformer.h[0].attn.output[0].save()
        """

    @eproperty(key="input")
    @requires_input
    def inputs(self) -> Tuple[Tuple[Object], Dict[str, Object]]:
        """Get the inputs to the module's forward pass.

        Returns:
            (args, kwargs) tuple of positional and keyword arguments.

        Examples:
            >>> with model.trace("Hello World"):
            ...     args, kwargs = model.transformer.h[0].attn.inputs
        """

    @eproperty(key="input")
    @requires_input
    def input(self) -> Object:
        """Get the first input to the module's forward pass.

        Convenience wrapper around :attr:`inputs` that extracts the first
        positional argument, or the first keyword argument if there are no
        positional arguments.

        Examples:
            >>> with model.trace("Hello World"):
            ...     hidden_states = model.transformer.h[0].attn.input.save()
        """

    @input.preprocess
    def input(self, value):
        return [*value[0], *value[1].values()][0]

    @input.postprocess
    def input(self, value):
        inputs = self.inputs
        return (value, *inputs[0][1:]), inputs[1]

    @property
    def source(self) -> SourceEnvoy:
        """Get the source code representation of the module.

        Lazily resolves to a :class:`SourceEnvoy` over the module's global
        :class:`SourceAccessor`. The accessor — and its per-call-site
        :class:`OperationAccessor` instances — are created once per module
        and shared across all Envoys / Interleavers / Mediators that touch
        it. This Envoy keeps a per-instance :class:`SourceEnvoy` so each
        Envoy has its own user-facing wrapper.

        Examples:
            >>> model = LanguageModel("gpt2", device_map='auto', dispatch=True)
            >>> print(model.transformer.h[0].attn.source)
            >>> with model.trace("Hello World"):
            ...     attn = model.transformer.h[0].attn.source.attention_interface_0.output.save()

        Returns:
            A :class:`SourceEnvoy` exposing operation-level access.
        """
        if self._source is None:
            accessor = get_or_create_source_accessor(self._module)
            self._source = SourceEnvoy(accessor, interleaver=self.interleaver)
        return self._source

    def __call__(self, *args, hook: bool = False, **kwargs):
        return (
            self._module.forward(*args, **kwargs)
            if self.interleaver.current is not None and not hook
            else self._module(*args, **kwargs)
        )

    #### Public methods ####

    def trace(
        self,
        *args,
        fn: Optional[Callable] = None,
        tracer_cls: Type[InterleavingTracer] = InterleavingTracer,
        **kwargs,
    ):
        """
        Create a tracer for this module.

        This method returns a tracer that can be used to capture and modify
        the execution of the module.

        Examples:
            >>> model = LanguageModel("gpt2", device_map='auto', dispatch=True)
            >>> with model.trace("Hello World"):
            ...     model.transformer.h[0].attn.output[0][:] = 0

            ...     output = model.output.save()
            >>> print(output)

        Args:
            *args: Arguments to pass to the tracer
            **kwargs: Keyword arguments to pass to the tracer

        Returns:
            An InterleavingTracer for this module
        """

        if fn is None:
            fn = self.__call__

        return tracer_cls(fn, self, *args, **kwargs)

    def scan(self, *args, **kwargs):
        """
        Just like .trace() but runs the model in fake tensor mode to validate operations and inspect tensor shapes.

        This method returns a tracer that runs the model in fake tensor mode to validate operations
        and inspect tensor shapes without performing actual computation. This is useful for:
        - Validating that operations will work with given input shapes
        - Inspecting the shapes and types of tensors that would flow through the model
        - Debugging shape mismatches or other tensor-related issues.

        Note this will not dispatch the model if not dispatched.

        Examples:
            >>> model = LanguageModel("gpt2", device_map='auto', dispatch=True)
            >>> # Value error as the fake inputs and outputs have not been scanned in.
            >>> print(model.transformer.h[0].mlp.output.shape)
            >>> # Scan the model to validate operations and inspect shapes
            >>> with model.scan("Hello World"):
            ...     # Access fake inputs/outputs to inspect shapes
            ...     attn_input = model.transformer.h[0].attn.input.save()
            ...     attn_output = model.transformer.h[0].attn.output[0].save()
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

        Examples:
            >>> model = LanguageModel("gpt2", device_map='auto', dispatch=True)
            >>> # Now the first layer attention output will always be 0.
            >>> with model.edit() as edited_model:
            ...     edited_model.transformer.h[0].attn.output[:] = 0


            >>> with model.trace("Hello World"):
            ...     output = model.output.save()
            >>> # The orignal model will have the default output.
            >>> print(output)

            >>> with edited_model.trace("Hello World"):
            ...     edited_output = edited_model.output.save()
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

        from . import serialization

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

        from . import serialization

        imported_mediators = serialization.load(
            os.path.join(export_dir, f"{variant}.dill"), self
        )

        self._default_mediators.extend(imported_mediators)

    # TODO legacy
    def session(self, *args, tracer_cls: Type[Tracer] = Tracer, **kwargs):
        tracer = tracer_cls(*args, **kwargs)
        setattr(tracer, "model", self)
        return tracer

    @property
    @trace_only
    def iter(self):
        warnings.warn(
            "model.iter is deprecated and will be removed in a future version. "
            "Use tracer.iter instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return IteratorProxy(self.interleaver)

    @trace_only
    def all(self):
        warnings.warn(
            "model.all() is deprecated and will be removed in a future version. "
            "Use tracer.all() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.iter[:]

    @trace_only
    def next(self, step: int = 1):
        warnings.warn(
            "model.next() is deprecated and will be removed in a future version. "
            "Use tracer.next() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.interleaver.current.iteration += step
        return self

    @trace_only
    @requires_input
    def skip(self, replacement: Any):
        """Skips the execution of this module duting execution / interleaving.
        Behavior is the module will not be executed and will return a replacement value instead.

        Examples:
            >>> model = LanguageModel("gpt2", device_map='auto', dispatch=True)
            >>> with model.trace("Hello World"):
            ...     # Skip the first layer and replace it with the input to the layer.
            ...     model.transformer.h[0].skip((model.transformer.h[0].input, None))
            ...     output = model.output.save()
            >>> print(output)

        Args:
            replacement (Any): The replacement value to replace the module's output with.
        """

        return self.interleaver.current.skip(
            self.interleaver.iterate_requester(f"{self.path}.input"), replacement
        )

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

    @property
    def devices(self) -> Optional[set[torch.device]]:
        """
        Get the devices the module is on. Finds all parameters and return their devices.
        """
        try:
            return {p.device for p in self._module.parameters()}
        except:
            return set()

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

    def interleave(self, fn: Union[Callable, str], *args, **kwargs):

        device = self.device

        (args, kwargs) = apply(
            (args, kwargs), lambda tensor: tensor.to(device), torch.Tensor
        )

        if isinstance(fn, str):
            fn = getattr(self, fn)

        try:
            with self.interleaver:
                result = fn(*args, **kwargs)

                self.interleaver.handle("result", result)

            self.interleaver.check_cache_full()
            self.interleaver.check_dangling_mediators()

        finally:
            self.interleaver.cancel()

    #### Private methods ####

    def _resolve_envoy_class(
        self, module: torch.nn.Module, path: Optional[str] = None
    ) -> Type[Envoy]:
        """Resolve which Envoy class to use for wrapping a child module.

        Consults ``self._envoys``:
        - If None, returns the base Envoy class.
        - If a class, returns that class.
        - If a dict, keys can be:
            - A ``torch.nn.Module`` subclass. Matches when the class appears in
              ``module``'s MRO. Type-keyed matches are tried first.
            - A string. Matches when ``path`` ends with the key treated as a
              dotted suffix (component-wise, not substring). With a rename dict
              in play, each component matches either literally or via an alias
              from a single-component rename entry — so a key like ``"attn"``
              will match a path ending in ``self_attn`` when the user passed
              ``rename={"self_attn": "attn"}``.
          Type keys are checked before string keys. Falls back to the base
          Envoy class if no entry matches.
        """
        mapping = self._envoys

        if mapping is None:
            return Envoy

        if isinstance(mapping, type):
            return mapping

        for cls in type(module).__mro__:
            if cls in mapping:
                return mapping[cls]

        if path is not None:
            for key, envoy_cls in mapping.items():
                if isinstance(key, str) and self._path_matches_key(path, key):
                    return envoy_cls

        return Envoy

    def _path_matches_key(self, path: str, key: str) -> bool:
        """Does ``path`` end with dotted ``key``, with alias-aware components?

        Both path and key are split on ``.``. The key is matched as a suffix of
        the path, component by component. A key component matches a path
        component if they are equal, or if the path component has the key
        component as a rename alias (see :meth:`_component_matches`).
        """
        key = key.removeprefix(".")
        if not key:
            return False
        path_parts = path.split(".")
        key_parts = key.split(".")
        if len(key_parts) > len(path_parts):
            return False
        tail = path_parts[-len(key_parts) :]
        return all(self._component_matches(pc, kc) for pc, kc in zip(tail, key_parts))

    def _component_matches(self, path_component: str, key_component: str) -> bool:
        """Whether ``key_component`` is a valid name for ``path_component``.

        True if they are equal, or if the rename dict contains a
        single-component entry ``{path_component: [..., key_component, ...]}``
        (the user aliased ``path_component`` to ``key_component``).

        Multi-component rename entries (e.g. ``{"transformer.h": "layers"}``)
        are not consulted here; component matching is single-component only.
        """
        if path_component == key_component:
            return True
        if self._alias is None:
            return False
        for rename_key, aliases in self._alias.rename.items():
            stripped = rename_key.removeprefix(".")
            if "." in stripped:
                continue
            if stripped != path_component:
                continue
            if isinstance(aliases, str):
                aliases = [aliases]
            if key_component in aliases:
                return True
        return False

    def _add_envoy(self, module: torch.nn.Module, name: str) -> Envoy:
        """
        Adds a new Envoy for a given torch module under this Envoy.

        This method creates a new Envoy for a child module and adds it to
        this Envoy's children.

        Args:
            module: Module to create Envoy for.
            name: Name of envoy/attribute.
        """
        module_path = f"{self.path}.{name}"

        envoy_cls = self._resolve_envoy_class(module, module_path)

        envoy = envoy_cls(
            module,
            path=module_path,
            rename=self._alias.rename if self._alias is not None else None,
            interleaver=self.interleaver,
            envoys=self._envoys,
        )

        setattr(self._module, name, module)

        # If the module already has a sub-module named 'input' or 'output',
        # mount the proxy access to 'nns_input' or 'nns_output instead.

        if hasattr(Envoy, name):
            self._handle_overloaded_mount(envoy, name)
        else:
            super().__setattr__(name, envoy)

        return envoy

    def _handle_overloaded_mount(self, envoy: Envoy, mount_point: str) -> None:
        """If a given module already has an attribute of the same name as something nnsight wants to add, we need to rename it.

        Directly edits the underlying class to accomplish this.

        Args:
            envoy (Envoy): Envoy to handle.
            mount_point (str): Overloaded attribute name.
        """

        warnings.warn(
            f"Module `{self.path}` of type `{type(self._module)}` has pre-defined a `{mount_point}` attribute. nnsight access for `{mount_point}` will be mounted at `.nns_{mount_point}` instead of `.{mount_point}` for this module only."
        )

        # If we already shifted a mount point dont create another new class.
        if "Preserved" in self.__class__.__name__:

            new_cls = self.__class__

        else:

            new_cls = type(
                f"{self.__class__.__name__}.Preserved",
                (self.__class__,),
                {},
            )

            object.__setattr__(self, "__class__", new_cls)

        # Get the normal proxy mount point
        mount = getattr(Envoy, mount_point)

        setattr(new_cls, f"nns_{mount_point}", mount)

        if isinstance(mount, eproperty):

            # Replace the eproperty with a property that returns the child
            # envoy from the instance dict, but delegates setter to the
            # original eproperty's __set__.
            original_ep = mount

            def _ep_getter(slf, _mp=mount_point):
                return slf.__dict__[_mp]

            def _ep_setter(slf, value, _ep=original_ep):
                _ep.__set__(slf, value)

            setattr(new_cls, mount_point, property(_ep_getter, _ep_setter))

        elif isinstance(mount, property):

            mount = property(
                lambda slf: slf.__dict__[mount_point],
                mount.fset,
                mount.fdel,
                mount.__doc__,
            )

            setattr(new_cls, mount_point, mount)

        # Move it to nns_<mount point>
        self.__dict__[mount_point] = envoy

    def _update(self, module: torch.nn.Module) -> None:
        """Updates the ._model attribute using a new model of the same architecture.
        Used when loading the real weights (dispatching) and need to replace the underlying modules.
        """

        for name, existing_child in self._module.named_children():

            if hasattr(module, name):

                child = getattr(module, name)

                self.__dict__[name]._update(child)

            else:
                setattr(module, name, existing_child)

        # Capture the existing SourceAccessor (if any) before the old module
        # is dropped — its OperationAccessors carry hook state and any
        # nested SourceAccessors for recursive source tracing.
        old_accessor = None
        if hasattr(self._module, "forward"):
            old_accessor = getattr(self._module.forward, "__source_accessor__", None)

        self._module = module
        self._module.__path__ = self.path
        self.interleaver.wrap_module(module)

        # Transfer the SourceAccessor onto the new module's nnsight_forward,
        # rebinding its injected forward against the new module's true fn.
        # OperationAccessors are kept intact so any pre-existing
        # OperationEnvoy / SourceEnvoy references remain valid after dispatch.
        if old_accessor is not None:
            old_accessor.rebind(resolve_true_forward(module))
            module.forward.__source_accessor__ = old_accessor

    def _update_alias(self, alias: Dict[str, str]):
        """
        Update the alias for this Envoy and its children.
        """
        if alias is not None:
            self._alias = Aliaser(alias)
            self._alias.build(self)

            for envoy in self._children:
                envoy._update_alias(alias)

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

        eproperty_lines = []
        for cls in type(self).__mro__:
            for attr_name, attr_val in cls.__dict__.items():
                if isinstance(attr_val, eproperty) and attr_val.description is not None:
                    eproperty_lines.append(
                        "(" + attr_val.name + "): " + attr_val.description
                    )

        lines = extra_lines + child_lines + eproperty_lines

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
            if not self.interleaver.interleaving and isinstance(
                value,
                (FunctionType, MethodType, BuiltinFunctionType, BuiltinMethodType),
            ):

                # If the Envoy defines a method with __nnsight_{name}__, use it instead to override
                value = getattr(self, f"__nnsight_{name}__", value)

                def trace(*args, **kwargs):
                    try:
                        tracer = self.trace(*args, fn=value, **kwargs)
                        tracer.capture()
                        return tracer
                    except WithBlockNotFoundError as e:

                        args, kwargs, _ = self._prepare_input(*args, **kwargs)

                        return value(*args, **kwargs)

                return trace

            elif isinstance(value, torch.nn.Module):
                # If the _module has a module in its __dict__ but wasn't picked up when creating the Envoy,
                # Hopefully it is alrady an Envoy somewhere in the tree.
                # https://github.com/ndif-team/nnsight/issues/479
                # This happened because some transformers models set this class attr: _checkpoint_conversion_mapping
                if hasattr(value, "__path__"):
                    return util.fetch_attr(self, value.__path__[len(self.path) :])
                return self._add_envoy(value, name)
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
            if "_module" in self.__dict__ and hasattr(self._module, key):
                return setattr(self._module, key, value)
            else:
                super().__setattr__(key, value)

    def __getstate__(self):

        state = self.__dict__.copy()

        state["interleaver"]._persistent_id = "Interleaver"
        state["_module"]._persistent_id = f"Module:{self.path}"

        state.pop("_source")

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        self._source = None


class Aliaser:

    def __init__(self, rename: Dict[str, Union[str, List[str]]]):
        """
        Initialize an Aliaser.

        Args:
            rename (Dict[str, Union[str, List[str]]]): Dictionary mapping module names to alias names.
                Examples: {"layer1": "first_layer", "layer2": "second_layer"}
                Examples: {".model.layers": ".layers"} <-- Mounts .layers to the root model.
                Examples: {".transformer": ["model", "mdl"]} <-- Allows access of .transformer as .model or .mdl

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
