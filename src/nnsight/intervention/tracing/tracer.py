import copy
import re
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Union)

import torch
from torch._subclasses.fake_tensor import FakeCopyMode, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from ... import util
from ..backends.base import Backend
from ..batching import Batcher
from ..interleaver import Interleaver
from .base import ExitTracingException, Tracer
from .globals import Object
from .invoker import Invoker
from .iterator import IteratorProxy

if TYPE_CHECKING:
    from ..envoy import Envoy
    from ..interleaver import Mediator
else:
    Envoy = Any


class Cache:
    """
    A cache for storing and transforming tensor values during tracing.

    This class provides functionality to store tensor values with optional
    transformations such as detaching from computation graph, moving to a
    specific device, or converting to a specific dtype.
    """

    @dataclass
    class Entry:
        output: Optional[Any] = None
        inputs: Optional[Tuple[Tuple[Any, ...], Dict[str, Any]]] = None

        @property
        def input(self):
            """
            Gets the first positional argument of the inputs value to the cached module. Returns None if no inputs were cached.
            """
            if self.inputs is None:
                return None
            return [*self.inputs[0], *self.inputs[1].values()][0]

    class CacheDict(Dict):

        """
        A dictionary subclass that provides convenient access to cached module activations.

        This class extends the standard dictionary to provide both dictionary-style access
        and attribute-style access to cached activations. It supports hierarchical access
        to nested modules using dot notation and indexing for module lists.

        Examples:
            Access cached activations using dictionary keys:
            >>> cache['model.transformer.h.0.attn']

            Access using attribute notation:
            >>> cache.model.transformer.h[0].attn

            Access module outputs and inputs:
            >>> cache.model.transformer.h[0].output
            >>> cache.model.transformer.h[0].inputs
            >>> cache.model.transformer.h[0].input  # First input argument

        The class maintains an internal path that tracks the current location in the
        module hierarchy, allowing for intuitive navigation through nested modules.
        """

        def __init__(self, data: "Union[Cache.CacheDict, Dict[str, Cache.Entry]]", path: Optional[str] = "", alias: Optional[Dict[str, str]] = None):
            self._path = path
            self._alias = alias

            super().__init__(data)
        
        @property
        def output(self):
            """
            Returns the output attribute from the Cache.Entry at the current path.
            """
            return dict.__getitem__(self, self._path).output

        @property
        def inputs(self):
            """
            Returns the inputs attribute from the Cache.Entry at the current path.
            """
            return dict.__getitem__(self, self._path).inputs

        @property
        def input(self):
            """
            Returns the input property from the Cache.Entry at the current path.
            """
            return dict.__getitem__(self, self._path).input
        
        def __getitem__(self, key):
            name = self._alias[key] if self._alias is not None and key in self._alias else key
            if isinstance(key, str):
                path = self._path + "." + name if self._path != "" else name
                return dict.__getitem__(self, path)
            
            if isinstance(name, int):
                path = self._path + "." + f"{name}"
                
                if any(key.startswith(path) for key in self):
                    return Cache.CacheDict(self, path, self._alias)
                elif any(key.startswith(self._path + ".") and len(key) >= len(self._path) + 1 and key[len(self._path) + 1].isdigit() for key in self):
                    raise IndexError(f"Index {key} is out of bounds for modulelist or module does not allow indexing.")
                
            return dict.__getitem__(self, key)

        def __getattr__(self, attr: str):
            path = self._path + "." + attr if self._path != "" else attr

            if any(key.startswith(path) for key in self):
                return Cache.CacheDict(self, path, self._alias)
            elif self._alias is not None and attr in self._alias:
                return self.__getattr__(self._alias[attr])
            else:
                raise AttributeError(f"'{attr}' module path was never cached. '{self.__class__.__name__}' has no matching attribute.")

    def __init__(
        self,
        modules: Optional[List[Union[Envoy, str]]] = None,
        device: Optional[torch.device] = torch.device("cpu"),
        dtype: Optional[torch.dtype] = None,
        detach: Optional[bool] = True,
        include_output: bool = True,
        include_inputs: bool = False,
        alias: Optional[Dict[str, str]] = None
    ):
        """
        Initialize a Cache with optional transformation parameters.

        Args:
            device: Optional device to move tensors to
            dtype: Optional dtype to convert tensors to
            detach: Whether to detach tensors from computation graph
            include_output: Whether to include output in the cached activations
            include_inputs: Whether to include inputs in the cached activations
        """
        self.device = device
        self.dtype = dtype
        self.detach = detach
        self.modules = modules
        self.include_output = include_output
        self.include_inputs = include_inputs

        if self.modules is not None:
            self.modules = {m if isinstance(m, str) else m.path for m in self.modules}

        self.cache = Cache.CacheDict({}, alias=alias).save()

    def add(self, provider: str, value: Any):
        """
        Add a value to the cache with optional transformations.

        Args:
            provider: The key to store the value under
            value: The tensor value to store
        """

        # Match pattern like "x.y.z.key.i1" into groups
        match = re.match(r"^(.+)\.([^.]+)\.i(\d+)$", provider)

        if match is None:
            return

        module_path, key, iteration = match.groups()

        if key not in ("input", "output"):
            return

        key = "inputs" if key == "input" else key

        if ".source." in module_path:
            return

        if self.modules is not None:
            if module_path not in self.modules:
                return
            
        if (key == "output" and not self.include_output) or (key == "inputs" and not self.include_inputs):
            return

        if self.detach:
            value = util.apply(value, lambda x: x.detach(), torch.Tensor)

        if self.device is not None:
            value = util.apply(value, lambda x: x.to(self.device), torch.Tensor)

        if self.dtype is not None:
            value = util.apply(value, lambda x: x.to(self.dtype), torch.Tensor)

        if module_path not in self.cache:
            self.cache[module_path] = Cache.Entry(**{key: value})
        else:

            if isinstance(self.cache[module_path], Cache.Entry):

                if key == "output":
                    if self.cache[module_path].output is None:
                        self.cache[module_path].output = value
                    else:
                        self.cache[module_path] = [
                            self.cache[module_path],
                            Cache.Entry(output=value),
                        ]
                else:
                    # if the entry exists and the key is input always create a new entry
                    self.cache[module_path] = [
                        self.cache[module_path],
                        Cache.Entry(inputs=value),
                    ]
            else:
                if key == "output":
                    if self.cache[module_path][-1].output is None:
                        self.cache[module_path][-1].output = value
                    else:
                        self.cache[module_path].append(Cache.Entry(output=value))
                else:
                    self.cache[module_path].append(Cache.Entry(inputs=value))


class InterleavingTracer(Tracer):
    """
    Tracer that manages the interleaving of model execution and interventions.

    This class coordinates the execution of the model's forward pass and
    user-defined intervention functions through the Interleaver.
    """

    def __init__(
        self, fn: Callable, model: Envoy, *args, backend: Backend = None, **kwargs
    ):
        """
        Initialize an InterleavingTracer with a function and model.

        Args:
            fn: The function to execute (typically the model's forward pass)
            model: The model envoy to intervene on
            *args: Additional arguments to pass to the function
            **kwargs: Additional keyword arguments to pass to the function
        """

        self.fn = fn
        self.model = model

        self.mediators: List[Mediator] = []

        self.batcher = Batcher(**kwargs)

        self.user_cache: List[Cache] = list()

        super().__init__(*args, backend=backend)

        if not hasattr(self, "obj_var_name"):
            try:
                self.obj_var_name = self.info.node.items[0].context_expr.func.value.id
            except:
                self.obj_var_name = None

        if not hasattr(self, "tracer_var_name"):
            self.tracer_var_name = (
                self.info.node.items[0].optional_vars.id
                if self.info.node.items[0].optional_vars is not None
                else "__nnsight_tracer__"
            )

    def compile(self) -> Callable:
        """
        Compile the captured code block into a callable function.

        Returns:
            A callable function that executes the captured code block
        """

        # If Envoy has a default mediators ( created via Envoy.edit() ), add them
        if self.model._default_mediators:

            for mediators in self.model._default_mediators:

                self.mediators.append(mediators)
                self.batcher.batch_groups.append((-1, -1))

        # If positional arguments were passed directly to a tracer, assume one invoker
        if self.args:
            invoker = self.invoke(*self.args, **self.kwargs)
            invoker.info = self.info.copy()

            invoker.__exit__(ExitTracingException, None, None)

            self.info.source = ["    pass\n"]

        self.info.source = [
            f"def __nnsight_tracer_{id(self)}__(__nnsight_tracing_info__,{self.tracer_var_name}):\n",
            *self.info.source,
            f"    {self.tracer_var_name}.push()\n",
        ]

        self.args = tuple()

    def execute(self, fn: Callable):
        """
        Execute the compiled function with interventions.

        First executes the parent Tracer's execute method to set up the context,
        then creates an Interleaver to manage the interventions during model execution.

        Args:
            fn: The compiled function to execute
        """

        fn(self.info, self)

        args = self.batcher.batched_args
        kwargs = self.batcher.batched_kwargs

        self.batcher.batched_args = tuple()
        self.batcher.batched_kwargs = {}

        interleaver = Interleaver(self.mediators, self, batcher=self.batcher, user_cache=self.user_cache)

        try:
            self.model.interleave(interleaver, self.fn, *args, **kwargs)

            self.push(interleaver.state)
        finally:
            interleaver.state.clear()
            interleaver.check_cache_full()
                    

    ### Public API ####

    def invoke(self, *args, **kwargs):
        """
        Create an Invoker to capture and execute an intervention function.

        Args:
            *args: Additional arguments to pass to the intervention function
            **kwargs: Additional keyword arguments to pass to the intervention function

        Returns:
            An Invoker instance
        """
        # TODO make sure not already executing
        return Invoker(self, *args, **kwargs)

    def stop(self):
        """
        Raise an EarlyStopException to stop the execution of the model.
        """
        self.model._interleaver.current.stop()

    @property
    def iter(self):
        return IteratorProxy(self.model._interleaver)

    def all(self):
        return self.iter[:]

    def cache(
        self,
        modules: Optional[List[Union[Envoy, str]]] = None,
        device: Optional[torch.device] = torch.device("cpu"),
        dtype: Optional[torch.dtype] = None,
        detach: Optional[bool] = True,
        include_output: bool = True,
        include_inputs: bool = False
    ) -> Union[Dict, Object]:
        """
        Get or create a cache for storing intermediate values during tracing.

        Args:
            modules: Optional list of modules to cache, defaults to all modules
            device: Optional device to move tensors to, defaults to cpu
            dtype: Optional dtype to convert tensors to, defaults to None
            detach: Whether to detach tensors from computation graph, defaults to True
            include_output: Whether to include output in the cached activations
            include_inputs: Whether to include inputs in the cached activations

        Returns:
            A dictionary containing the cached values
        """

        alias_dict = {value: key for key, value in self.model._alias.rename.items()} if self.model._alias is not None else None

        if self.model._interleaver is None:
            self.user_cache.append(Cache(modules, device, dtype, detach, include_output, include_inputs, alias_dict))

            return self.user_cache[-1].cache

        self.model._interleaver.current.set_user_cache(
            Cache(modules, device, dtype, detach, include_output, include_inputs, alias_dict)
        )

        return self.model._interleaver.current.user_cache[-1].cache

    ### Serialization ###

    def __getstate__(self):
        """Get the state of the tracer for serialization."""
        state = super().__getstate__()
        state["fn"] = self.fn.__name__
        state["model"] = self.model
        state["tracer_var_name"] = self.tracer_var_name
        state["batcher"] = self.batcher
        state["mediators"] = self.mediators

        return state

    def __setstate__(self, state):
        """Set the state of the tracer for deserialization."""
        super().__setstate__(state)

        
        self.model = state["model"]
        self.fn = getattr(self.model, state["fn"])
        self.tracer_var_name = state["tracer_var_name"]
        self.mediators = state["mediators"]
        self.batcher = state["batcher"]

        self.obj_var_name = None
        self.user_cache = list()

class ScanningTracer(InterleavingTracer):
    """
    A tracer that runs the model in fake tensor mode to validate operations and inspect tensor shapes.
    
    This tracer uses PyTorch's FakeTensorMode to run the model without actual computation,
    allowing for shape validation and operation checking. It populates the _fake_inputs and 
    _fake_output attributes on each Envoy to store the shapes and types of tensors that would
    flow through the model during a real forward pass.
    """
    
    def execute(self, fn: Callable):
        """
        Execute the model in fake tensor mode.
        
        This method:
        1. Registers forward hooks on all modules to capture fake input/output
        2. Runs the model in fake tensor mode to validate operations
        3. Stores the fake inputs/outputs on each Envoy for later inspection
        
        Args:
            fn: The function to execute (typically the model's forward pass)
        """
        # Get all Envoys in the model
        envoys = self.model.modules()
                        
        hooks = []
        
        # Register hooks on each module to capture shapes
        for envoy in envoys:
            def _hook(
                module: torch.nn.Module,
                input: Any,
                input_kwargs: Dict,
                output: Any,
                envoy=envoy
            ):
                # Store the shapes/types of inputs and outputs on the Envoy
                envoy._fake_inputs = (input, input_kwargs)
                envoy._fake_output = output
                
            hooks.append(envoy._module.register_forward_hook(
                _hook, with_kwargs=True
            ))
        
        # Run the model in fake tensor mode
        with FakeTensorMode(
            allow_non_fake_inputs=True,  # Allow real tensors as input
            shape_env=ShapeEnv(assume_static_by_default=True),  # Assume static shapes
        ) as fake_mode:
            with FakeCopyMode(fake_mode):
                # Deep copy batched args/kwargs to avoid modifying originals
                self.batcher.batched_args = copy.deepcopy(self.batcher.batched_args)
                self.batcher.batched_kwargs = copy.deepcopy(self.batcher.batched_kwargs)
        
                # Execute the model in fake mode
                super().execute(fn)
                        
        # Clean up hooks
        for hook in hooks:
            hook.remove()