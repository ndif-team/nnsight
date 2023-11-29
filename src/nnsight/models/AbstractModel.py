from __future__ import annotations

import copy
import gc
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Union

import accelerate
import torch
from torch.utils.hooks import RemovableHandle

from ..alteration import REPOID_TO_ALTERATION
from ..contexts.DirectInvoker import DirectInvoker
from ..contexts.Runner import Runner
from ..editing.Editor import Edit, Editor
from ..editing.GraphEdit import GraphEdit
from ..editing.WrapperModuleEdit import WrapperModuleEdit
from ..intervention import HookModel, intervene
from ..logger import logger
from ..module import Module
from ..patching import Patch, Patcher
from ..tracing.Graph import Graph


class AbstractModel(ABC):
    """Abstract class to be implemented for PyTorch models wishing to gain this package's functionality.

    Attributes:
        repoid_path_clsname (str): Hugging face repo id of model to load, path to checkpoint, or class name of custom model.
        args (List[Any]): Positional arguments used to initialize model.
        kwargs (Dict[str,Any]): Keyword arguments used to initialize model.
        alter (bool): If to check for alterations and apply them to the model. Defaults to True.
        dispatched (bool): If the local_model has bee loaded yet.
        dispatch (bool): If to load and dispatch model on init. Defaults to False.
        custom_model (bool): If the value passed to repoid_path_model was a custom model.
        meta_model (nnsight.Module): Version of the root model where all parameters and tensors are on the 'meta'
            device. All modules are wrapped in nnsight.Module adding interleaving operation functionality.
        local_model (torch.nn.Module): Locally loaded and dispatched model. Only loaded and dispatched on first use.
            This is the actual model that is ran with hooks added to it to enter the intervention graph.
    """

    def __init__(
        self,
        repoid_path_model: Union[str, torch.nn.Module],
        *args,
        dispatch: bool = False,
        alter: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.repoid_path_clsname = repoid_path_model
        self.args = args
        self.kwargs = kwargs
        self.alter = alter
        self.dispatch = dispatch
        self.dispatched = False
        self.custom_model = False
        self.meta_model: Module = None
        self.local_model: torch.nn.Module = None
        self.edits: List[Edit] = list()

        # If using a custom passed in model.
        if isinstance(repoid_path_model, torch.nn.Module):
            self.repoid_path_clsname = repoid_path_model.__class__.__name__
            self.custom_model = True
            self.dispatched = True
            self.local_model = repoid_path_model

        logger.info(f"Initializing `{self.repoid_path_clsname}`...")

        # If alter and alteration exist, use alteration patcher context while loading module.
        with self.alteration() if self.alter else Patcher():
            # Use accelerate and .to('meta') to assure tensors are loaded to 'meta' device
            with accelerate.init_empty_weights(include_buffers=True):
                if self.custom_model:
                    # Need to force parameters when deepcopied, to instead create a meta tensor of the same size/dtype
                    def meta_deepcopy(self, memo):
                        if id(self) in memo:
                            return memo[id(self)]
                        else:
                            result = type(self)(
                                torch.empty_like(
                                    self.data, dtype=self.data.dtype, device="meta"
                                ),
                                self.requires_grad,
                            )
                            memo[id(self)] = result
                            return result

                    # Patching Parameter __deepcopy__
                    with Patcher() as patcher:
                        patcher.add(
                            Patch(
                                torch.nn.parameter.Parameter.__deepcopy__, meta_deepcopy
                            )
                        )

                        self.meta_model: Module = Module.wrap(
                            copy.deepcopy(self.local_model).to("meta")
                        )
                else:
                    self.meta_model: Module = Module.wrap(
                        self._load_meta(self.repoid_path_clsname, *args, **kwargs).to(
                            "meta"
                        )
                    )

        # Wrap all modules in our Module class.
        for name, module in self.meta_model.named_children():
            module = Module.wrap(module)

            setattr(self.meta_model, name, module)

        # Set module_path attribute so Modules know their place.
        for name, module in self.meta_model.named_modules():
            module.module_path = name

        # Run initial dummy string to populate Module shapes, dtypes etc
        self._scan(self._prepare_inputs(self._example_input()))

        if self.dispatch:
            self.dispatch_local_model()

        logger.info(f"Initialized `{self.repoid_path_clsname}`")

    def __repr__(self) -> str:
        return repr(self.meta_model)

    def __getattr__(self, key: Any) -> Any:
        """Allows access of sub-modules on meta_model directly from AbstractModel object

        Args:
            key (Any): Key.

        Returns:
            Any: Attribute.
        """
        return getattr(self.meta_model, key)

    def __call__(
        self,
        fn: Callable,
        inputs: Any,
        graph: Graph,
        *args,
        edits: List[Edit] = None,
        inference: bool = True,
        **kwargs,
    ) -> Any:
        """Runs some function with some inputs and some graph with the appropriate contexts for this model.

        Loads and dispatched local_model if not already done so.

        Args:
            fn (Callable): Function to run.
            inputs (Any): Inputs to give to function.
            graph (Graph): Intervention graph to interleave with model's computation graph.
            inference (bool): If running in inference mode. Defaults to True.

        Returns:
            Any: Output of model.
        """
        if edits is None:
            edits = self.edits

        # If local_model not yet loaded, do so.
        if not self.dispatched:
            self.dispatch_local_model()

        with Editor(self, edits):
            increment_hook = self._register_increment_hook(
                lambda module, input, output: graph.increment()
            )

            # The intervention graph for running a Model will have the modules that are involved
            # in the graph's argument_node_names.
            modules = set(
                [
                    ".".join(name.split(".")[:-2])
                    for name in graph.argument_node_names.keys()
                ]
            )

            logger.info(f"Running `{self.repoid_path_clsname}`...")

            # Send local_model to graph to re-compile
            graph.compile(self.local_model)
  
            inputs = self._prepare_inputs(inputs)

            with torch.inference_mode(mode=inference):
                with HookModel(
                    self.local_model,
                    list(modules),
                    input_hook=lambda activations, module_path: intervene(
                        activations, module_path, graph, "input"
                    ),
                    output_hook=lambda activations, module_path: intervene(
                        activations, module_path, graph, "output"
                    ),
                    backward_input_hook=lambda activations, module_path: intervene(
                        activations, module_path, graph, "backward_input"
                    ),
                    backward_output_hook=lambda activations, module_path: intervene(
                        activations, module_path, graph, "backward_output"
                    ),
                ):
                    output = fn(inputs, *args, **kwargs)

            increment_hook.remove()

            logger.info(f"Completed `{self.repoid_path_clsname}`")

            gc.collect()
            torch.cuda.empty_cache()

        return output

    def dispatch_local_model(self, *args, **kwargs) -> None:
        with self.alteration() if self.alter else Patcher():
            self.local_model = self._load_local(
                self.repoid_path_clsname, *self.args, *args, **kwargs, **self.kwargs
            )

        self.dispatched = True

    def generate(self, *args, **kwargs) -> Runner:
        """Returns a Runner context for this model's _generation method.

        Runner contexts are used to trace and interleave operations on the model's computation graph.

        Arguments passed to ``.generate()`` are passed downstream to the model specific _generation method.

        Runners are used in tandem with their Invoker contexts to enter inputs for operation tracing and execution.

        The output of the run is ultimately saved in the generator's ``.output`` attribute.

        Returns:
            Runner: Runner.

        Examples:

            A simple entering of a generation context on a language model, and running a prompt with no interventions:

            .. code-block:: python

                with model.generate(max_new_tokens=1) as generator:
                    with generator.invoke('The Eiffel Tower is in the city of') as invoker:
                        pass

                print(generator.output)

            Keyword arguments like 'max_new_tokens' are model specific and in this case limits the amount of tokens to predict to one.

            See the Runner docs for more information.

        """
        return Runner(self, *args, **kwargs, generation=True)

    def forward(self, *args, **kwargs) -> Runner:
        """Returns a Runner context for this model's _forward method.

        Runner contexts are used to trace and interleave operations on the model's computation graph.

        Arguments passed to ``.forward()`` are passed downstream to the model specific _forward method.

        Runners are used in tandem with their Invoker contexts to enter inputs for operation tracing and execution.

        The output of the run is ultimately saved in the runner's ``.output`` attribute.

        Returns:
            Runner: Runner.

        Examples:

            A simple entering of a forward context on a language model, and running a prompt with no interventions:

            .. code-block:: python

                with model.forward() as runner:
                    with runner.invoke('The Eiffel Tower is in the city of') as invoker:
                        pass

                print(runner.output)

            See the Runner docs for more information.

        """
        return Runner(self, *args, **kwargs, generation=False)

    def invoke(self, *args, **kwargs):
        """Returns a Runner context for this model's _forward method. Also acts as an invocation for the runner allowing you to create a Runner anf Invocation context in one go.
        Gives option to create Runner context and call an invoke on it in one call. Adds an extra 'fwd_args' kwarg to allow the args normally passed to ``.forward()``
        Returns:
            Runner: Runner.

        Examples:

            A simple entering of a forward context on a language model, and running a prompt with no interventions:

            .. code-block:: python

                with model.invoke('The Eiffel Tower is in the city of', fwd_args={}) as invoker:
                    pass

                print(runner.output)

            See the Runner docs for more information.

        """
        return DirectInvoker(self, *args, **kwargs)

    def alteration(self) -> Patcher:
        return REPOID_TO_ALTERATION.get(self.repoid_path_clsname, Patcher())

    def modulize(self, module: Module, node_name: str, module_name: str) -> None:
        """_summary_

        Args:
            module (Module): _description_
            node_name (str): _description_
            module_name (str): _description_
        """

        # Create a WrapperModuleEdit which just adds a WrapperModule to an existing module at the given module_name.
        wme = WrapperModuleEdit(module.module_path, module_name)
        # Wrap with our Module and update new attributes.
        wme.wrapper: Module = Module.wrap(wme.wrapper)
        wme.wrapper.module_path = f"{module.module_path}.{module_name}"
        wme.wrapper.tracer = module.tracer
        wme.wrapper.output_shape = module.output_shape
        # Carry out the edit on the meta_model.
        wme.edit(self.meta_model)

        # Get/create the execution graph for the module's forward method.
        graph = module.graph

        # Add two proxies/nodes, one to get the new WrapperModule we added and another to call it with the data from the original module.
        # Passing the data through the wrapper module allows hooking of the module's output like usual.
        module_proxy = getattr(graph.module_proxy, module_name)
        module_proxy(graph.nodes[node_name])

        # Create and carry out the edit on the meta_model.
        ge = GraphEdit(module.module_path, module.graph)
        ge.edit(self.meta_model)

        # Append to self.edits so when we call the local model, we temporarily edit the module in the same way as the meta model.
        self.edits.append(wme)
        self.edits.append(ge)

    @abstractmethod
    def _prepare_inputs(self, inputs: Any, **kwargs) -> Any:
        """Abstract method which prepares inputs. To be implemented by inheritors.

        Args:
            inputs (Any): Inputs.

        Returns:
            Any: Prepared inputs.
        """
        raise NotImplementedError()

    @abstractmethod
    def _load_meta(self, repoid_or_path: str, *args, **kwargs) -> torch.nn.Module:
        """
        Abstract method to initialize meta_model. To be implemented by inheritors.

        Args:
            repoid_or_path (str): Huggingface repo id or path to checkpoint.

        Returns:
            torch.nn.Module: Meta version of model
        """
        raise NotImplementedError()

    @abstractmethod
    def _load_local(self, repoid_or_path, *args, **kwargs) -> torch.nn.Module:
        """
        Abstract method to initialize and dispatch the local_model. To be implemented by inheritors.

        Args:
            repoid_or_path (str): Huggingface repo id or path to checkpoint.

        Returns:
            torch.nn.Module: Local version of model
        """
        raise NotImplementedError()

    @abstractmethod
    def _scan(self, inputs, *args, **kwargs) -> None:
        """
        Abstract method to directly call the meta_model and therefore populate the input/output shapes etc. To be implemented by inheritors.

        Used for tracing operations and their input/output shapes/dtypes.

        Args:
            inputs (Any): Inputs.
        """
        raise NotImplementedError()

    @abstractmethod
    def _forward(self, inputs, *args, **kwargs) -> Any:
        """
        Abstract method to directly call the local_model. To be implemented by inheritors.

        Args:
            inputs (Any): Inputs.

        Returns:
            Any: Output.
        """
        raise NotImplementedError()

    @abstractmethod
    def _generation(self, inputs, *args, **kwargs) -> Any:
        """
        Abstract method to do iterative generation on the local_model. To be implemented by inheritors.

        Args:
            inputs (Any): Inputs.

        Returns:
            Any: Output.
        """
        raise NotImplementedError()

    @abstractmethod
    def _register_increment_hook(self, hook: Callable) -> RemovableHandle:
        """Abstract method to hook a function on the local_model on the main module that is incremented during generation.

        Args:
            hook (Callable): Increment hook. Probably from the Generator context.

        Returns:
            RemovableHandle: Handle to remove the applied hook after generation is done.
        """
        raise NotImplementedError()

    @abstractmethod
    def _example_input(self) -> Any:
        """Abstract to provide an example input to be used with ``._scan(...)``

        Returns:
            Any: Example input.
        """
        raise NotImplementedError()

    @abstractmethod
    def _batched_inputs(self, prepared_inputs: Any) -> List[Any]:
        """Abstract to return a version of the prepared inputs from ``._prepare_inputs(...)`` that can be batched with others.
        For example with a LanguageModel, prepare_inputs returns a dictionary. The implementation for this method just returns the 'input_ids'.

        Args:
            prepared_inputs (Any): Inputs from ``._prepare_inputs(...)``

        Returns:
            List[Any]: Batched version of prepared_inputs.
        """
        raise NotImplementedError()
