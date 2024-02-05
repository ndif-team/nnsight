from __future__ import annotations

import copy
from functools import wraps
import gc
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import accelerate
import torch
from transformers import AutoConfig, AutoModel

from .. import util
from ..contexts.Invoker import Invoker
from ..contexts.Runner import Runner
from ..intervention import HookModel, InterventionHandler, InterventionProxy, intervene
from ..logger import logger
from ..module import Module
from ..patching import Patch, Patcher
from ..tracing.Graph import Graph


class NNsight:
    """Main class to be implemented as a wrapper for PyTorch models wishing to gain this package's functionality. Can be used "as is" for basic models.

    Class Attributes:

        proxy_class (Type[InterventionProxy]): InterventionProxy like type to use as a Proxy for this Model's inputs and outputs. Can have Model specific functionality added to a new sub-class.

    Attributes:
        model_key (str): String representing what kind of model this is. Usually hugging face repo id of model to load, path to checkpoint, or class name of custom model.
        args (List[Any]): Positional arguments used to initialize model.
        kwargs (Dict[str,Any]): Keyword arguments used to initialize model.
        dispatched (bool): If the local_model has bee loaded yet.
        dispatch (bool): If to load and dispatch model on init. Defaults to False.
        custom_model (bool): If the value passed to repoid_path_model was a custom model.
        meta_model (nnsight.Module): Version of the root model where all parameters and tensors are on the 'meta'
            device. All modules are wrapped in nnsight.Module adding interleaving operation functionality.
        local_model (torch.nn.Module): Locally loaded and dispatched model. Only loaded and dispatched on first use.
            This is the actual model that is ran with hooks added to it to enter the intervention graph.
    """

    proxy_class: Type[InterventionProxy] = InterventionProxy

    def __init__(
        self,
        model_key: Union[str, torch.nn.Module],
        *args,
        dispatch: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.model_key = model_key

        self.args = args
        self.kwargs = kwargs

        self.dispatch = dispatch

        self.dispatched = False
        self.custom_model = False

        self.local_model: torch.nn.Module = None

        # Handle passing in a pre-initialized model to wrap.
        # Therefore the NNsight model is "pre-dispatched".
        if isinstance(model_key, torch.nn.Module):
            self.model_key = model_key.__class__.__name__
            self.custom_model = True
            self.dispatched = True
            self.local_model = model_key

        logger.info(f"Initializing `{self.model_key}`...")

        # We want the meta_model parameters to be loaded to 'meta'.
        with accelerate.init_empty_weights(include_buffers=True):

            # If a pre-initialized model was passed in, we want to deepcopy a 'meta' version for the meta_model.
            if self.custom_model:

                # We want to wrap the deepcopy logic of Tensors and Parameters to be on the 'meta' device.
                # So we don't double the memory of the model briefly by cloning the whole thing and then moving to 'meta'.
                with Patcher() as patcher:

                    patcher.add(
                        Patch(
                            torch.nn.parameter.Parameter,
                            util.meta_deepcopy,
                            "__deepcopy__",
                        )
                    )

                    patcher.add(Patch(torch.Tensor, util.meta_deepcopy, "__deepcopy__"))

                    self.meta_model: torch.nn.Module = copy.deepcopy(
                        self.local_model
                    ).to("meta")

            # Otherwise use _load_meta.
            else:
                self.meta_model: Module = self._load_meta(
                    self.model_key, *args, **kwargs
                ).to("meta")

        # Wrap root meta_model in nnsight's Module wrapper class.
        self.meta_model = Module.wrap(self.meta_model)

        # Wrap meta_model's submodules in nnsight's Module wrapper class.
        for name, module in list(self.meta_model.named_modules()):

            if isinstance(module, (Module, torch.nn.ModuleList)):
                continue

            module = Module.wrap(module)

            # Set Module's module_path so they know their place in the Module tree.
            module.module_path = name

            setattr(self.meta_model, name, module)

        if self.dispatch:
            # Dispatch local_model on initialization vs lazy dispatching.
            self.dispatch_local_model()

        logger.info(f"Initialized `{self.model_key}`")

    def __repr__(self) -> str:
        """Wrapper of meta_model's representation as the NNsight model's representation.

        Returns:
            str: Representation.
        """
        return repr(self.meta_model)

    def __getattr__(self, key: Any) -> Any:
        """Wrapper of meta_model's attributes to access Module's inputs and outputs.

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
        **kwargs,
    ) -> Any:
        """Runs some function with some inputs and some graph with the appropriate contexts for this model.

        Loads and dispatched local_model if not already done so.

        Re-compiles Graph with local_model to prepare for a new execution of it.

        Runs _prepare_inputs and _batch_inputs one last time to get total_batch_size.

        Handles adding and removing hooks on Modules and tracking number of times a Module has been called.

        After execution, garbage collects and clears cuda memory.

        Args:
            fn (Callable): Function or method to run.
            inputs (Any): Inputs to give to function.
            graph (Graph): Intervention graph to interleave with model's computation graph.

        Returns:
            Any: Output of model.
        """

        if not self.dispatched:
            self.dispatch_local_model()

        logger.info(f"Running `{self.model_key}`...")

        graph.compile(self.local_model)

        inputs, total_batch_size = self._prepare_inputs(inputs)

        intervention_handler = InterventionHandler(graph, total_batch_size)

        with HookModel(
            self.local_model,
            list(graph.argument_node_names.keys()),
            input_hook=lambda activations, module_path: intervene(
                activations, module_path, "input", intervention_handler
            ),
            output_hook=lambda activations, module_path: intervene(
                activations, module_path, "output", intervention_handler
            ),
        ):
            output = fn(inputs, *args, **kwargs)

        logger.info(f"Completed `{self.model_key}`")

        gc.collect()
        torch.cuda.empty_cache()

        return output

    def dispatch_local_model(self, *args, **kwargs) -> None:
        """Dispatched local_model using _load_local."""
        logger.info(f"Dispatching `{self.model_key}`...")

        self.local_model = self._load_local(
            self.model_key, *self.args, *args, **kwargs, **self.kwargs
        )

        self.dispatched = True

        logger.info(f"Dispatched `{self.model_key}`")

    def trace(self, *args, **kwargs) -> Runner:
        """Returns a Runner context for this model's ``._execute()`` method.

        Runner contexts are used to trace and interleave operations on the model's computation graph.

        Arguments passed to ``.trace()`` are passed downstream to the model specific ``._execute()`` method.

        Runners are used in tandem with their Invoker contexts to enter inputs for operation tracing and execution.

        Returns:
            Runner: Runner.

        Examples:

            A simple entering of a runner context on a language model, and running a prompt with no interventions:

            .. code-block:: python

                with model.trace() as tracer:
                    with tracer.invoke('The Eiffel Tower is in the city of') as invoker:
                        output = model.output.save()

                print(output.value)

            See the Runner docs for more information.

        """
        return Runner(self, *args, **kwargs)

    def invoke(
        self,
        inputs: Any,
        *args,
        trace: bool = True,
        fwd_args: Dict[str, Any] = {},
        **kwargs,
    ) -> Union[Invoker, Any]:
        """Creates a Runner context for this model's ``._execute()`` method and creates an Invoker context off of it for the given input and returns it.

        Keyword arguments usually passed to ``.trace()`` (and therefore downstream to ``._execute()``) can be given as a dictionary with the fwd_args keyword argument.

        This enables the option to directly invoke and trace the model, while providing access to the created Invoker object.

        With `trace=False`, does not trace and runs without a context manager. Directly runs and returns the output of the model.

        Args:
            inputs (Any): Inputs.
            trace (bool): If to not trace and run without a context manager. Directly runs and returns the output of the model.
            fwd_args (Dict[str, Any]): Dictionary as keyword arguments to pass to Runner.

        Returns:
            Invoker: Invoker.

        Examples:

            A simple entering of a forward context on a language model, and running a prompt with no interventions:

            .. code-block:: python

                with model.invoke('The Eiffel Tower is in the city of') as invoker:
                    output = model.output.save()

                print(output.value)

            With `trace=False`:

            .. code-block python

                output = model.invoke('The Eiffel Tower is in the city of', trace=False)

                print(output)

            See the Runner and Invoker docs for more information.

        """

        # Create and enter runner context.
        runner = Runner(self, validate=trace, **fwd_args).__enter__()

        # Create invoker context to return.
        invoker = runner.invoke(inputs, *args, scan=trace, **kwargs)

        if not trace:

            with invoker:

                output = self.meta_model.output.save()

            runner.__exit__(None, None, None)

            return output.value

        # We need the Runner to exit along with the Invoker so we combine the __exit__ methods and replace.
       
        def on_exit():
            runner.__exit__(None,None, None)

        setattr(invoker, 'on_exit', on_exit)

        return invoker

    def forward(
        self,
        inputs: Any,
        *args,
        trace: bool = True,
        invoke_args: Dict[str, Any] = {},
        **kwargs,
    ) -> Union[NNsight, Any]:
        """Creates a Runner context for this model's ``._execute()`` method and creates an Invoker context off of it for the given input and returns not the Invoker, but the NNsight model.

        Keyword arguments usually passed to the Invoker (via ``.invoke()``) (and therefore downstream to ``.prepare_inputs()``) can be given as a dictionary with the invoker_args keyword argument.

        This enables the option to directly invoke and trace the model, while providing access to the given NNsight model.

        With `trace=False`, does not trace and runs without a context manager. Directly runs and returns the output of the model.

        Args:
            inputs (Any): Inputs.
            trace (bool): If to not trace and run without a context manager. Directly runs and returns the output of the model.
            invoker_args (Dict[str, Any]): Dictionary as keyword arguments to pass to Invoker.

        Returns:
            NNsight: NNsight model.

        Examples:

            A simple entering of a forward context on a language model, and running a prompt with no interventions:

            .. code-block:: python

                with NNsight(model).forward('The Eiffel Tower is in the city of') as model:
                    output = model.output.save()

            With `trace=False`:

            .. code-block python

                output = NNsight(model).forward('The Eiffel Tower is in the city of', trace=False)

                print(output)

            See the Runner and Invoker docs for more information.

        """

        invoker_or_output = self.invoke(
            inputs, fwd_args=kwargs, trace=trace, **invoke_args
        )

        if isinstance(invoker_or_output, Invoker):
            invoker_or_output.__enter__()
        else:
            return invoker_or_output

        return self

    def __enter__(self) -> NNsight:
        """Used with ``.forward()``.
        No need to do anything as both the Runner and Invoker contexts are already entered.

        Returns:
            NNsight: NNsight model.
        """

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Used with ``.forward()``.
        We can access the current Tracer object from the meta_model and exit it from there.
        """
        if isinstance(exc_val, BaseException):
            raise exc_val

        self.meta_model.tracer.__exit__(None, None, None)

    ### NNsight VIRTUAL METHODS BELOW #####################################

    def _load_meta(self, model_key: str, *args, **kwargs) -> torch.nn.Module:
        """Virtual method to load the meta_model from scratch.

        Default implementation loads a config from AutoConfig.from_pretrained and loads a model from AutoModel.from_config.

        Args:
            model_key (str): String value used to initialize meta_model. Usually huggingface repo_id or checkpoint path.

        Returns:
            torch.nn.Module: Meta model.
        """
        self.config = AutoConfig.from_pretrained(model_key, *args, **kwargs)

        return AutoModel.from_config(self.config, trust_remote_code=True)

    def _load_local(self, model_key: str, *args, **kwargs) -> torch.nn.Module:
        """Virtual method to load the local_model from scratch.

        Default implementation loads a model from AutoModel.from_pretrained using self.config.

        Args:
            model_key (str): String value used to initialize local_model. Usually huggingface repo_id or checkpoint path.

        Returns:
            torch.nn.Module: Local model.
        """

        return AutoModel.from_pretrained(model_key, *args, config=self.config, **kwargs)

    def _scan(self, prepared_inputs: Any, *args, **kwargs) -> None:
        """Virtual method to run the meta_model with some input in order to compute the shapes of tensors during execution of the input.

        Default implementation util.applies moving all tensors to the 'meta' device and passes the value into meta_model.

        Args:
            prepared_inputs (Any): Prepared inputs.
        """
        device = torch.device("meta")

        prepared_inputs = util.apply(
            prepared_inputs, lambda x: x.clone().to(device), torch.Tensor
        )

        with accelerate.init_empty_weights(include_buffers=True):
            return self.meta_model(prepared_inputs, *args, **kwargs)

    def _execute(self, prepared_inputs: Any, *args, **kwargs) -> Any:
        """Virtual method to run the local_model with some input.

        Default implementation util.applies moving all tensors to the device of the first parameter in local_model and passes the value into meta_model.

        Args:
            prepared_inputs (Any): Prepared inputs.
        """
        device = next(self.local_model.parameters()).device

        prepared_inputs = util.apply(
            prepared_inputs, lambda x: x.to(device), torch.Tensor
        )

        return self.local_model(
            prepared_inputs,
            *args,
            **kwargs,
        )

    def _prepare_inputs(self, inputs: Any, *args, **kwargs) -> Tuple[Any, int]:
        """Virtual method to prepare inputs before batching and execution and return batch size of prepared_inputs.

        Default implementation just returns inputs.

        Args:
            inputs (Any): Inputs to prepare for batching and execution.
            int: Batch size of prepared_inputs.

        Returns:
            Any: Prepared inputs.
        """
        return inputs, 1

    def _batch_inputs(
        self, prepared_inputs: List[Any], batched_inputs: Optional[Any]
    ) -> Any:
        """Virtual method to batch together results from _prepare_inputs.

        Default implementation returns list of all prepared_inputs.

        Args:
            prepared_inputs (List[Any]): Most recent result from _prepare_inputs.
            batched_inputs (Any): Current state of batched_inputs. Initially None.

        Returns:
            Any: Batched inputs.
        """

        if batched_inputs is None:
            batched_inputs = prepared_inputs

        else:
            batched_inputs = torch.concatenate((batched_inputs, prepared_inputs))

        return batched_inputs
