from __future__ import annotations

import copy
from functools import wraps
import gc
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from torch._subclasses.fake_tensor import FakeTensorMode, FakeCopyMode

import accelerate
import torch
from transformers import AutoConfig, AutoModel

from .. import util
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

        self.model: torch.nn.Module = None

        # Handle passing in a pre-initialized model to wrap.
        # Therefore the NNsight model is "pre-dispatched".
        if isinstance(model_key, torch.nn.Module):
            self.model_key = model_key.__class__.__name__
            self.custom_model = True
            self.dispatched = True
            self.model = model_key

        logger.info(f"Initializing `{self.model_key}`...")

        if not self.custom_model:
            self.model: Module = self._load_meta(self.model_key, *args, **kwargs).to(
                "meta"
            )
        
        self.wrap()

        if self.dispatch:
            # Dispatch local_model on initialization vs lazy dispatching.
            self.dispatch_model()

        logger.info(f"Initialized `{self.model_key}`")

    def __repr__(self) -> str:
        """Wrapper of meta_model's representation as the NNsight model's representation.

        Returns:
            str: Representation.
        """
        return repr(self.model)

    def __getattr__(self, key: Any) -> Any:
        """Wrapper of meta_model's attributes to access Module's inputs and outputs.

        Returns:
            Any: Attribute.
        """
        return getattr(self.model, key)

    def interleave(
        self,
        fn: Callable,
        graph: Graph,
        *inputs: List[Any],
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
            graph (Graph): Intervention graph to interleave with model's computation graph.
            inputs (List[Any]): Inputs to give to function.

        Returns:
            Any: Output of model.
        """

        if not self.dispatched:
            self.dispatch_model()

        logger.info(f"Running `{self.model_key}`...")

        graph.compile(self.model)

        inputs, total_batch_size = self._prepare_inputs(*inputs)

        intervention_handler = InterventionHandler(graph, total_batch_size)

        with HookModel(
            self.model,
            list(graph.argument_node_names.keys()),
            input_hook=lambda activations, module_path: intervene(
                activations, module_path, "input", intervention_handler
            ),
            output_hook=lambda activations, module_path: intervene(
                activations, module_path, "output", intervention_handler
            ),
        ):
            output = fn(*inputs, **kwargs)

        logger.info(f"Completed `{self.model_key}`")

        gc.collect()
        torch.cuda.empty_cache()

        return output
    
    def wrap(self):

        self.model = Module.wrap(self.model)

        for name, module in list(self.model.named_modules()):

            if isinstance(module, (Module, torch.nn.ModuleList)):
                continue

            module = Module.wrap(module)

            # Set Module's module_path so they know their place in the Module tree.
            module.module_path = name

            setattr(self.model, name, module)

    def dispatch_model(self, *args, **kwargs) -> None:
        """Dispatched local_model using _load_local."""
        logger.info(f"Dispatching `{self.model_key}`...")
        
        self.model = self._load(self.model_key, *self.args, *args, **kwargs, **self.kwargs)

        self.wrap()

        self.dispatched = True

        logger.info(f"Dispatched `{self.model_key}`")

    def trace(
        self,
        *inputs: Tuple[Any],
        trace: bool = True,
        invoker_args: Dict[str, Any] = None,
        **kwargs,
    ) -> Union[Runner, Any]:

        runner = Runner(self, **kwargs)

        if len(inputs) > 0:

            invoker_args = invoker_args or {}

            if not trace:

                with runner:
                    with runner.invoke(*inputs, **invoker_args):

                        output = self.model.output.save()

                return output.value

            runner.invoke(*inputs, **invoker_args).__enter__()

        if not trace:

            raise ValueError("Can't execute on no inputs!")

        return runner

    ### NNsight VIRTUAL METHODS BELOW #####################################

    def _load_meta(self, model_key: str, *args, **kwargs) -> torch.nn.Module:
        """Virtual method to load the meta_model from scratch.

        Default implementation loads a config from AutoConfig.from_pretrained and loads a model from AutoModel.from_config.

        Args:
            model_key (str): String value used to initialize meta_model. Usually huggingface repo_id or checkpoint path.

        Returns:
            torch.nn.Module: Meta model.
        """
        config = AutoConfig.from_pretrained(model_key, *args, **kwargs)

        return AutoModel.from_config(config, trust_remote_code=True)

    def _load(self, model_key: str, *args, **kwargs) -> torch.nn.Module:
        """Virtual method to load the local_model from scratch.

        Default implementation loads a model from AutoModel.from_pretrained using self.config.

        Args:
            model_key (str): String value used to initialize local_model. Usually huggingface repo_id or checkpoint path.

        Returns:
            torch.nn.Module: Local model.
        """

        return accelerate.load_checkpoint_and_dispatch(self.model, model_key, **kwargs)

    def _execute(self, *prepared_inputs: Tuple[Any], **kwargs) -> Any:
        """Virtual method to run the local_model with some inputs.

        Default implementation util.applies moving all tensors to the device of the first parameter in local_model and passes the value into meta_model.

        Args:
            prepared_inputs (Tuple[Any]): Prepared inputs.
        """
        device = next(self.model.parameters()).device

        prepared_inputs = util.apply(
            prepared_inputs, lambda x: x.to(device), torch.Tensor
        )

        return self.model(
            *prepared_inputs,
            **kwargs,
        )

    def _prepare_inputs(self, *inputs: Tuple[Any], **kwargs) -> Tuple[Tuple[Any], int]:
        """Virtual method to prepare inputs before batching and execution and return batch size of prepared_inputs.

        Default implementation just returns inputs.

        Args:
            inputs (Tuple[Any]): Inputs to prepare for batching and execution.
            int: Batch size of prepared_inputs.

        Returns:
            Tuple[Tuple[Any], int]: Prepared inputs, batch size of inputs.
        """
        return inputs, 1

    def _batch_inputs(
        self,
        batched_inputs: Optional[Any],
        *prepared_inputs: Tuple[Any],
    ) -> Any:
        """Virtual method to batch together results from _prepare_inputs.

        Default implementation returns list of all prepared_inputs.

        Args:
            batched_inputs (Any): Current state of batched_inputs. Initially None.
            prepared_inputs (Tuple[Any]): Most recent result from _prepare_inputs.

        Returns:
            Any: Batched inputs.
        """

        if batched_inputs is None:
            batched_inputs = prepared_inputs

        else:
            batched_inputs = torch.concatenate((batched_inputs, prepared_inputs))

        return batched_inputs
