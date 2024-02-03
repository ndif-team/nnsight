from __future__ import annotations

import copy
import gc
from typing import Any, Callable, Dict, List, Tuple, Union

import accelerate
import torch
from torch.utils.hooks import RemovableHandle
from transformers import AutoConfig, AutoModel

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


class NNsightModel:
    """Class to be implemented for PyTorch models wishing to gain this package's functionality. Can be used "as is" for basic models.

    Attributes:
        repoid_path_clsname (str): Hugging face repo id of model to load, path to checkpoint, or class name of custom model.
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

    def __init__(
        self,
        repoid_path_model: Union[str, torch.nn.Module],
        *args,
        dispatch: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.repoid_path_clsname = repoid_path_model
        self.args = args
        self.kwargs = kwargs
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

        # Use accelerate and .to('meta') to assure tensors are loaded to 'meta' device
        with accelerate.init_empty_weights(include_buffers=True):
            if self.custom_model:
                # Need to force parameters when deepcopied, to instead create a meta tensor of the same size/dtype
                def meta_deepcopy(self: torch.nn.parameter.Parameter, memo):
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
                            torch.nn.parameter.Parameter, meta_deepcopy, "__deepcopy__"
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
        """Allows access of sub-modules on meta_model

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
        grad: bool = True,
        **kwargs,
    ) -> Any:


        if not self.dispatched:
            self.dispatch_local_model()

        logger.info(f"Running `{self.repoid_path_clsname}`...")

        graph.compile(self.local_model)

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad(mode=grad):
            with HookModel(
                self.local_model,
                list(graph.argument_node_names.keys()),
                input_hook=lambda activations, module_path: intervene(
                    activations, module_path, graph, "input"
                ),
                output_hook=lambda activations, module_path: intervene(
                    activations, module_path, graph, "output"
                )
            ):
                output = fn(inputs, *args, **kwargs)

        logger.info(f"Completed `{self.repoid_path_clsname}`")

        gc.collect()
        torch.cuda.empty_cache()

        return output

    def dispatch_local_model(self, *args, **kwargs) -> None:
        logger.info(f"Dispatching `{self.repoid_path_clsname}`...")

        self.local_model = self._load_local(
            self.repoid_path_clsname, *self.args, *args, **kwargs, **self.kwargs
        )

        self.dispatched = True

        logger.info(f"Dispatched `{self.repoid_path_clsname}`")


    def forward(self, *args, **kwargs) -> Runner:

        return Runner(self, *args, **kwargs)

    def invoke(self, *args, **kwargs):

        return DirectInvoker(self, *args, **kwargs)

    def _prepare_inputs(self, inputs: Any, **kwargs) -> Any:

        return inputs

    def _load_meta(self, repoid_or_path: str, *args, **kwargs) -> torch.nn.Module:

        self.config = AutoConfig.from_pretrained(repoid_or_path, *args, **kwargs)

        return AutoModel.from_config(self.config, trust_remote_code=True)

    def _load_local(self, repoid_or_path, *args, **kwargs) -> torch.nn.Module:

        return AutoModel.from_pretrained(
            repoid_or_path, *args, config=self.config, **kwargs
        )

    def _scan(self, prepared_inputs, *args, **kwargs) -> None:

        device = torch.device('meta')

        prepared_inputs = util.apply(prepared_inputs, lambda x : x.to(device), torch.Tensor)

        with accelerate.init_empty_weights(include_buffers=True):
            return self.meta_model(**prepared_inputs.copy().to("meta"))

    def _execute(self, prepared_inputs, *args, **kwargs) -> Any:

        device = next(self.local_model.parameters()).device

        prepared_inputs = util.apply(prepared_inputs, lambda x : x.to(device), torch.Tensor)

        return self.local_model(
            prepared_inputs,
            *args,
            **kwargs,
        )

    def _batch_inputs(
        self, prepared_inputs: Any, batched_inputs: Any
    ) -> Tuple[Any, int]:

        if batched_inputs is None:
            batched_inputs = prepared_inputs

        else:
            batched_inputs = [*batched_inputs, prepared_inputs]

        return batched_inputs, len(prepared_inputs)
