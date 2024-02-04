from __future__ import annotations

import copy
import gc
from typing import Any, Callable, Dict, List, Tuple, Union

import accelerate
import torch
from transformers import AutoConfig, AutoModel

from .. import util
from ..contexts.Runner import Runner
from ..intervention import HookModel, InterventionHandler, intervene
from ..logger import logger
from ..module import Module
from ..patching import Patch, Patcher
from ..tracing.Graph import Graph
from ..contexts.Tracer import Tracer


class NNsight:

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

        if isinstance(model_key, torch.nn.Module):
            self.model_key = model_key.__class__.__name__
            self.custom_model = True
            self.dispatched = True
            self.local_model = model_key

        logger.info(f"Initializing `{self.model_key}`...")

        with accelerate.init_empty_weights(include_buffers=True):
            if self.custom_model:

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

            else:
                self.meta_model: Module = self._load_meta(
                    self.model_key, *args, **kwargs
                ).to("meta")

        self.meta_model = Module.wrap(self.meta_model)

        for name, module in list(self.meta_model.named_modules()):

            if isinstance(module, (Module, torch.nn.ModuleList)):
                continue

            module = Module.wrap(module)

            module.module_path = name

            setattr(self.meta_model, name, module)

        if self.dispatch:
            self.dispatch_local_model()

        logger.info(f"Initialized `{self.model_key}`")

    def __repr__(self) -> str:
        return repr(self.meta_model)

    def __getattr__(self, key: Any) -> Any:
        return getattr(self.meta_model, key)

    def __call__(
        self,
        fn: Callable,
        inputs: Any,
        graph: Graph,
        *args,
        grad: bool = False,
        **kwargs,
    ) -> Any:

        if not self.dispatched:
            self.dispatch_local_model()

        logger.info(f"Running `{self.model_key}`...")

        graph.compile(self.local_model)

        inputs = self._prepare_inputs(inputs)

        _, total_batch_size = self._batch_inputs(inputs, None)

        intervention_handler = InterventionHandler(graph, total_batch_size)

        if not grad:

            torch.no_grad().__enter__()

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

        if not grad:

            torch.no_grad().__exit__(None, None, None)

        logger.info(f"Completed `{self.model_key}`")

        gc.collect()
        torch.cuda.empty_cache()

        return output

    def dispatch_local_model(self, *args, **kwargs) -> None:
        logger.info(f"Dispatching `{self.model_key}`...")

        self.local_model = self._load_local(
            self.model_key, *self.args, *args, **kwargs, **self.kwargs
        )

        self.dispatched = True

        logger.info(f"Dispatched `{self.model_key}`")

    def forward(self, *args, **kwargs) -> Runner:
        return Runner(self, *args, **kwargs)

    def trace(self, inputs: Any, *args, invoke_args={}, **kwargs) -> NNsight:

        runner = Runner(self, *args, **kwargs).__enter__()

        runner.invoke(inputs, **invoke_args).__enter__()

        return self

    def invoke(self, inputs: Any, *args, fwd_args={}, **kwargs) -> NNsight:

        return self.trace(inputs, invoke_args=kwargs, **fwd_args)

    def __enter__(self) -> NNsight:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.meta_model.tracer.__exit__(None, None, None)

    def _prepare_inputs(self, inputs: Any, **kwargs) -> Any:
        return inputs

    def _load_meta(self, model_key: str, *args, **kwargs) -> torch.nn.Module:

        self.config = AutoConfig.from_pretrained(model_key, *args, **kwargs)

        return AutoModel.from_config(self.config, trust_remote_code=True)

    def _load_local(self, model_key: str, *args, **kwargs) -> torch.nn.Module:

        return AutoModel.from_pretrained(model_key, *args, config=self.config, **kwargs)

    def _scan(self, prepared_inputs: Any, *args, **kwargs) -> None:

        device = torch.device("meta")

        prepared_inputs = util.apply(
            prepared_inputs, lambda x: x.clone().to(device), torch.Tensor
        )

        with accelerate.init_empty_weights(include_buffers=True):
            return self.meta_model(prepared_inputs, *args, **kwargs)

    def _execute(self, prepared_inputs: Any, *args, **kwargs) -> Any:

        device = next(self.local_model.parameters()).device

        prepared_inputs = util.apply(
            prepared_inputs, lambda x: x.to(device), torch.Tensor
        )

        return self.local_model(
            prepared_inputs,
            *args,
            **kwargs,
        )

    def _batch_inputs(
        self, prepared_inputs: List[Any], batched_inputs: Any
    ) -> Tuple[Any, int]:

        if batched_inputs is None:
            batched_inputs = prepared_inputs

        else:
            batched_inputs = [*batched_inputs, prepared_inputs]

        return batched_inputs, len(prepared_inputs)

    def _example_input(self) -> Any:

        raise NotImplementedError()