import weakref
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ...contexts import resolve_dependencies
from ...contexts.backends import Backend
from ...contexts.Tracer import Tracer
from ...tracing import protocols
from ...tracing.Graph import Graph
from ...util import WrapperModule
from ..mixins import GenerationMixin
from .executors.GPUExecutor import NNsightGPUExecutor
from .sampling import NNsightSamplingParams

try:
    from vllm.distributed import (destroy_distributed_environment,
                                  destroy_model_parallel,
                                  init_distributed_environment,
                                  initialize_model_parallel)
    from vllm.engine.arg_utils import EngineArgs
    from vllm.entrypoints.llm import LLM
    from vllm.model_executor.model_loader.loader import _initialize_model
except Exception as e:

    raise type(e)(
        "Install vllm in your environment to use it with NNsight. "
        + "https://docs.vllm.ai/en/latest/getting_started/installation.html"
    ) from e


class VLLMTracer(Tracer):

    def local_backend_execute(self) -> Graph:

        if not self.model._dispatched:
            self.model.dispatch_model()

        self.model: VLLM

        self.graph.reset()

        invoker_inputs = self._invoker_inputs

        # If ths graph has a Bridge, we need to check for Nodes in the input itself.
        if protocols.BridgeProtocol.has_bridge(self.graph):

            invoker_inputs = resolve_dependencies(invoker_inputs)

        self.graph.execute()

        inputs = []
        params = []

        for invoker_group, invoker_input in enumerate(invoker_inputs):
            invoker_input = invoker_input[0]

            if not type(invoker_input) is list:
                invoker_input = [invoker_input]

            for input in invoker_input:
                
                param = NNsightSamplingParams(
                    **self._kwargs,
                    intervention_graph=self.graph,
                    invoker_group=invoker_group,
                )

                inputs.append(input)
                params.append(param)

            llm_engine: LLM = self.model._model.llm_engine

        output = llm_engine.generate(inputs, sampling_params=params)

        graph = self.graph
        graph.alive = False

        if not isinstance(graph, weakref.ProxyType):
            self.graph = weakref.proxy(graph)

        return graph


class VLLM(GenerationMixin):
    """NNsight wrapper to conduct interventions on a vLLM inference engine.

    .. code-block:: python
        from nnsight.models.VLLM import VLLM
        from vllm import SamplingParams

        model = VLLM("gpt2")

        prompt = ["The Eiffel Tower is in the city of"]
        sampling_params = SamplingParams(temperature=0.0, top_p=0.95, stop=["."])

        with model.trace(prompt, sampling_params=sampling_params) as tracer:
            model.model.transformer.h[8].output[-1][:] = 0

            outputs = model.output.save()

        for output in outputs.value:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    """

    tracer_class = VLLMTracer

    class VLLModel(WrapperModule):
        """Pytorch Wrapper for the vLLM engine to work seamlessly with NNsight.

        Attributes:
            model (torch.nn.Module): Underlying model of the vLLM instance.
            llm_engine (vllm.LLM): vLLM inference engine instance.
        """

        def __init__(self, model:torch.nn.Module, llm_engine=None) -> None:
            
         

            super().__init__()

            for name, child in model.named_children():
                setattr(self, name, child)

            self.llm_engine = llm_engine
            
            self.logits = WrapperModule()
            self.tokens = WrapperModule()
            
    def __init__(
        self,
        model_key: Union[str, torch.nn.Module],
        *args,
       
        **kwargs,
    ) -> None:
      
        if isinstance(model_key, torch.nn.Module):
            
            model_key.logits = WrapperModule()
            model_key.tokens = WrapperModule()

        super().__init__(model_key, *args, **kwargs)

    def _load(self, repo_id: str, **kwargs) -> VLLModel:

        if self._model is None:

            # no parallelism during initialization
            kwargs["tensor_parallel_size"] = 1
            kwargs["pipeline_parallel_size"] = 1

            # creating vLLM Engine args
            engine_args = EngineArgs(
                model=repo_id,
                **kwargs,
            )

            # creating the vllm engine configuration
            engine_config_dict = engine_args.create_engine_config().to_dict()

            # starting the distributed environment
            init_distributed_environment(
                engine_config_dict["parallel_config"].world_size,
                0,
                "tcp://127.0.0.1:47303",
                0,
                backend="nccl",
            )

            # start tensor parallel group
            initialize_model_parallel(
                engine_config_dict["parallel_config"].tensor_parallel_size,
                engine_config_dict["parallel_config"].pipeline_parallel_size,
                "nccl",
            )

            # initialize the model
            model = _initialize_model(
                model_config=engine_config_dict["model_config"],
                load_config=engine_config_dict["load_config"],
                lora_config=None,
                cache_config=engine_config_dict["cache_config"],
                scheduler_config=engine_config_dict["scheduler_config"],
            )

            return VLLM.VLLModel(model)
        else:

            # destroy the distributed environment created from the initial model initialization
            destroy_model_parallel()
            destroy_distributed_environment()

            llm = LLM(
                repo_id, **kwargs, distributed_executor_backend=NNsightGPUExecutor
            )

            self._model.llm_engine = llm

            return self._model

    # def _prepare_inputs(self, *inputs: Union[List[str], str]) -> Tuple[Tuple[List[str]], int]:
    #     if isinstance(inputs[0], list):
    #         return inputs, len(inputs[0])
    #     else:
    #         return ([inputs[0]],), 1

    # def _batch_inputs(
    #     self,
    #     batched_inputs: Optional[Tuple[List[str]]],
    #     prepared_inputs: List[str],
    # ) -> Tuple[List[str]]:
    #     if batched_inputs is None:

    #         return (prepared_inputs, )

    #     return (batched_inputs[0] + prepared_inputs, )
