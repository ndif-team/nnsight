from typing import Any, Callable, Dict, List, Tuple
import weakref

from nnsight.intervention import InterventionHandler
from torch.nn.modules import Module


from ...contexts import resolve_dependencies
from ...contexts.backends import Backend
from ...contexts.Tracer import Tracer
from ...tracing import protocols
from ...tracing.Graph import Graph
from ...util import WrapperModule
from .executors.GPUExecutor import NNsightGPUExecutor
from .sampling import NNsightSamplingParams
from ..NNsightModel import NNsight
from ..mixins import RemoteableMixin
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


class VLLM(RemoteableMixin):
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
    
    
    def _load_meta(self, repo_id:str, *args, **kwargs):


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
            1,
            0,
            "tcp://127.0.0.1:47303",
            0,
            backend="gloo",
        )

        # start tensor parallel group
        initialize_model_parallel(backend="gloo")

        # initialize the model
        model = _initialize_model(
            engine_config_dict["model_config"],
            engine_config_dict["load_config"],
            None,
            engine_config_dict["cache_config"]
        )
        
        destroy_model_parallel()
        destroy_distributed_environment()

        return model
    
    def _load(self, repo_id: str, **kwargs):

        llm = LLM(
            repo_id, **kwargs, distributed_executor_backend=NNsightGPUExecutor
        )

        self.vllm_entrypoint = llm

        return self._model
    
    def _prepare_input(self, *args, **kwargs) -> Tuple[Tuple[Tuple[Any], Dict[str, Any]], int]:
        
        input = []
        
        for arg in args:
            
            if not type(arg) is list:
                arg = [arg]
                
            for prompt in arg:
                
                 param = NNsightSamplingParams(
                    **kwargs,
  
                )
                 
                 input.append((prompt,param))
                 
        return (input, ), {} 
                
            
    
    def _batch(self, batched_inputs: Tuple[Tuple[Any] | protocols.Dict[str, Any]] | None, inputs:List[Tuple[str, NNsightSamplingParams]]) -> Tuple[Tuple[Any] | protocols.Dict[str, Any]]:
        
        if batched_inputs is None:
            batched_inputs = [], {'invoker_group': 0}
        
        batched_inputs, kwargs = batched_inputs
        
        invoker_group = kwargs['invoker_group']
            
        for prompt, param in inputs:
            
            param.invoker_group = invoker_group
            
            batched_inputs.append((prompt, param))
            
        kwargs['invoker_group'] += 1
        
        return batched_inputs, kwargs
                
    
    def interleave(
        self,
        fn: Callable,
        intervention_graph: Graph,
        *args,
        intervention_handler: InterventionHandler = None,
        **kwargs,
    ) -> Any:
        
        
        
    
    
    def _execute(self, *args, **kwargs) -> weakref.Any:
        return super()._execute(*args, **kwargs)