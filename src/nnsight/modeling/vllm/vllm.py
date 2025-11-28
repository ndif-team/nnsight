from ... import NNS_VLLM_VERSION

import torch
import vllm 

from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Tuple,
                    Union)

from vllm import LLM, envs
from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel,
                              init_distributed_environment,
                              initialize_model_parallel)
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.llm import LLM

from ...intervention.envoy import Envoy
from ...intervention.tracing.util import push_variables
from ...util import WrapperModule
from ..mixins import RemoteableMixin
from .sampling import NNsightSamplingParams
from ...intervention.tracing.globals import Globals
from .engines.engine import NNsightLLMEngine
if TYPE_CHECKING:
    from torch.nn import Module

    from vllm.transformers_utils.tokenizer import AnyTokenizer
    
    
envs.VLLM_ENABLE_V1_MULTIPROCESSING = False


class VLLM(RemoteableMixin):
    """NNsight wrapper to conduct interventions on a vLLM inference engine.\
    
    Attributes:
        - vllm_entrypoint (vllm.LLM): vLLM language model.
        - tokenizer (vllm.transformers_utils.tokenizer.AnyTokenizer): tokenizer.
        - logits (nnsight.WrapperModule): logits.
        - samples (nnsight.WrapperModule): sampled tokens.

    .. code-block:: python
        from nnsight.models.VLLM import VLLM
        from vllm import SamplingParams

        model = VLLM("gpt2")

        prompt = ["The Eiffel Tower is in the city of"]

        with model.trace(prompt, temperature=0.0, top_p=0.95, stop=['.']) as tracer:
            model.transformer.h[8].output[-1][:] = 0

            output = model.output.save()

        print(model.tokenizer.decode(output.value.argmax(dim=-1)[-1]))
    """

    def __init__(self, *args, **kwargs) -> None:

        self.vllm_entrypoint: LLM = None
        self.tokenizer: "AnyTokenizer" = None

        if not torch.distributed.is_initialized():

            init_distributed_environment(
                1,
                0,
                "tcp://127.0.0.1:47303",
                0,
                backend="gloo",
            )

            initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

        super().__init__(*args, **kwargs)

        self.logits: Envoy = WrapperModule()
        self.samples: Envoy = WrapperModule()
        self.generator: Envoy = WrapperModule()

    def _load_meta(self, repo_id: str, **kwargs) -> "Module":
        
        # no parallelism during initialization
        kwargs["tensor_parallel_size"] = 1
        kwargs["pipeline_parallel_size"] = 1

        # creating vLLM Engine args
        engine_args = EngineArgs(
            model=repo_id,
            **kwargs,
        )

        # creating the vllm engine configuration
        vllm_config = engine_args.create_engine_config()

        vllm_config.load_config.device = "meta"

        loader = DummyModelLoader(vllm_config.load_config)
        loader.load_weights = lambda *args, **kwargs: None
        model = loader.load_model(vllm_config, vllm_config.model_config)

     
        # self.tokenizer = init_tokenizer_from_configs(vllm_config.model_config,
        #                         vllm_config.scheduler_config,
        #                         vllm_config.lora_config)

        return model

    def _load(self, repo_id: str, **kwargs) -> "Module":

        meta_model = self._load_meta(repo_id, **kwargs)

        destroy_model_parallel()
        destroy_distributed_environment()
        
        llm = LLM(
            repo_id,
            worker_cls='nnsight.modeling.vllm.workers.GPUWorker.NNsightGPUWorker',
            enforce_eager=True, 
            **kwargs,
        )
                
        llm.llm_engine.__class__ = NNsightLLMEngine

        self.vllm_entrypoint = llm

        # load the tokenizer

        self.tokenizer = llm.llm_engine.tokenizer._tokenizer

        return meta_model

    def _prepare_input(
        self, *args, **kwargs
    ) -> Tuple[Tuple[Tuple[Any], Dict[str, Any]], int]:

        if "processed" in kwargs:
            return args, kwargs

        prompts = []
        params = []

        for arg in args:

            if not type(arg) is list:
                arg = [arg]

            for prompt in arg:

                param = NNsightSamplingParams(
                    interleaver=self._interleaver,
                    **kwargs,
                )

                if kwargs != {}:
                    param.is_default_param = False

                prompts.append(prompt)
                params.append(param)

        return (prompts, params), {"processed": True}
    
    def _batch(
        self,
        batched_inputs,
        prompts,
        params,
        **kwargs
    ) -> Tuple[Tuple[Tuple[Any], Dict[str, Any]], int]:
        
        batch_size = len(prompts)
        
        if batched_inputs is None:

            return ((prompts, params), kwargs), batch_size
        
        batched_args = batched_inputs[0]
        batched_kwargs = batched_inputs[1]

        batched_args[0].extend(prompts)
        batched_args[1].extend(params)

        batched_kwargs.update(kwargs)

        return batched_inputs, batch_size

    def __call__(
        self,
        prompts: List[str],
        params: List[NNsightSamplingParams],
        **kwargs,
    ) -> Any:
        
        default_param = NNsightSamplingParams.from_optional()

        kwargs.pop('hook', None)
        kwargs.pop('processed', None)

        max_max_tokens = 0
        for param in params:
            for attr, value in kwargs.items():
                if hasattr(NNsightSamplingParams, attr) and getattr(param, attr) == getattr(default_param, attr):
                    setattr(param, attr, value)

            max_max_tokens = param.max_tokens if param.max_tokens > max_max_tokens else max_max_tokens

        for mediator in self._interleaver.invokers:
            if mediator.batch_group != None:
                mediator.all_stop = params[self._interleaver.batcher.batch_groups[mediator.batch_group][0]].max_tokens
            else:
                mediator.all_stop = max_max_tokens
        
        output = self.vllm_entrypoint.generate(prompts, sampling_params=params)

        saves = output[0].saves

        for value in saves.values():

            Globals.saves.add(id(value))

        push_variables(self._interleaver.invokers[0].info.frame, output[0].saves)

        
    def interleave(self, fn: Callable, *args, **kwargs):
        
        try:
            fn(*args, **kwargs)
        finally:
            self._interleaver.check_cache_full()
            self._interleaver.cancel()

if TYPE_CHECKING:

    class VLLM(VLLM, LLM):
        pass
