from dataclasses import fields
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Union)

import torch
from vllm.config import CompilationConfig
from vllm import LLM
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs

from ...intervention.envoy import Envoy
from ...intervention.interleaver import Interleaver
from ...util import WrapperModule
from ..mixins import RemoteableMixin
from .sampling import NNsightSamplingParams
from .workers.GPUWorker import NNsightGPUWorker

if TYPE_CHECKING:
    from torch.nn import Module
    from vllm.transformers_utils.tokenizer import AnyTokenizer
    from vllm.config import ModelConfig, SchedulerConfig, ParallelConfig, LoRAConfig

from vllm import envs
from vllm.distributed import (destroy_distributed_environment,
                              destroy_model_parallel, get_world_group,
                              init_distributed_environment,
                              init_model_parallel_group,
                              initialize_model_parallel, parallel_state)
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.llm import LLM
from vllm.model_executor.model_loader.utils import (get_model_architecture,
                                                    initialize_model)

envs.VLLM_ENABLE_V1_MULTIPROCESSING = False
# from vllm.model_executor.model_loader.loader import _initialize_model
# except Exception as e:
#     raise type(e)(
#         "Install vllm in your environment to use it with NNsight. "
#         + "https://docs.vllm.ai/en/latest/getting_started/installation.html"
#     ) from e


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
        vllm_config_dict = {
            field.name: getattr(vllm_config, field.name)
            for field in fields(type(vllm_config))
        }

        # starting the distributed environment
        init_distributed_environment(
            1,
            0,
            "tcp://127.0.0.1:47303",
            0,
            backend="gloo",
        )

        # message queue broadcaster is only used in tensor model parallel group
        parallel_state._TP = parallel_state._PP = parallel_state._DP = (
            init_model_parallel_group(
                [[0]],
                get_world_group().local_rank,
                "gloo",
                use_message_queue_broadcaster=True,
                group_name="tp",
            )
        )
        # initialize the model
        vllm_config.compilation_config.level = 1
        model_class, _ = get_model_architecture(vllm_config.model_config)
        model =  model_class(vllm_config=vllm_config, prefix="")
        #model = initialize_model(vllm_config)
      
        self.tokenizer = init_tokenizer_from_configs(vllm_config.model_config,
                                vllm_config.scheduler_config,
                                vllm_config.lora_config)

        return model

    def _load(self, repo_id: str, **kwargs) -> "Module":
        comp_cfg = CompilationConfig(
            level=0,             # disables all optimization layers
            use_inductor=False,  # ensures TorchInductor isn't used
            compile_sizes=[],    # no piecewise compilation
            # you can also tweak cudagraph flags if needed:
            use_cudagraph=False,
        )
        llm = LLM(
            repo_id,
            worker_cls='nnsight.modeling.vllm.workers.GPUWorker.NNsightGPUWorker',
            compilation_config=comp_cfg,
            enforce_eager=True, 
            **kwargs,
        )
                
        self.vllm_entrypoint = llm

        # load the tokenizer
        self.tokenizer = llm.llm_engine.tokenizer.tokenizer
        
        
        return llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model

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
        batched_inputs: Union[Tuple[Tuple[Any], Dict[str, Any]], None],
        prompts: List[str],
        params: List[NNsightSamplingParams],
        **kwargs,
    ) -> Tuple[Union[Tuple[Any], Dict[str, Any]]]:

        if batched_inputs is None:
            batched_inputs = (([], []), {"invoker_group": 0}), len(prompts)

        (bprompts, bparams), kwargs = batched_inputs

        invoker_group = kwargs["invoker_group"]

        for prompt in prompts:
            bprompts.append(prompt)

        for param in params:

            param.invoker_group = invoker_group

            bparams.append(param)

        kwargs["invoker_group"] += 1

        return ((bprompts, bparams), kwargs), len(prompts)

    def __call__(
        self,
        prompts: List[str],
        params: List[NNsightSamplingParams],
        **kwargs,
    ) -> Any:
        
        kwargs.pop('hook', None)
        kwargs.pop('processed', None)

        output = self.vllm_entrypoint.generate(prompts, sampling_params=params, **kwargs)
        
        with self._interleaver:
            self.generator(output, hook=True)
        
    def interleave(self, fn: Callable, *args, **kwargs):
        
        try:
            fn(*args, **kwargs)
        finally:
            self._interleaver.check_cache_full()
            self._interleaver.cancel()

if TYPE_CHECKING:

    class VLLM(VLLM, LLM):
        pass
