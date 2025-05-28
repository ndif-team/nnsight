from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Union, Optional

from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs

from ...util import WrapperModule
from ..mixins import RemoteableMixin
from .workers.GPUWorker import NNsightGPUWorker
from .sampling import NNsightSamplingParams
from dataclasses import fields
from ...intervention.interleaver import Interleaver
from ...intervention import Envoy

if TYPE_CHECKING:
    from torch.nn import Module
    from vllm.transformers_utils.tokenizer import AnyTokenizer
    from vllm.config import ModelConfig, SchedulerConfig, ParallelConfig, LoRAConfig

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

    __methods__ = {"generate": "_execute"}

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
        vllm_config_dict = {field.name: getattr(vllm_config, field.name) for field in fields(type(vllm_config))}

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

        model = _initialize_model(vllm_config)

        # load the tokenzier
        self.tokenizer = self._load_tokenizer(
            model_config=vllm_config_dict["model_config"],
            scheduler_config=vllm_config_dict["scheduler_config"],
            parallel_config=vllm_config_dict["parallel_config"],
            enable_lora=bool(vllm_config_dict["lora_config"]),
        )

        return model
    
    def _load_tokenizer(
        self, 
        model_config: "ModelConfig", 
        scheduler_config: "SchedulerConfig", 
        parallel_config: "ParallelConfig", 
        lora_config: "LoRAConfig") -> "AnyTokenizer":
        
        return init_tokenizer_from_configs(
            model_config=model_config,
            scheduler_config=scheduler_config,
            parallel_config=parallel_config,
            lora_config=lora_config,
        ).tokenizer

    def _load(self, repo_id: str, **kwargs) -> "Module":

        destroy_model_parallel()
        destroy_distributed_environment()
        
        llm = LLM(
            repo_id,
            worker_cls=NNsightGPUWorker,
            **kwargs,
        )

        self.vllm_entrypoint = llm

        # load the tokenizer
        self.tokenizer = self._load_tokenizer(
            model_config=llm.llm_engine.model_config,
            scheduler_config=llm.llm_engine.scheduler_config,
            parallel_config=llm.llm_engine.parallel_config,
            lora_config=llm.llm_engine.lora_config,
        )

        if kwargs.get("tensor_parallel_size", 1) > 1:
            return llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
        else:
            return llm.llm_engine.model_executor.driver_worker.model_runner.model

    def _prepare_input(
        self, *args, **kwargs
    ) -> Tuple[Tuple[Tuple[Any], Dict[str, Any]], int]:

        if "processed" in kwargs:
            return (args, kwargs), len(args[0])

        prompts = []
        params = []

        for arg in args:

            if not type(arg) is list:
                arg = [arg]

            for prompt in arg:

                param = NNsightSamplingParams(
                    **kwargs,
                )

                if kwargs != {}:
                    param.is_default_param = False

                prompts.append(prompt)
                params.append(param)

        return ((prompts, params), {"processed": True}), len(prompts)

    def _batch(
        self,
        batched_inputs: Union[Tuple[Tuple[Any], Dict[str, Any]], None],
        prompts: List[str],
        params: List[NNsightSamplingParams],
        **kwargs,
    ) -> Tuple[Union[Tuple[Any], Dict[str, Any]]]:

        if batched_inputs is None:
            batched_inputs = ([], []), {"invoker_group": 0}

        (bprompts, bparams), kwargs = batched_inputs

        invoker_group = kwargs["invoker_group"]

        for prompt in prompts:
            bprompts.append(prompt)

        for param in params:

            param.invoker_group = invoker_group

            bparams.append(param)

        kwargs["invoker_group"] += 1

        return (bprompts, bparams), kwargs

    def interleave(
        self,
        interleaver: Interleaver,
        *args,
        fn: Optional[Union[Callable, str]] = None,
        **kwargs,
    ) -> Any:

        """ if not self.dispatched:
            self.dispatch()

        for param in params:

            param.intervention_graph = intervention_graph

        fn(prompts, params, **kwargs)

        intervention_graph.alive = False """
        
        
        if not self.dispatched:
            self.dispatch()

        for param in args[1]:

            param.intervention_graph = interleaver.graph
            param.nns_batch_groups = interleaver.batch_groups

        if fn is None:
            fn = self._execute
        elif isinstance(fn, str):
            fn = getattr(self, fn)            
    
        return fn(*args, interleaver=interleaver, **kwargs)

    def __call__(
        self,
        prompts: List[str],
        params: List[NNsightSamplingParams],
        interleaver: Interleaver,
        **kwargs,
    ) -> Any:

        kwargs.pop('invoker_group')

        for param in params:
            if param.is_default_param:
                for attr, value in kwargs.items():
                    if hasattr(NNsightSamplingParams, attr):
                        setattr(param, attr, value)

        out = self.vllm_entrypoint.generate(prompts, sampling_params=params)
        
        with interleaver:
            self.generator(out)

if TYPE_CHECKING:
    
    class VLLM(VLLM,LLM):
        pass