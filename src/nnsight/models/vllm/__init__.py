from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Union

import torch

from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather)
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs

from ...envoy import Envoy
from ...intervention import InterventionProtocol
from ...patching import Patch, Patcher
from ...tracing import protocols
from ...tracing.Graph import Graph
from ...util import TypeHint, WrapperModule, hint
from ..mixins import RemoteableMixin
from .executors.GPUExecutor import NNsightGPUExecutor
from .executors.RayGPUExecutor import NNsightRayGPUExecutor
from .sampling import NNsightSamplingParams

if TYPE_CHECKING:
    from ...intervention import InterventionHandler

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


@hint
class VLLM(RemoteableMixin, TypeHint[Union[LLM, Envoy]]):
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

    __methods__ = {"generate": "_execute"}

    def __init__(self, *args, **kwargs) -> None:

        self.vllm_entrypoint: LLM = None
        self.tokenizer: AnyTokenizer = None

        super().__init__(*args, **kwargs)

        self.logits = WrapperModule()
        self.tokens = WrapperModule()

    def _load_meta(self, repo_id: str, **kwargs):

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
            model_config=engine_config_dict["model_config"],
            load_config=engine_config_dict["load_config"],
            lora_config=None,
            cache_config=engine_config_dict["cache_config"],
            scheduler_config=engine_config_dict["scheduler_config"],
        )

        self.tokenizer = init_tokenizer_from_configs(
            model_config=engine_config_dict["model_config"],
            scheduler_config=engine_config_dict["scheduler_config"],
            parallel_config=engine_config_dict["parallel_config"],
            enable_lora=bool(engine_config_dict["lora_config"]),
        ).tokenizer

        return model

    def _load(self, repo_id: str, **kwargs):

        destroy_model_parallel()
        destroy_distributed_environment()

        distributed_executor_backend = NNsightGPUExecutor
        if (
            "tensor_parallel_size" in kwargs.keys()
            and kwargs["tensor_parallel_size"] > 1
        ):
            distributed_executor_backend = NNsightRayGPUExecutor

        llm = LLM(
            repo_id,
            **kwargs,
            distributed_executor_backend=distributed_executor_backend,
        )

        self.vllm_entrypoint = llm

        return self._model

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

                prompts.append(prompt)
                params.append(param)

        return ((prompts, params), {"processed": True}), len(prompts)

    def _batch(
        self,
        batched_inputs: Tuple[Tuple[Any] | protocols.Dict[str, Any]] | None,
        prompts: List[str],
        params: List[NNsightSamplingParams],
        **kwargs,
    ) -> Tuple[Tuple[Any] | protocols.Dict[str, Any]]:

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
        fn: Callable,
        intervention_graph: Graph,
        prompts: List[str],
        params: List[NNsightSamplingParams],
        **kwargs,
    ) -> Any:

        if not self.dispatched:
            self.dispatch()

        for param in params:

            param.intervention_graph = intervention_graph

        def parallel_intervene(intervene_func: Callable) -> Callable:
            """ Create an intervene wrapper that handles tensor parallelism execution of vLLM models.

            Args:
                intervene_func (Callable): intervention function.
            
            Returns 
            """

            @wraps(intervene_func)
            def parallel_intervene_wrapper(
                activations: Any, 
                module_path: str, 
                module: torch.nn.Module, 
                key: str, 
                intervention_handler: "InterventionHandler"
            ) -> Any:
                """ InterventionProtocol.intervene wrapper handling the parallelized modules of vLLM.
                If some activations were parallelized, then they need to be gathered as a full tensor to intervene on them,
                and then split again before returning them.

                Args:
                    activations (Any): Either the inputs or outputs of a torch module.
                    module_path (str): Module path of the current relevant module relative to the root model.
                    module (torch.nn.Module): Module to be intervened on.
                    key (str): Key denoting either "input" or "output" of module.
                    intervention_handler (InterventionHandler): Handler object that stores the intervention graph and keeps track of module call count.

                Returns:
                    Any: The activations, potentially modified by the intervention graph.
                """
                # If the activations are parallelized, they must be gathered before intervening on them
                if isinstance(module, ColumnParallelLinear) and key == "output" and not module.gather_output:
                    full_tensor = tensor_model_parallel_all_gather(activations[0])
                    activations = (full_tensor, ) + activations[1:]
                if isinstance(module, RowParallelLinear) and key == "input" and module.input_is_parallel:
                    full_tensor = tensor_model_parallel_all_gather(activations[0][0])
                    activations = ((full_tensor,) + activations[0][1:], ) + activations[1:]

                activations = intervene_func(activations, module_path, module, key, intervention_handler)

                # If the activations were parallelized originally, they must be split again before returning them
                if isinstance(module, ColumnParallelLinear) and key == "output" and not module.gather_output:
                    tp_rank = get_tensor_model_parallel_rank()
                    splitted_input = split_tensor_along_last_dim(activations[0], num_partitions=get_tensor_model_parallel_world_size())
                    activations = (splitted_input[tp_rank].contiguous(),) + activations[1:]
                if isinstance(module, RowParallelLinear) and key == "input" and module.input_is_parallel:
                    tp_rank = get_tensor_model_parallel_rank()
                    splitted_input = split_tensor_along_last_dim(activations[0][0], num_partitions=get_tensor_model_parallel_world_size())
                    activations = ((splitted_input[tp_rank].contiguous(),) + activations[0][1:],) + activations[1:]

                return activations
            
            return parallel_intervene_wrapper

        # handle parallelmodules with custom intervene function for inference with tensor parallelism
        if get_tensor_model_parallel_world_size() > 1:
            intervene_patch = Patch(InterventionProtocol, parallel_intervene(InterventionProtocol.intervene), "intervene")
        else:
            intervene_patch = Patch(InterventionProtocol, InterventionProtocol.intervene, "intervene")

        with Patcher([intervene_patch]):

            fn(prompts, params, **kwargs)

        intervention_graph.alive = False

    def _execute(
        self,
        prompts: List[str],
        params: List[NNsightSamplingParams],
        **kwargs,
    ) -> Any:

        self.vllm_entrypoint.generate(prompts, sampling_params=params)
