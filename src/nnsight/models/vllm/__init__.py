import weakref
from typing import Any, Callable, Dict, List, Tuple, Union

from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs

from ...envoy import Envoy
from ...tracing import protocols
from ...tracing.Graph import Graph
from ...util import TypeHint, WrapperModule, hint
from ..mixins import RemoteableMixin
from .executors.GPUExecutor import NNsightGPUExecutor
from .executors.RayGPUExecutor import NNsightRayGPUExecutor
from .sampling import NNsightSamplingParams

try:
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
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

        destroy_model_parallel()
        destroy_distributed_environment()

        return model

    def _load(self, repo_id: str, **kwargs):

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

        fn(prompts, params, **kwargs)

        intervention_graph.alive = False

    def _execute(
        self,
        prompts: List[str],
        params: List[NNsightSamplingParams],
        **kwargs,
    ) -> Any:

        self.vllm_entrypoint.generate(prompts, sampling_params=params)
