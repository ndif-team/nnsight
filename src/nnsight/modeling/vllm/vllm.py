from ... import NNS_VLLM_VERSION

import vllm

# Check vLLM version compatibility
_installed_version = getattr(vllm, "__version__", "unknown")
if _installed_version != NNS_VLLM_VERSION:
    raise ImportError(
        f"nnsight requires vLLM version {NNS_VLLM_VERSION}, but found {_installed_version}. "
        f"Please install the correct version:\n\n"
        f"    pip install vllm=={NNS_VLLM_VERSION}\n"
    )

import torch

from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Union
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.inputs import TokensPrompt

from vllm import LLM
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.llm import LLM

from ...intervention.envoy import Envoy
from ...intervention.tracing.tracer import ScanningTracer
from ...intervention.tracing.util import push_variables
from ...util import WrapperModule
from ..mixins import RemoteableMixin
from .sampling import NNsightSamplingParams
from ...intervention.serialization import save as serialize
from ... import save
from .engines.engine import NNsightLLMEngine
from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT

if TYPE_CHECKING:
    from torch.nn import Module

    from vllm.transformers_utils.tokenizer import AnyTokenizer


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

            import socket

            def get_free_port():
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("127.0.0.1", 0))
                addr, port = s.getsockname()
                s.close()
                return port

            port = get_free_port()
            init_distributed_environment(
                1,
                0,
                f"tcp://127.0.0.1:{port}",
                0,
                backend="gloo",
            )

            initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )

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

        _ROPE_DICT.clear()

        self.tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return model

    def _load(self, repo_id: str, **kwargs) -> "Module":

        meta_model = self._load_meta(repo_id, **kwargs)

        destroy_model_parallel()
        destroy_distributed_environment()

        if kwargs.get("distributed_executor_backend") == "ray":
            from .executors.ray_workaround import NNsightRayExecutor
            kwargs["distributed_executor_backend"] = NNsightRayExecutor

        llm = LLM(
            repo_id,
            worker_cls="nnsight.modeling.vllm.workers.GPUWorker.NNsightGPUWorker",
            enforce_eager=True,
            **kwargs,
        )

        llm.llm_engine.__class__ = NNsightLLMEngine

        self.vllm_entrypoint = llm

        return meta_model

    def _prepare_input(
        self, *args, **kwargs
    ) -> Tuple[Tuple[Tuple[Any], Dict[str, Any]], int]:
        """Normalize user inputs into ``((prompts, params), kwargs, batch_size)``.

        Accepts strings, token ID lists, or HuggingFace tokenizer
        outputs and converts them into vLLM-compatible prompt objects
        with :class:`NNsightSamplingParams`.

        Returns:
            Tuple of ``((prompts, params), kwargs, batch_size)``.
        """

        prompts = []
        params = []

        for arg in args:
            if arg == []:
                raise ValueError("Empty list of prompts is not allowed")

            if type(arg) is dict:
                keys = set(arg.keys())
                if "input_ids" in keys and keys.issubset(
                    {"input_ids", "attention_mask"}
                ):
                    # is hf tokenizer result
                    batch_input_ids = arg["input_ids"]
                    batch_attention_mask = arg.get("attention_mask", None)
                    if isinstance(batch_input_ids, torch.Tensor):
                        batch_input_ids = batch_input_ids.tolist()
                    if isinstance(batch_attention_mask, torch.Tensor):
                        batch_attention_mask = batch_attention_mask.tolist()
                    if batch_input_ids == []:
                        raise ValueError("Empty list of token ids is not allowed")
                    if isinstance(batch_input_ids[0], int):
                        # list of token ids
                        batch_input_ids = [batch_input_ids]
                        batch_attention_mask = [batch_attention_mask]

                    for input_ids, attention_mask in zip(
                        batch_input_ids, batch_attention_mask
                    ):
                        prompt = TokensPrompt(
                            prompt_token_ids=[
                                t for t, m in zip(input_ids, attention_mask) if m != 0
                            ]
                        )
                        prompts.append(prompt)
                        params.append(NNsightSamplingParams(**kwargs))
                    continue

            if type(arg) is not list or isinstance(arg[0], int):
                # if arg is a list of ints (token ids), we also need to wrap it in a list
                arg = [arg]

            for i, prompt in enumerate(arg):

                param = NNsightSamplingParams(
                    **kwargs,
                )

                if kwargs != {}:
                    param.is_default_param = False

                if type(prompt) is list and isinstance(prompt[0], int):
                    prompt = TokensPrompt(prompt_token_ids=prompt)

                prompts.append(prompt)
                params.append(param)

        kwargs = kwargs if not args else {}

        return (prompts, params), kwargs, len(prompts)

    def _batch(
        self, batched_inputs, prompts, params, **kwargs
    ) -> Tuple[Tuple[Tuple[Any], Dict[str, Any]], int]:
        """Combine multiple invokes' prompts and sampling params into a single batch."""

        kwargs = {**kwargs, **batched_inputs[1]}

        if len(batched_inputs[0]) == 0:

            return (prompts, params), kwargs

        batched_args = batched_inputs[0]
        batched_kwargs = batched_inputs[1]

        batched_args[0].extend(prompts)
        batched_args[1].extend(params)

        return batched_args, batched_kwargs

    def __call__(
        self,
        prompts: List[str],
        params: List[NNsightSamplingParams],
        **kwargs,
    ) -> Any:
        """Execute vLLM generation with NNsight interventions.

        Attaches mediators to sampling params so the vLLM workers can
        deserialize and run intervention code, then collects saved
        variables from the outputs.
        """

        default_param = NNsightSamplingParams.from_optional()

        param_idx = 0

        # Find the sampling params associated with each mediator
        for mediator in self._interleaver.mediators:

            # If its the only invoker in the batch group, set the batch size to the total number of prompts
            if mediator.batch_group is None:
                batch_size = len(params)

            else:
                batch_size = mediator.batch_group[1]

            # For each prompt in the batch group associated with the mediator
            for i in range(batch_size):

                param = params[param_idx]

                # If its the first prompt in the batch group, it will transfer the mediator to the workers
                if i == 0:

                    mediator.intervention.__source__ = "".join(mediator.info.source)
                    param.extra_args = {"nnsight_mediator": serialize(mediator)}

                else:
                    # Mark subsequent prompts as batch members so the worker
                    # can associate them with the same mediator, even if the
                    # scheduler splits them across steps.
                    param.extra_args = {"nnsight_batch_member": True}

                param_idx += 1

                # Update the sampling params for each prompt with any kwargs passed to the root trace
                for attr, value in kwargs.items():
                    if hasattr(NNsightSamplingParams, attr) and getattr(
                        param, attr
                    ) == getattr(default_param, attr):
                        setattr(param, attr, value)

        # Do VLLM generation with NNsight
        outputs = self.vllm_entrypoint.generate(prompts, sampling_params=params)

        saves = {}

        # Some of the output objects will have a saves attribute, which contains the saved variables
        for output in outputs:
            if hasattr(output, "saves"):
                saves.update(output.saves)

        # Save the variables in our local environment
        for value in saves.values():

            save(value)

        # Push the variables to the interleaver frame
        push_variables(self._interleaver.mediators[0].info.frame, saves)

    def interleave(self, fn: Callable, *args, **kwargs):
        """Execute the traced function with vLLM, dispatching the engine if needed."""
        if not self.dispatched and not isinstance(
            self._interleaver.tracer, ScanningTracer
        ):
            self.dispatch()

        try:
            fn(*args, **kwargs)
        finally:
            self._interleaver.check_cache_full()
            self._interleaver.cancel()

    def _remoteable_persistent_objects(self) -> dict:
        persistent_objects = super()._remoteable_persistent_objects()
        persistent_objects["Tokenizer"] = self.tokenizer
        return persistent_objects

    def __getstate__(self):

        state = super().__getstate__()
        state["vllm_entrypoint"] = None
        if self.tokenizer is not None:
            self.tokenizer._persistent_id = "Tokenizer"
        state["tokenizer"] = self.tokenizer
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.vllm_entrypoint = state["vllm_entrypoint"]
        self.tokenizer = state["tokenizer"]
