from ... import NNS_VLLM_VERSION

import atexit
import uuid
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
from ...intervention.tracing.globals import Globals
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

        mode = kwargs.pop("mode", "sync")
        if mode not in ("sync", "async"):
            raise ValueError(
                f"Invalid mode {mode!r}. Must be 'sync' or 'async'."
            )
        self._async_engine: bool = mode == "async"

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

        atexit.register(VLLM._cleanup_distributed)

        super().__init__(*args, **kwargs)

        self.logits: Envoy = WrapperModule()
        self.samples: Envoy = WrapperModule()
        self.generator: Envoy = WrapperModule()

    @staticmethod
    def _cleanup_distributed():
        try:
            destroy_model_parallel()
        except Exception:
            pass
        try:
            destroy_distributed_environment()
        except Exception:
            pass

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

        # Swap Ray executor for both sync and async paths.
        _uses_ray = kwargs.get("distributed_executor_backend") == "ray"
        if _uses_ray:
            from .executors.ray_workaround import NNsightRayExecutor
            kwargs["distributed_executor_backend"] = NNsightRayExecutor

        if self._async_engine:
            # AsyncLLM spawns EngineCore in a subprocess.  When using Ray,
            # the subprocess needs a running Ray cluster to connect to via
            # ray.init(address="auto").  Pre-initialize Ray here so the
            # cluster is available before the subprocess starts.
            if _uses_ray:
                import ray
                if not ray.is_initialized():
                    ray.init()
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM

            engine_args = AsyncEngineArgs(
                model=repo_id,
                worker_cls="nnsight.modeling.vllm.workers.GPUWorker.NNsightGPUWorker",
                enforce_eager=True,
                **kwargs,
            )
            async_llm = AsyncLLM.from_engine_args(engine_args)
            self.vllm_entrypoint = async_llm
        else:
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
        """Normalize a single user input into ``((prompts, params), kwargs, batch_size)``.

        Accepts a single string, a single token ID list, or a single-sequence
        HuggingFace tokenizer output and converts it into a vLLM-compatible
        prompt with :class:`NNsightSamplingParams`.

        Each invoke must contain exactly one prompt.  To process multiple
        prompts, use separate ``tracer.invoke()`` calls.

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
                        # single sequence of token ids
                        batch_input_ids = [batch_input_ids]
                        batch_attention_mask = [batch_attention_mask]

                    if len(batch_input_ids) > 1:
                        raise ValueError(
                            "Multiple prompts per invoke are not supported. "
                            "Use separate tracer.invoke() calls for each prompt."
                        )

                    input_ids = batch_input_ids[0]
                    attention_mask = batch_attention_mask[0]
                    prompt = TokensPrompt(
                        prompt_token_ids=[
                            t for t, m in zip(input_ids, attention_mask) if m != 0
                        ]
                    )
                    prompts.append(prompt)
                    params.append(NNsightSamplingParams(**kwargs))
                    continue

            if type(arg) is list and isinstance(arg[0], int):
                # single list of token ids
                arg = [arg]
            elif type(arg) is not list:
                # single string
                arg = [arg]
            else:
                # arg is a list but not of ints — reject multi-prompt
                raise ValueError(
                    "Multiple prompts per invoke are not supported. "
                    "Use separate tracer.invoke() calls for each prompt."
                )

            prompt = arg[0]

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

    def _prepare_generation(
        self,
        prompts: List[str],
        params: List[NNsightSamplingParams],
        **kwargs,
    ) -> Tuple[List[str], List[NNsightSamplingParams]]:
        """Serialize mediators and attach them to sampling params.

        Collects all input mediators from the interleaver, serializes
        each one into the corresponding ``NNsightSamplingParams.extra_args``,
        and propagates any root-trace kwargs to params that still carry
        defaults.

        Returns:
            ``(prompts, params)`` with mediator data attached.
        """

        default_param = NNsightSamplingParams.from_optional()

        # Collect all input mediators (those with batch_group, i.e. not empty invokes)
        input_mediators = []
        for mediator in self._interleaver.mediators:
            if mediator.batch_group is not None:
                mediator.intervention.__source__ = "".join(mediator.info.source)
                input_mediators.append(mediator)

        # Compute saved_names: parent-frame variable names whose values are in Globals.saves.
        # These are variables defined in the parent trace scope (e.g., shared lists)
        # that need to be collected after all invokes complete on the worker.
        saved_names = []
        if input_mediators:
            frame_globals = input_mediators[0].intervention.__globals__
            saved_names = [
                name for name, val in frame_globals.items()
                if id(val) in Globals.saves
            ]

        trace_id = str(uuid.uuid4())

        param_idx = 0
        for idx, mediator in enumerate(input_mediators):
            param = params[param_idx]
            param.extra_args = {
                "nnsight_mediator": serialize(mediator),
                "nnsight_trace_id": trace_id,
                "nnsight_trace_idx": idx,
                "nnsight_saved_names": saved_names,
                "nnsight_expected_count": len(input_mediators),
            }
            param_idx += 1

            # Update the sampling params with any kwargs passed to the root trace
            for attr, value in kwargs.items():
                if hasattr(NNsightSamplingParams, attr) and getattr(
                    param, attr
                ) == getattr(default_param, attr):
                    setattr(param, attr, value)

        return prompts, params

    def __call__(
        self,
        prompts: List[str],
        params: List[NNsightSamplingParams],
        **kwargs,
    ) -> Any:
        """Execute synchronous vLLM generation with NNsight interventions.

        Each mediator maps to exactly one prompt/param (1:1).
        """

        prompts, params = self._prepare_generation(prompts, params, **kwargs)

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

    def trace(self, *inputs, **kwargs):
        if self._async_engine and kwargs.get('backend') is None and not kwargs.get('remote'):
            from .async_backend import AsyncVLLMBackend
            from .async_tracer import AsyncInterleavingTracer

            kwargs['backend'] = AsyncVLLMBackend(self)
            # Bypass RemoteableMixin to pass custom tracer_cls.
            return Envoy.trace(self, *inputs, tracer_cls=AsyncInterleavingTracer, **kwargs)
        return super().trace(*inputs, **kwargs)

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
