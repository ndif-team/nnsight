import atexit
import uuid

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
from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.entrypoints.llm import LLM

from ...intervention.envoy import eproperty
from ...intervention.tracing.globals import Globals
from ...intervention.tracing.tracer import ScanningTracer
from ...intervention.tracing.util import push_variables
from ..mixins import RemoteableMixin
from .sampling import NNsightSamplingParams
from ...intervention.serialization import save as serialize
from ... import save
from ... import CONFIG
from .engines.engine import NNsightLLMEngine
from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT


CONFIG.APP.CROSS_INVOKER = False
if TYPE_CHECKING:
    from torch.nn import Module

    from vllm.transformers_utils.tokenizer import AnyTokenizer


class VLLM(RemoteableMixin):
    """NNsight wrapper to conduct interventions on a vLLM inference engine.\

    Attributes:
        - vllm_entrypoint (vllm.LLM): vLLM language model.
        - tokenizer (vllm.transformers_utils.tokenizer.AnyTokenizer): tokenizer.
        - logits (eproperty): logit tensor.
        - samples (eproperty): sampled token ids.

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
            raise ValueError(f"Invalid mode {mode!r}. Must be 'sync' or 'async'.")
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

        atexit.register(VLLM._cleanup_distributed)

        super().__init__(*args, **kwargs)

    @eproperty(description="Logits", iterate=True)
    def logits(self):
        """The logit tensor produced by the model before sampling.

        Access during a trace to observe or modify logits::

            with model.trace("Hello", temperature=0.0, top_p=1):
                logits = model.logits.save()
        """

    @eproperty(description="Sampled token ids", iterate=True)
    def samples(self):
        """The sampled token IDs produced by the sampler after logits.

        Access during a trace to observe or modify sampled tokens::

            with model.trace("Hello", temperature=0.8, top_p=0.95, max_tokens=3) as tracer:
                tokens = list().save()
                for step in tracer.iter[:]:
                    tokens.append(model.samples.item())
        """

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

        # vLLM requires config context for model parallel init and model loading
        with set_current_vllm_config(vllm_config):

            initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )

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
        self, *args, lora_request=None, **kwargs
    ) -> Tuple[Tuple[Tuple[Any], Dict[str, Any]], int]:
        """Normalize a single user input into ``((prompts, params, lora_requests), kwargs, batch_size)``.

        Accepts one of:
        - A string prompt (e.g. ``"Hello world"``)
        - A list of token IDs (e.g. ``[101, 2023, ...]``)
        - A HuggingFace tokenizer output dict with ``input_ids`` and
          optional ``attention_mask``

        Each invoke must contain exactly one prompt. To process multiple
        prompts, use separate ``tracer.invoke()`` calls.

        Returns:
            Tuple of ``((prompts, params, lora_requests), kwargs, batch_size)``.
        """

        prompts = []
        params = []
        lora_requests = []

        for arg in args:
            if arg == []:
                raise ValueError("Empty list of prompts is not allowed")

            # --- HuggingFace tokenizer dict (e.g. tokenizer("hello")) ---
            if type(arg) is dict:
                keys = set(arg.keys())
                if "input_ids" in keys and keys.issubset(
                    {"input_ids", "attention_mask"}
                ):
                    prompt = self._parse_hf_tokenizer_dict(arg)
                    prompts.append(prompt)
                    params.append(NNsightSamplingParams(**kwargs))
                    lora_requests.append(lora_request)
                    continue

            # --- Token ID list (e.g. [101, 2023, ...]) ---
            if type(arg) is list and isinstance(arg[0], int):
                prompt = TokensPrompt(prompt_token_ids=arg)

            # --- String prompt (e.g. "Hello world") ---
            elif type(arg) is not list:
                prompt = arg

            # --- Multi-prompt list (not supported) ---
            else:
                raise ValueError(
                    "Multiple prompts per invoke are not supported. "
                    "Use separate tracer.invoke() calls for each prompt."
                )

            param = NNsightSamplingParams(**kwargs)
            if kwargs:
                param.is_default_param = False

            prompts.append(prompt)
            params.append(param)
            lora_requests.append(lora_request)

        # If args were provided, kwargs were already consumed as sampling params above.
        kwargs = kwargs if not args else {}

        return (prompts, params, lora_requests), kwargs, len(prompts)

    def _parse_hf_tokenizer_dict(self, arg: dict) -> TokensPrompt:
        """Convert a HuggingFace tokenizer output dict to a vLLM ``TokensPrompt``.

        Handles tensor-to-list conversion, single vs batched sequences,
        and attention mask filtering.
        """
        batch_input_ids = arg["input_ids"]
        batch_attention_mask = arg.get("attention_mask", None)

        # Convert tensors to lists
        if isinstance(batch_input_ids, torch.Tensor):
            batch_input_ids = batch_input_ids.tolist()
        if isinstance(batch_attention_mask, torch.Tensor):
            batch_attention_mask = batch_attention_mask.tolist()

        if batch_input_ids == []:
            raise ValueError("Empty list of token ids is not allowed")

        # Normalize single sequence to batch format
        if isinstance(batch_input_ids[0], int):
            batch_input_ids = [batch_input_ids]
            if batch_attention_mask is not None:
                batch_attention_mask = [batch_attention_mask]

        if len(batch_input_ids) > 1:
            raise ValueError(
                "Multiple prompts per invoke are not supported. "
                "Use separate tracer.invoke() calls for each prompt."
            )

        input_ids = batch_input_ids[0]
        attention_mask = (
            batch_attention_mask[0] if batch_attention_mask is not None else None
        )

        # Filter out masked tokens if attention mask is provided
        if attention_mask is not None:
            return TokensPrompt(
                prompt_token_ids=[
                    t for t, m in zip(input_ids, attention_mask) if m != 0
                ]
            )
        return TokensPrompt(prompt_token_ids=input_ids)

    def _batch(
        self, batched_inputs, prompts, params, lora_requests, **kwargs
    ) -> Tuple[Tuple[Tuple[Any], Dict[str, Any], List[Any]], int]:
        """Combine multiple invokes' prompts and sampling params into a single batch."""

        kwargs = {**kwargs, **batched_inputs[1]}

        if len(batched_inputs[0]) == 0:

            return (prompts, params, lora_requests), kwargs

        batched_args = batched_inputs[0]
        batched_kwargs = batched_inputs[1]

        batched_args[0].extend(prompts)
        batched_args[1].extend(params)
        batched_args[2].extend(lora_requests)
        return batched_args, batched_kwargs

    def _serialize_mediators(
        self,
        prompts: List[str],
        params: List[NNsightSamplingParams],
        lora_requests: List[Any],
        **kwargs,
    ) -> Tuple[List[str], List[NNsightSamplingParams], List[Any]]:
        """Serialize mediators and attach them to sampling params.

        Collects all input mediators from the interleaver, serializes
        each one into the corresponding ``NNsightSamplingParams.extra_args``,
        and propagates any root-trace kwargs to params that still carry
        defaults.

        Returns:
            ``(prompts, params, lora_requests)`` with mediator data attached.
        """

        default_param = NNsightSamplingParams.from_optional()

        # Collect all input mediators (those with batch_group, i.e. not empty invokes)
        input_mediators = []
        for mediator in self.interleaver.mediators:
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
                name for name, val in frame_globals.items() if id(val) in Globals.saves
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

        return prompts, params, lora_requests

    def __call__(
        self,
        prompts: List[str],
        params: List[NNsightSamplingParams],
        lora_requests: List[Any],
        **kwargs,
    ) -> Any:
        """Execute synchronous vLLM generation with NNsight interventions.

        Each mediator maps to exactly one prompt/param (1:1).
        """

        prompts, params, lora_requests = self._serialize_mediators(
            prompts, params, lora_requests, **kwargs
        )

        # Do VLLM generation with NNsight
        outputs = self.vllm_entrypoint.generate(
            prompts, sampling_params=params, lora_request=lora_requests
        )

        saves = {}

        # Some of the output objects will have a saves attribute, which contains the saved variables
        for output in outputs:
            if hasattr(output, "saves"):
                saves.update(output.saves)

        # Save the variables in our local environment
        for value in saves.values():

            save(value)

        # Push the variables to the interleaver frame
        push_variables(self.interleaver.mediators[0].info.frame, saves)

    def trace(self, *inputs, **kwargs):
        serve = kwargs.pop("serve", None)
        if serve is not None and kwargs.get("backend") is None:
            from ...intervention.backends.local_serve import LocalServeBackend
            from .serve_tracer import ServeInterleavingTracer

            blocking = kwargs.pop("blocking", True)
            api_key = kwargs.pop("api_key", None)
            kwargs["backend"] = LocalServeBackend(
                self, host=serve, blocking=blocking, api_key=api_key
            )
            kwargs.setdefault("tracer_cls", ServeInterleavingTracer)
        else:
            if "api_key" in kwargs:
                raise ValueError(
                    "api_key= requires serve= to specify the server URL"
                )
            if (
                self._async_engine
                and kwargs.get("backend") is None
                and not kwargs.get("remote")
            ):
                from .async_backend import AsyncVLLMBackend

                kwargs["backend"] = AsyncVLLMBackend(self)
        return super().trace(*inputs, **kwargs)

    def interleave(self, fn: Callable, *args, **kwargs):
        """Execute the traced function with vLLM, dispatching the engine if needed."""
        if not self.dispatched and not isinstance(
            self.interleaver.tracer, ScanningTracer
        ):
            self.dispatch()

        try:
            fn(*args, **kwargs)
        finally:
            self.interleaver.check_cache_full()
            self.interleaver.cancel()

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
