import atexit
import uuid
import vllm

import torch

from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
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
from vllm.config import VllmConfig, set_current_vllm_config
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
        self._compat: bool = kwargs.pop("compat", True)

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

            with set_current_vllm_config(VllmConfig()):
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

        # Disable prefix caching by default.  Prefix caching reuses KV values
        # from previous requests, which can skip the forward pass for cached
        # tokens — hooks won't fire and interventions on those tokens are
        # silently skipped.  Users can override with enable_prefix_caching=True.
        kwargs.setdefault("enable_prefix_caching", False)

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
                if "input_ids" in keys and keys.issubset({"input_ids", "attention_mask"}):
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
        attention_mask = batch_attention_mask[0] if batch_attention_mask is not None else None

        # Filter out masked tokens if attention mask is provided
        if attention_mask is not None:
            return TokensPrompt(
                prompt_token_ids=[t for t, m in zip(input_ids, attention_mask) if m != 0]
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
        mediators: Optional[List[Any]] = None,
        **kwargs,
    ) -> Tuple[List[str], List[NNsightSamplingParams], List[Any]]:
        """Serialize mediators and attach them to sampling params.

        Collects all input mediators, serializes each into the corresponding
        ``NNsightSamplingParams.extra_args``, and propagates any root-trace
        kwargs to params that still carry defaults.

        Args:
            mediators: Explicit mediator list. Server paths pass
                ``tracer.mediators`` because they call ``_run_user_fn``
                without ``_init_shared_interleaver`` to avoid racing with
                ``Mediator.start()`` on the vLLM worker thread; those paths
                cannot rely on ``self._interleaver.mediators`` being current.
                When ``None`` (local sync/async paths), falls back to
                ``self._interleaver.mediators`` as populated by
                ``Interleaver.initialize``.

        Returns:
            ``(prompts, params, lora_requests)`` with mediator data attached.
        """

        default_param = NNsightSamplingParams.from_optional()

        source_mediators = (
            mediators if mediators is not None else self._interleaver.mediators
        )

        # Collect all input mediators (those with batch_group, i.e. not empty invokes)
        input_mediators = []
        for mediator in source_mediators:
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

        prompts, params, lora_requests = self._serialize_mediators(prompts, params, lora_requests, **kwargs)

        outputs = self.vllm_entrypoint.generate(prompts, sampling_params=params, lora_request=lora_requests)

        saves = {}
        all_exceptions: dict = {}

        for output in outputs:
            if not hasattr(output, "saves"):
                continue
            # Each output's saves is this request's own sub-dict now
            # (per-request namespacing in the engine/worker).  Its
            # ``__nnsight_exceptions__`` entry is a ``{base_id: exc}``
            # map holding only THIS request's failure; combine across
            # invokes into one trace-level map.
            for k, v in output.saves.items():
                if k == "__nnsight_exceptions__":
                    all_exceptions.update(v)
                else:
                    saves[k] = v

        deferred_exceptions = all_exceptions or None

        # Save the variables in our local environment
        for value in saves.values():

            save(value)

        # Push the variables to the interleaver frame
        push_variables(self._interleaver.mediators[0].info.frame, saves)

        # Raise deferred exceptions at the trace boundary (after saves are
        # pushed so any values saved before the error are still accessible).
        # EarlyStopException is intentional control flow — filter it out.
        if deferred_exceptions is not None:
            real_errors = {
                req_id: exc_info for req_id, exc_info in deferred_exceptions.items()
                if exc_info.get("type") != "EarlyStopException"
            }
            if real_errors:
                # Raise the first real error with its original type.
                first_id, first_exc = next(iter(real_errors.items()))
                exc_type_name = first_exc.get("type", "Exception")
                exc_message = first_exc.get("message", "")
                builtins_ref = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
                exc_type = builtins_ref.get(exc_type_name, RuntimeError) if isinstance(builtins_ref, dict) else getattr(builtins_ref, exc_type_name, RuntimeError)
                if not isinstance(exc_type, type) or not issubclass(exc_type, BaseException):
                    exc_type = RuntimeError
                raise exc_type(f"[vLLM worker] {exc_message}")

    def trace(self, *inputs, **kwargs):
        serve = kwargs.pop("serve", None)
        if serve is not None and kwargs.get("backend") is None:
            from ...intervention.backends.local_serve import LocalServeBackend
            from .serve_tracer import ServeInterleavingTracer
            blocking = kwargs.pop("blocking", True)
            api_key = kwargs.pop("api_key", None)
            kwargs["backend"] = LocalServeBackend(self, host=serve, blocking=blocking, api_key=api_key)
            kwargs.setdefault("tracer_cls", ServeInterleavingTracer)
        else:
            if "api_key" in kwargs:
                raise ValueError("api_key= requires serve= to specify the server URL")
            if self._async_engine and kwargs.get('backend') is None and not kwargs.get('remote'):
                from .async_backend import AsyncVLLMBackend
                kwargs['backend'] = AsyncVLLMBackend(self)
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
