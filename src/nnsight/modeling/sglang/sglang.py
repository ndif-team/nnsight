import atexit
import base64
import pickle
import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...intervention.envoy import Envoy
from ...intervention.tracing.globals import Globals
from ...intervention.tracing.tracer import ScanningTracer
from ...intervention.tracing.util import push_variables
from ...util import WrapperModule
from ..mixins import RemoteableMixin
from ...intervention.serialization import save as serialize
from ... import save

if TYPE_CHECKING:
    from torch.nn import Module


class SGLang(RemoteableMixin):
    """NNsight wrapper to conduct interventions on an SGLang inference engine.

    Attributes:
        engine: SGLang Engine instance.
        tokenizer: HuggingFace tokenizer.
        logits (WrapperModule): Hook for logits output.
        samples (WrapperModule): Hook for sampled tokens.

    Example::

        from nnsight.modeling.sglang import SGLang

        model = SGLang("gpt2", dispatch=True)

        with model.trace("The Eiffel Tower is in the city of", temperature=0, max_new_tokens=1):
            logits = model.logits.output.save()

        print(model.tokenizer.decode(logits.argmax(dim=-1)))
    """

    def __init__(self, *args, dispatch: bool = False, **kwargs) -> None:
        self.engine = None
        self.tokenizer = None

        if isinstance(args[0], torch.nn.Module) or dispatch:
            # Direct module wrapping or dispatch mode
            super().__init__(*args, dispatch=dispatch, **kwargs)
        else:
            # Meta model loading: skip MetaMixin's init_empty_weights context
            # because SGLang's model imports fail under meta tensor mode.
            # Instead, we load a CPU model with random weights.
            from ..base import NNsight
            rename = kwargs.pop("rename", None)
            model = self._load_meta(*args, **kwargs)
            NNsight.__init__(self, model, rename=rename)
            self.dispatched = False
            self.args = args
            self.kwargs = kwargs

        self.logits: Envoy = WrapperModule()
        self.samples: Envoy = WrapperModule()
        self.generator: Envoy = WrapperModule()

    def _load_meta(self, repo_id: str, **kwargs) -> "Module":
        """Load a meta-device model using SGLang's model architecture.

        Uses SGLang's DummyModelLoader to create a model with random weights
        that has the exact same module tree as the real model in the subprocess.
        """
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_loader.loader import DummyModelLoader
        from sglang.srt.configs.load_config import LoadConfig, LoadFormat
        from sglang.srt.configs.device_config import DeviceConfig
        from sglang.srt.server_args import ServerArgs

        trust_remote_code = kwargs.get("trust_remote_code", True)

        # We need to build SGLang's model architecture on meta device.
        # This requires: distributed env, global server args, and ModelConfig.
        # We initialize them, create the model, and clean up.
        import socket
        import torch.distributed as dist
        from sglang.srt.server_args import set_global_server_args_for_scheduler

        server_args = ServerArgs(model_path=repo_id, trust_remote_code=trust_remote_code)
        set_global_server_args_for_scheduler(server_args)
        model_config = ModelConfig.from_server_args(server_args)

        # Patch hf_config with attributes SGLang models expect
        hf_config = model_config.hf_config
        if not hasattr(hf_config, "num_key_value_heads"):
            hf_config.num_key_value_heads = getattr(
                hf_config, "num_attention_heads", None
            )

        # Initialize a temporary distributed env for meta model construction.
        # SGLang models require TP groups to be initialized even for
        # meta-device construction (they query world_size/rank).
        from sglang.srt.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        need_dist_init = not dist.is_initialized()
        if need_dist_init:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            s.close()
            init_distributed_environment(
                world_size=1, rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{port}",
                local_rank=0, backend="gloo",
            )
        try:
            initialize_model_parallel(1, 1)
        except AssertionError:
            pass  # Already initialized

        # Create the model on meta device (init_empty_weights from MetaMixin)
        from sglang.srt.model_loader.loader import get_model_architecture
        model_class, _ = get_model_architecture(model_config)
        model = model_class(config=hf_config, quant_config=None)

        # Clean up distributed env so Engine's subprocess can reinitialize
        from sglang.srt.distributed import (
            destroy_model_parallel,
            destroy_distributed_environment,
        )
        try:
            destroy_model_parallel()
        except Exception:
            pass
        try:
            destroy_distributed_environment()
        except Exception:
            pass

        self.tokenizer = AutoTokenizer.from_pretrained(
            repo_id, trust_remote_code=trust_remote_code,
        )
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return model

    def _load(self, repo_id: str, **kwargs) -> "Module":
        from .engine import NNsightEngine

        meta_model = self._load_meta(repo_id, **kwargs)

        # Filter kwargs for SGLang Engine — remove nnsight/HF-specific ones
        sglang_kwargs = {k: v for k, v in kwargs.items() if k not in (
            "dispatch", "meta_buffers", "rename",
        )}

        # Force settings required for nnsight intervention support
        sglang_kwargs["disable_cuda_graph"] = True
        sglang_kwargs["disable_overlap_schedule"] = True

        self.engine = NNsightEngine(
            model_path=repo_id,
            **sglang_kwargs,
        )

        return meta_model

    def _prepare_input(
        self, *args, **kwargs
    ) -> Tuple[Tuple[Any], Dict[str, Any], int]:
        """Normalize a single user input for SGLang.

        Accepts a string prompt or a list of token IDs.
        Each invoke must contain exactly one prompt.
        """
        prompts = []
        sampling_params = []

        for arg in args:
            if isinstance(arg, str):
                prompts.append(arg)
            elif isinstance(arg, list) and arg and isinstance(arg[0], int):
                # Token ID list — SGLang accepts input_ids directly
                prompts.append(arg)
            elif isinstance(arg, dict) and "input_ids" in arg:
                # HF tokenizer dict
                input_ids = arg["input_ids"]
                if isinstance(input_ids, torch.Tensor):
                    input_ids = input_ids.squeeze().tolist()
                if isinstance(input_ids[0], list):
                    input_ids = input_ids[0]
                prompts.append(input_ids)
            else:
                raise ValueError(
                    f"Unsupported input type: {type(arg)}. "
                    "Use a string prompt, token ID list, or HF tokenizer dict."
                )

        # Build sampling params dict from kwargs
        sp = {}
        for key in ("temperature", "top_p", "top_k", "max_new_tokens",
                     "max_tokens", "stop", "frequency_penalty",
                     "presence_penalty", "repetition_penalty"):
            if key in kwargs:
                sp[key] = kwargs.pop(key)

        for _ in prompts:
            sampling_params.append(dict(sp))

        return (prompts, sampling_params), kwargs, len(prompts)

    def _batch(
        self, batched_inputs, prompts, sampling_params, **kwargs
    ) -> Tuple[Tuple[Any], Dict[str, Any]]:
        """Combine multiple invokes into a single batch."""
        kwargs = {**kwargs, **batched_inputs[1]}

        if len(batched_inputs[0]) == 0:
            return (prompts, sampling_params), kwargs

        batched_args = batched_inputs[0]
        batched_args[0].extend(prompts)
        batched_args[1].extend(sampling_params)
        return batched_args, kwargs

    def _serialize_mediators(
        self,
        prompts: List,
        sampling_params: List[Dict],
        **kwargs,
    ) -> Tuple[List, List[Dict]]:
        """Serialize mediators and attach them to sampling params."""
        input_mediators = []
        for mediator in self._interleaver.mediators:
            if mediator.batch_group is not None:
                mediator.intervention.__source__ = "".join(mediator.info.source)
                input_mediators.append(mediator)

        # Compute saved_names
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
            nnsight_data = {
                "nnsight_mediator": serialize(mediator),
                "nnsight_trace_id": trace_id,
                "nnsight_trace_idx": idx,
                "nnsight_saved_names": saved_names,
                "nnsight_expected_count": len(input_mediators),
            }
            # Encode as base64 string with prefix for transport
            encoded = "nnsight:" + base64.b64encode(
                pickle.dumps(nnsight_data)
            ).decode("ascii")

            sampling_params[param_idx]["_nnsight_clp"] = encoded
            param_idx += 1

        return prompts, sampling_params

    def __call__(
        self,
        prompts: List,
        sampling_params: List[Dict],
        **kwargs,
    ) -> Any:
        """Execute SGLang generation with NNsight interventions."""
        prompts, sampling_params = self._serialize_mediators(
            prompts, sampling_params, **kwargs
        )

        # Extract nnsight custom_logit_processor strings
        clp_list = []
        for sp in sampling_params:
            clp_list.append(sp.pop("_nnsight_clp", None))

        # Generate unique request IDs
        rids = [str(uuid.uuid4()) for _ in prompts]

        # Separate string prompts from token ID prompts
        text_prompts = []
        input_ids_list = []
        for p in prompts:
            if isinstance(p, str):
                text_prompts.append(p)
                input_ids_list.append(None)
            else:
                text_prompts.append(None)
                input_ids_list.append(p)

        # Call engine.generate for each prompt individually
        # (SGLang batches internally via its scheduler)
        for i, (prompt, input_ids, sp, clp, rid) in enumerate(
            zip(text_prompts, input_ids_list, sampling_params, clp_list, rids)
        ):
            output = self.engine.generate(
                prompt=prompt,
                input_ids=input_ids,
                sampling_params=sp,
                custom_logit_processor=clp,
                rid=rid,
            )

        # Collect saves via RPC
        try:
            saves_bytes = self.engine.collect_nnsight_saves(rids)
            if saves_bytes:
                saves = pickle.loads(saves_bytes)
                for value in saves.values():
                    save(value)
                push_variables(
                    self._interleaver.mediators[0].info.frame, saves
                )
        except Exception:
            pass

    def interleave(self, fn: Callable, *args, **kwargs):
        """Execute the traced function with SGLang."""
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
        state["engine"] = None  # ZMQ sockets can't be pickled
        if self.tokenizer is not None:
            self.tokenizer._persistent_id = "Tokenizer"
        state["tokenizer"] = self.tokenizer
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.engine = state["engine"]
        self.tokenizer = state["tokenizer"]
