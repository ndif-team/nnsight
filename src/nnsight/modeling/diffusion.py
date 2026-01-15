from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

import torch
from diffusers import DiffusionPipeline, pipelines
from transformers import BatchEncoding, PreTrainedTokenizerBase

from .. import util
from .huggingface import HuggingFaceModel
import diffusers

# from diffusers.models.modeling_utils import ContextManagers, no_init_weights
from contextlib import contextmanager
from diffusers.models.modeling_utils import ContextManagers


class Diffuser(util.WrapperModule):
    def __init__(
        self, automodel: Type[DiffusionPipeline] = DiffusionPipeline, *args, meta=False, device_map=None,**kwargs
    ) -> None:
        super().__init__()

        if meta:

            import diffusers
            import copy
            import accelerate
            from diffusers import __version__
            from diffusers.models.modeling_utils import no_init_weights
            from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT

            @classmethod
            def _from_config(cls, pretrained_model_name_or_path, *args, **kwargs):

                config_path = pretrained_model_name_or_path
                cache_dir = kwargs.pop("cache_dir", None)
                ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
                force_download = kwargs.pop("force_download", False)
                from_flax = kwargs.pop("from_flax", False)
                proxies = kwargs.pop("proxies", None)
                output_loading_info = kwargs.pop("output_loading_info", False)
                local_files_only = kwargs.pop("local_files_only", None)
                token = kwargs.pop("token", None)
                revision = kwargs.pop("revision", None)
                torch_dtype = kwargs.pop("torch_dtype", None)
                subfolder = kwargs.pop("subfolder", None)
                device_map = kwargs.pop("device_map", None)
                max_memory = kwargs.pop("max_memory", None)
                offload_folder = kwargs.pop("offload_folder", None)
                offload_state_dict = kwargs.pop("offload_state_dict", None)
                low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
                variant = kwargs.pop("variant", None)
                use_safetensors = kwargs.pop("use_safetensors", None)
                quantization_config = kwargs.pop("quantization_config", None)
                dduf_entries = kwargs.pop("dduf_entries", None)
                disable_mmap = kwargs.pop("disable_mmap", False)
                parallel_config = kwargs.pop("parallel_config", None)

                user_agent = {
                    "diffusers": __version__,
                    "file_type": "model",
                    "framework": "pytorch",
                }
                dduf_entries = kwargs.get("dduf_entries", None)

                # load config
                config, unused_kwargs, commit_hash = cls.load_config(
                    config_path,
                    cache_dir=cache_dir,
                    return_unused_kwargs=True,
                    return_commit_hash=True,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    dduf_entries=dduf_entries,
                    **kwargs,
                )
                # no in-place modification of the original config.
                config = copy.deepcopy(config)

                init_contexts = [no_init_weights()]

                if low_cpu_mem_usage:
                    init_contexts.append(accelerate.init_empty_weights())

                with ContextManagers(init_contexts):
                    model = cls.from_config(config, **unused_kwargs)

                return model

            import diffusers
            diffusers.models.modeling_utils.ModelMixin._from_config = _from_config
            diffusers.pipelines.pipeline_loading_utils.ALL_IMPORTABLE_CLASSES["ModelMixin"] = ["save_pretrained", "_from_config"]
            diffusers.pipelines.pipeline_loading_utils.ALL_IMPORTABLE_CLASSES["PretrainedModel"] = ["save_pretrained", "from_config"]

            import transformers

            @classmethod
            def from_config(
                cls,
                pretrained_model_name_or_path,
                *model_args,
                config = None,
                cache_dir = None,
                ignore_mismatched_sizes: bool = False,
                force_download: bool = False,
                local_files_only: bool = False,
                token: Optional[Union[str, bool]] = None,
                revision: str = "main",
                use_safetensors: Optional[bool] = None,
                weights_only: bool = True,
                **kwargs,
                ):

                from transformers.configuration_utils import PretrainedConfig
                from transformers.modeling_utils import cached_file
                from transformers.modeling_utils import extract_commit_hash
                from transformers.modeling_utils import is_deepspeed_zero3_enabled
                from transformers.modeling_utils import CONFIG_NAME
                from transformers.modeling_utils import _is_ds_init_called
                from transformers.modeling_utils import get_torch_context_manager_or_global_device
                from transformers.modeling_utils import VLMS

                state_dict = kwargs.pop("state_dict", None)
                from_tf = kwargs.pop("from_tf", False)
                from_flax = kwargs.pop("from_flax", False)
                proxies = kwargs.pop("proxies", None)
                output_loading_info = kwargs.pop("output_loading_info", False)
                use_auth_token = kwargs.pop("use_auth_token", None)
                from_pipeline = kwargs.pop("_from_pipeline", None)
                from_auto_class = kwargs.pop("_from_auto", False)
                dtype = kwargs.pop("dtype", None)
                torch_dtype = kwargs.pop("torch_dtype", None)  # kept for BC
                device_map = kwargs.pop("device_map", None)
                max_memory = kwargs.pop("max_memory", None)
                offload_folder = kwargs.pop("offload_folder", None)
                offload_state_dict = kwargs.pop("offload_state_dict", False)
                offload_buffers = kwargs.pop("offload_buffers", False)
                load_in_8bit = kwargs.pop("load_in_8bit", False)
                load_in_4bit = kwargs.pop("load_in_4bit", False)
                quantization_config = kwargs.pop("quantization_config", None)
                subfolder = kwargs.pop("subfolder", "")
                commit_hash = kwargs.pop("_commit_hash", None)
                variant = kwargs.pop("variant", None)
                adapter_kwargs = kwargs.pop("adapter_kwargs", {})
                adapter_name = kwargs.pop("adapter_name", "default")
                generation_config = kwargs.pop("generation_config", None)
                gguf_file = kwargs.pop("gguf_file", None)
                tp_plan = kwargs.pop("tp_plan", None)
                tp_size = kwargs.pop("tp_size", None)
                distributed_config = kwargs.pop("distributed_config", None)
                device_mesh = kwargs.pop("device_mesh", None)
                trust_remote_code = kwargs.pop("trust_remote_code", None)
                use_kernels = kwargs.pop("use_kernels", False)

                key_mapping = kwargs.pop("key_mapping", None)
                # Load models with hardcoded key mapping on class for VLMs only, to keep BC and standardize model
                if key_mapping is None and any(
                    allowed_name in class_name.__name__.lower() for class_name in cls.__mro__[:-1] for allowed_name in VLMS
                ):
                    key_mapping = cls._checkpoint_conversion_mapping

                # if distributed_config is not None:
                #     tp_plan = "auto"

                # Not used anymore -- remove them from the kwargs
                _ = kwargs.pop("resume_download", None)
                _ = kwargs.pop("mirror", None)
                _ = kwargs.pop("_fast_init", True)
                _ = kwargs.pop("low_cpu_mem_usage", None)

                if use_auth_token is not None:
                    # warnings.warn(
                    #     "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                    #     FutureWarning,
                    # )
                    if token is not None:
                        raise ValueError(
                            "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                        )
                    token = use_auth_token

                if token is not None and adapter_kwargs is not None and "token" not in adapter_kwargs:
                    adapter_kwargs["token"] = token

                if commit_hash is None:
                    if not isinstance(config, PretrainedConfig):
                        # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
                        resolved_config_file = cached_file(
                            pretrained_model_name_or_path,
                            CONFIG_NAME,
                            cache_dir=cache_dir,
                            force_download=force_download,
                            proxies=proxies,
                            local_files_only=local_files_only,
                            token=token,
                            revision=revision,
                            subfolder=subfolder,
                            _raise_exceptions_for_gated_repo=False,
                            _raise_exceptions_for_missing_entries=False,
                            _raise_exceptions_for_connection_errors=False,
                        )
                        commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
                    else:
                        commit_hash = getattr(config, "_commit_hash", None)

                _adapter_model_path = None

                # Potentially detect context manager or global device, and use it (only if no device_map was provided)
                if device_map is None and not is_deepspeed_zero3_enabled():
                    device_in_context = get_torch_context_manager_or_global_device()
                    device_map = device_in_context

                user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
                if from_pipeline is not None:
                    user_agent["using_pipeline"] = from_pipeline

                # Load config if we don't provide a configuration
                if not isinstance(config, PretrainedConfig):
                    config_path = config if config is not None else pretrained_model_name_or_path
                    config, model_kwargs = cls.config_class.from_pretrained(
                        config_path,
                        cache_dir=cache_dir,
                        return_unused_kwargs=True,
                        force_download=force_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=token,
                        revision=revision,
                        subfolder=subfolder,
                        gguf_file=gguf_file,
                        _from_auto=from_auto_class,
                        _from_pipeline=from_pipeline,
                        **kwargs,
                    )
                    if "gguf_file" in model_kwargs:
                        model_kwargs.pop("gguf_file")
                    
                else:
                    config = copy.deepcopy(config)
                    model_kwargs = kwargs

                # Because some composite configs call super().__init__ before instantiating the sub-configs, we need this call
                # to correctly redispatch recursively if the kwarg is provided
                if "attn_implementation" in kwargs:
                    config._attn_implementation = kwargs.pop("attn_implementation")

                is_quantized = False

                config.name_or_path = pretrained_model_name_or_path
                model_init_context = cls.get_init_context(is_quantized, _is_ds_init_called)
                config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
                with ContextManagers(model_init_context):
                    # Let's make sure we don't run the init function of buffer modules
                    model = cls(config, *model_args, **model_kwargs)

                return model

            transformers.modeling_utils.PreTrainedModel.from_config = from_config

            self.pipeline = automodel.from_pretrained(*args, **kwargs)

            diffusers.pipelines.pipeline_loading_utils.ALL_IMPORTABLE_CLASSES["ModelMixin"] = ["save_pretrained", "from_pretrained"]
            diffusers.pipelines.pipeline_loading_utils.ALL_IMPORTABLE_CLASSES["PretrainedModel"] = ["save_pretrained", "from_pretrained"]

            # breakpoint()

            # self.pipeline = automodel.from_config(*args, device_map=device_map, **kwargs)

            # breakpoint()

        else:

            self.pipeline = automodel.from_pretrained(*args, device_map=device_map, **kwargs)


        for key, value in self.pipeline.__dict__.items():
            if (
                isinstance(value, torch.nn.Module)
                or isinstance(value, PreTrainedTokenizerBase)
            ):

                setattr(self, key, value)

        self.config = self.pipeline.config
                
    def generate(self, *args, **kwargs):
        return self.pipeline.generate(*args, **kwargs)

class DiffusionModel(HuggingFaceModel):

    def __init__(
        self, *args, automodel: Type[DiffusionPipeline] = DiffusionPipeline, **kwargs
    ) -> None:

        self.automodel = (
            automodel
            if not isinstance(automodel, str)
            else getattr(pipelines, automodel)
        )

        self._model: Diffuser = None

        super().__init__(*args, **kwargs)

    def _load_meta(self, repo_id: str, revision: Optional[str] = None, **kwargs):

        model = Diffuser(
            self.automodel,
            repo_id,
            meta=True,
            revision=revision,
            device_map=None,
            low_cpu_mem_usage=False,
            **kwargs,
        )

        return model

    def _load(self, repo_id: str, revision: Optional[str] = None, device_map=None, **kwargs) -> Diffuser:
        # https://github.com/huggingface/diffusers/issues/11555
        device_map = "balanced" if device_map == "auto" or device_map is None else device_map

        model = Diffuser(
            self.automodel, 
            repo_id, 
            revision=revision, 
            device_map=device_map, 
            **kwargs
        )

        return model

    def _prepare_input(
        self,
        inputs: Union[str, List[str]],
    ) -> Any:

        if isinstance(inputs, str):
            inputs = [inputs]

        return ((inputs,), {})

    def _batch(
        self,
        batched_inputs: Optional[Dict[str, Any]],
        prepared_inputs: BatchEncoding,
    ) -> torch.Tensor:
        if batched_inputs is None:

            return ((prepared_inputs,), {})

        return (batched_inputs + prepared_inputs,)

    def __call__(self, prepared_inputs: Any, *args, **kwargs):

        return self._model.unet(
            prepared_inputs,
            *args,
            **kwargs,
        )

    def __nnsight_generate__(
        self, prepared_inputs: Any, *args, seed: int = None, **kwargs
    ):

        if self._interleaver is not None:
            steps = kwargs.get("num_inference_steps")
            if steps is None:
                try:
                    steps = (
                        inspect.signature(self.pipeline.generate)
                        .parameters["num_inference_steps"]
                        .default
                    )
                except:
                    steps = 50
            self._interleaver.default_all = steps

        generator = torch.Generator(self.device)

        if seed is not None:

            if isinstance(prepared_inputs, list) and len(prepared_inputs) > 1:
                generator = [
                    torch.Generator(self.device).manual_seed(seed + offset)
                    for offset in range(
                        len(prepared_inputs) * kwargs.get("num_images_per_prompt", 1)
                    )
                ]
            else:
                generator = generator.manual_seed(seed)

        output = self._model.pipeline(
            prepared_inputs, *args, generator=generator, **kwargs
        )

        if self._interleaver is not None:
            self._interleaver.default_all = None

        output = self._model(output)

        return output

    def _remoteable_model_key(self) -> str:
        return json.dumps(
            {"repo_id": self.repo_id}  # , "torch_dtype": str(self._model.dtype)}
        )

    @classmethod
    def _remoteable_from_model_key(cls, model_key: str, **kwargs) -> Self:

        kwargs = {**json.loads(model_key), **kwargs}

        repo_id = kwargs.pop("repo_id")

        return DiffusionModel(repo_id, **kwargs)


if TYPE_CHECKING:

    class DiffusionModel(DiffusionModel, DiffusionPipeline):

        def generate(self, *args, **kwargs):
            return self._model.pipeline(*args, **kwargs)
