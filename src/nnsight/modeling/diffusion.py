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

        # if device_map is None:
        #     breakpoint()
        #     self.pipeline = automodel.from_config(*args, **kwargs)
        #     self.pipeline = automodel.from_config(self.pipeline.config, *args, **kwargs)
        # else:

        # @contextmanager
        # def noop():
        #     try:
        #         yield
        #     finally:
        #         pass


        # diffusers.models.modeling_utils.no_init_weights = noop
        
        # old_dict = diffusers.pipelines.pipeline_loading_utils.LOADABLE_CLASSES
        # new_dict = {
        #         "diffusers": {
        #             "ModelMixin": ["save_pretrained", "from_pretrained"],
        #             "SchedulerMixin": ["save_pretrained", "from_pretrained"],
        #             "DiffusionPipeline": ["save_pretrained", "from_pretrained"],
        #             "OnnxRuntimeModel": ["save_pretrained", "from_pretrained"],
        #             "BaseGuidance": ["save_pretrained", "from_pretrained"],
        #         },
        #         "transformers": {
        #             "PreTrainedTokenizer": ["save_pretrained", "from_pretrained"],
        #             "PreTrainedTokenizerFast": ["save_pretrained", "from_pretrained"],
        #             "PreTrainedModel": ["save_pretrained", "from_config"],
        #             "FeatureExtractionMixin": ["save_pretrained", "from_pretrained"],
        #             "ProcessorMixin": ["save_pretrained", "from_pretrained"],
        #             "ImageProcessingMixin": ["save_pretrained", "from_pretrained"],
        #         },
        #         "onnxruntime.training": {
        #             "ORTModule": ["save_pretrained", "from_pretrained"],
        #         },
        # }
        # for library in new_dict:
        #     diffusers.pipelines.pipeline_loading_utils.ALL_IMPORTABLE_CLASSES.update(new_dict[library])


        # @contextmanager
        # def no_init():
        #     def _skip_init(*args, **kwargs):
        #         pass

        #     for name, init_func in diffusers.models.modeling_utils.TORCH_INIT_FUNCTIONS.items():
        #         setattr(torch.nn.init, name, _skip_init)
        #     try:
        #         yield
        #     finally:
        #         # Restore the original initialization functions
        #         for name, init_func in diffusers.models.modeling_utils.TORCH_INIT_FUNCTIONS.items():
        #             setattr(torch.nn.init, name, init_func)
        

        # with diffusers.models.modeling_utils.no_init_weights():
        # with diffusers.models.modeling_utils.ContextManagers([no_init()]):

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

            # @classmethod
            # def from_config(cls, *args, **kwargs):
            #     breakpoint()

            #     download_kwargs = {
            #         "cache_dir": kwargs.get("cache_dir", None),
            #         "force_download": kwargs.get("force_download", False),
            #         "proxies": kwargs.get("proxies", None),
            #         "local_files_only": kwargs.get("local_files_only", False),
            #         "token": kwargs.get("token", None),
            #         "revision": kwargs.get("revision", "main"),
            #         "subfolder": kwargs.pop("subfolder", ""),
            #     }
            #     config, model_kwargs = cls.config_class.from_pretrained(
            #         args[0],
            #         return_unused_kwargs=True,
            #         gguf_file=kwargs.pop("gguf_file", None),
            #         _from_auto=kwargs.pop("_from_auto", False),
            #         _from_pipeline=kwargs.pop("_from_pipeline", None),
            #         **download_kwargs,
            #         **kwargs,
            #     )
            #     if "gguf_file" in model_kwargs:
            #         model_kwargs.pop("gguf_file")

            #     model = transformers.modeling_utils.PreTrainedModel._from_config(config, **kwargs)

            #     return model

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

                # For BC on torch_dtype argument
                # if torch_dtype is not None:
                #     logger.warning_once("`torch_dtype` is deprecated! Use `dtype` instead!")
                #     # If both kwargs are provided, use `dtype`
                #     dtype = dtype if dtype is not None else torch_dtype

                # if state_dict is not None and (pretrained_model_name_or_path is not None or gguf_file is not None):
                #     raise ValueError(
                #         "`state_dict` cannot be passed together with a model name or a `gguf_file`. Use one of the two loading strategies."
                #     )
                # if tp_size is not None and tp_plan is None:
                #     raise ValueError("tp_plan has to be set when tp_size is passed.")
                # if tp_plan is not None and tp_plan != "auto":
                #     # TODO: we can relax this check when we support taking tp_plan from a json file, for example.
                #     raise ValueError(f"tp_plan supports 'auto' only for now but got {tp_plan}.")
                # if tp_plan is not None and device_map is not None:
                #     raise ValueError(
                #         "`tp_plan` and `device_map` are mutually exclusive. Choose either one for parallelization."
                #     )

                # if device_map == "auto" and int(os.environ.get("WORLD_SIZE", "0")):
                #     logger.info(
                #         "You've set device_map=`auto` while triggering a distributed run with torchrun. This might lead to unexpected behavior. "
                #         "If your plan is to load the model on each device, you should set device_map={"
                #         ": PartialState().process_index} where PartialState comes from accelerate library"
                #     )

                # We need to correctly dispatch the model on the current process device. The easiest way for this is to use a simple
                # `device_map` pointing to the correct device
                # if tp_plan is not None:
                #     if device_mesh is None:
                #         tp_plan, device_map, device_mesh, tp_size = initialize_tensor_parallelism(tp_plan, tp_size=tp_size)
                #     else:
                #         if device_mesh.ndim > 1:
                #             if "tp" not in device_mesh.mesh_dim_names:
                #                 raise ValueError(
                #                     "When using `tp_plan` and n-d `device_mesh`, it must contain a 'tp' dimension. "
                #                     "Please provide a valid `device_mesh`."
                #                 )
                #             device_mesh = device_mesh["tp"]
                #         tp_size = device_mesh.size()
                #         device_map = torch.device(f"{device_mesh.device_type}:{int(os.environ['LOCAL_RANK'])}")

                #     if tp_size is None:
                #         tp_size = torch.distributed.get_world_size()

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

                # if use_safetensors is None and not is_safetensors_available():
                #     use_safetensors = False

                # if gguf_file is not None and not is_accelerate_available():
                #     raise ValueError("accelerate is required when loading a GGUF file `pip install accelerate`.")

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

                # if is_peft_available():
                #     _adapter_model_path = adapter_kwargs.pop("_adapter_model_path", None)

                #     if _adapter_model_path is None:
                #         _adapter_model_path = find_adapter_config_file(
                #             pretrained_model_name_or_path,
                #             cache_dir=cache_dir,
                #             force_download=force_download,
                #             proxies=proxies,
                #             local_files_only=local_files_only,
                #             _commit_hash=commit_hash,
                #             **adapter_kwargs,
                #         )
                #     if _adapter_model_path is not None and os.path.isfile(_adapter_model_path):
                #         with open(_adapter_model_path, "r", encoding="utf-8") as f:
                #             _adapter_model_path = pretrained_model_name_or_path
                #             pretrained_model_name_or_path = json.load(f)["base_model_name_or_path"]
                # else:
                #     _adapter_model_path = None

                _adapter_model_path = None

                # Potentially detect context manager or global device, and use it (only if no device_map was provided)
                if device_map is None and not is_deepspeed_zero3_enabled():
                    device_in_context = get_torch_context_manager_or_global_device()
                    device_map = device_in_context

                # change device_map into a map if we passed an int, a str or a torch.device
                # if isinstance(device_map, torch.device):
                #     device_map = {"": device_map}
                # elif isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
                #     try:
                #         device_map = {"": torch.device(device_map)}
                #     except RuntimeError:
                #         raise ValueError(
                #             "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
                #             f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
                #         )
                # elif isinstance(device_map, int):
                #     if device_map < 0:
                #         raise ValueError(
                #             "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
                #         )
                #     else:
                #         device_map = {"": device_map}

                # if device_map is not None:
                #     if is_deepspeed_zero3_enabled():
                #         raise ValueError("DeepSpeed Zero-3 is not compatible with passing a `device_map`.")
                #     if not is_accelerate_available():
                #         raise ValueError(
                #             "Using a `device_map`, `tp_plan`, `torch.device` context manager or setting `torch.set_default_device(device)` "
                #             "requires `accelerate`. You can install it with `pip install accelerate`"
                #         )

                # handling bnb config from kwargs, remove after `load_in_{4/8}bit` deprecation.
                # if load_in_4bit or load_in_8bit:
                #     if quantization_config is not None:
                #         raise ValueError(
                #             "You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing "
                #             "`quantization_config` argument at the same time."
                #         )

                #     # preparing BitsAndBytesConfig from kwargs
                #     config_dict = {k: v for k, v in kwargs.items() if k in inspect.signature(BitsAndBytesConfig).parameters}
                #     config_dict = {**config_dict, "load_in_4bit": load_in_4bit, "load_in_8bit": load_in_8bit}
                #     quantization_config, kwargs = BitsAndBytesConfig.from_dict(
                #         config_dict=config_dict, return_unused_kwargs=True, **kwargs
                #     )
                #     logger.warning(
                #         "The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. "
                #         "Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead."
                #     )

                # from_pt = not (from_tf | from_flax)

                user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
                if from_pipeline is not None:
                    user_agent["using_pipeline"] = from_pipeline

                # if is_offline_mode() and not local_files_only:
                #     logger.info("Offline mode: forcing local_files_only=True")
                #     local_files_only = True

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

                # transformers_explicit_filename = getattr(config, "transformers_weights", None)

                # if transformers_explicit_filename is not None:
                #     if not transformers_explicit_filename.endswith(
                #         ".safetensors"
                #     ) and not transformers_explicit_filename.endswith(".safetensors.index.json"):
                #         raise ValueError(
                #             "The transformers file in the config seems to be incorrect: it is neither a safetensors file "
                #             "(*.safetensors) nor a safetensors index file (*.safetensors.index.json): "
                #             f"{transformers_explicit_filename}"
                #         )

                # hf_quantizer, config, dtype, device_map = get_hf_quantizer(
                #     config, quantization_config, dtype, from_tf, from_flax, device_map, weights_only, user_agent
                # )

                # if gguf_file is not None and hf_quantizer is not None:
                #     raise ValueError(
                #         "You cannot combine Quantization and loading a model from a GGUF file, try again by making sure you did not passed a `quantization_config` or that you did not load a quantized model from the Hub."
                #     )

                # if (
                #     gguf_file
                #     and device_map is not None
                #     and ((isinstance(device_map, dict) and "disk" in device_map.values()) or "disk" in device_map)
                # ):
                #     raise RuntimeError(
                #         "One or more modules is configured to be mapped to disk. Disk offload is not supported for models "
                #         "loaded from GGUF files."
                #     )

                # checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
                #     pretrained_model_name_or_path=pretrained_model_name_or_path,
                #     subfolder=subfolder,
                #     variant=variant,
                #     gguf_file=gguf_file,
                #     from_tf=from_tf,
                #     from_flax=from_flax,
                #     use_safetensors=use_safetensors,
                #     cache_dir=cache_dir,
                #     force_download=force_download,
                #     proxies=proxies,
                #     local_files_only=local_files_only,
                #     token=token,
                #     user_agent=user_agent,
                #     revision=revision,
                #     commit_hash=commit_hash,
                #     is_remote_code=cls._auto_class is not None,
                #     transformers_explicit_filename=transformers_explicit_filename,
                # )

                # is_sharded = sharded_metadata is not None
                # is_quantized = hf_quantizer is not None
                is_quantized = False
                # is_from_file = pretrained_model_name_or_path is not None or gguf_file is not None

                # if (
                #     is_safetensors_available()
                #     and is_from_file
                #     and not is_sharded
                #     and checkpoint_files[0].endswith(".safetensors")
                # ):
                #     with safe_open(checkpoint_files[0], framework="pt") as f:
                #         metadata = f.metadata()

                #     if metadata is None:
                #         # Assume it's a pytorch checkpoint (introduced for timm checkpoints)
                #         pass
                #     elif metadata.get("format") == "pt":
                #         pass
                #     elif metadata.get("format") == "tf":
                #         from_tf = True
                #         logger.info("A TensorFlow safetensors file is being loaded in a PyTorch model.")
                #     elif metadata.get("format") == "flax":
                #         from_flax = True
                #         logger.info("A Flax safetensors file is being loaded in a PyTorch model.")
                #     elif metadata.get("format") == "mlx":
                #         # This is a mlx file, we assume weights are compatible with pt
                #         pass
                #     else:
                #         raise ValueError(
                #             f"Incompatible safetensors file. File metadata is not ['pt', 'tf', 'flax', 'mlx'] but {metadata.get('format')}"
                #         )

                # from_pt = not (from_tf | from_flax)

                # if from_pt:
                #     # if gguf_file:
                #     #     from .modeling_gguf_pytorch_utils import load_gguf_checkpoint

                #     #     # we need a dummy model to get the state_dict - for this reason, we keep the state_dict as if it was
                #     #     # passed directly as a kwarg from now on
                #     #     with torch.device("meta"):
                #     #         dummy_model = cls(config)
                #     #         breakpoint()
                #     #     state_dict = load_gguf_checkpoint(checkpoint_files[0], return_tensors=True, model_to_load=dummy_model)[
                #     #         "tensors"
                #     #     ]

                #     # Find the correct dtype based on current state
                #     config, dtype, dtype_orig = _get_dtype(
                #         cls, dtype, checkpoint_files, config, sharded_metadata, state_dict, weights_only
                #     )

                config.name_or_path = pretrained_model_name_or_path
                model_init_context = cls.get_init_context(is_quantized, _is_ds_init_called)
                config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
                with ContextManagers(model_init_context):
                    # Let's make sure we don't run the init function of buffer modules
                    model = cls(config, *model_args, **model_kwargs)


                breakpoint()

                return model

            transformers.modeling_utils.PreTrainedModel.from_config = from_config

            self.pipeline = automodel.from_pretrained(*args, **kwargs)

            diffusers.pipelines.pipeline_loading_utils.ALL_IMPORTABLE_CLASSES["ModelMixin"] = ["save_pretrained", "from_pretrained"]
            diffusers.pipelines.pipeline_loading_utils.ALL_IMPORTABLE_CLASSES["PretrainedModel"] = ["save_pretrained", "from_pretrained"]

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
