from .huggingface import HuggingFaceModel
import torch
from huggingface_hub import HfApi, constants
from huggingface_hub.file_download import repo_folder_name
from torch.nn.modules import Module
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoTokenizer, BatchEncoding, PretrainedConfig,
                          PreTrainedModel, PreTrainedTokenizer)
from typing import Optional, Self
from transformers.models.llama.configuration_llama import LlamaConfig
from typing_extensions import Self
import json

class TransformersModel(HuggingFaceModel):
    
    def _load_meta(
        self,
        repo_id: str,
        revision:Optional[str] = None,
        patch_llama_scan: bool = True,
        **kwargs,
    ) -> PreTrainedModel:

        self._load_config(repo_id, revision=revision, **kwargs)


        if (
            patch_llama_scan
            and isinstance(self.config, LlamaConfig)
            and isinstance(self.config.rope_scaling, dict)
            and "rope_type" in self.config.rope_scaling
        ):
            self.config.rope_scaling["rope_type"] = "default"

        model = self.automodel.from_config(self.config, trust_remote_code=True)
        
        self.config = model.config
        
        self._patch_generation_config(model)

        return model
    
    def _load(
        self,
        repo_id: str,
        revision:Optional[str] = None,
        patch_llama_scan: bool = True,
        **kwargs,
    ) -> PreTrainedModel:

        self._load_config(repo_id, revision=revision, **kwargs)

        if (
            patch_llama_scan
            and isinstance(self.config, LlamaConfig)
            and isinstance(self.config.rope_scaling, dict)
            and "rope_type" in self.config.rope_scaling
        ):
            self.config.rope_scaling["rope_type"] = "llama3"

        model = self.automodel.from_pretrained(repo_id, config=self.config, revision=revision, **kwargs)
        
        self.config = model.config
        
        self._patch_generation_config(model)
        
        return model
    
    
    def _remoteable_model_key(self) -> str:
        
        repo_id = HfApi().model_info(self.repo_id).id
        
        return json.dumps(
            {"repo_id": repo_id, "revision": self.revision}  # , "torch_dtype": str(self._model.dtype)}
        )

    @classmethod
    def _remoteable_from_model_key(cls, model_key: str, **kwargs) -> Self:

        kwargs = {**json.loads(model_key), **kwargs}

        repo_id = kwargs.pop("repo_id")

        revision = kwargs.pop("revision", "main")

        return LanguageModel(repo_id, revision=revision, **kwargs)