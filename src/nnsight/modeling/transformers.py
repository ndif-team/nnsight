from .huggingface import HuggingFaceModel

from torch.nn.modules import Module
from transformers import AutoConfig, PreTrainedModel, PretrainedConfig
from typing import Optional
from typing import Type
from transformers.models.auto import modeling_auto
from transformers import AutoModel


class TransformersModel(HuggingFaceModel):

    def __init__(self, *args, config_model: Type[PretrainedConfig] = None, automodel: Type[AutoModel] = AutoModel, **kwargs):

        self.config: PretrainedConfig = config_model
        
        self.automodel = (
            automodel
            if not isinstance(automodel, str)
            else getattr(modeling_auto, automodel)
        )

        super().__init__(*args, **kwargs)

    def _load_config(self, repo_id: str, revision: Optional[str] = None, **kwargs):

        if self.config is None:

            self.config = AutoConfig.from_pretrained(
                repo_id, revision=revision, **kwargs
            )

    def _load_meta(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        **kwargs,
    ) -> Module:

        self._load_config(repo_id, revision=revision, **kwargs)

        model = self.automodel.from_config(self.config, trust_remote_code=True)

        self.config = model.config

        return model

    def _load(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        **kwargs,
    ) -> PreTrainedModel:

        self._load_config(repo_id, revision=revision, **kwargs)

        model = self.automodel.from_pretrained(repo_id, revision=revision, **kwargs)

        self.config = model.config

        return model
