from __future__ import annotations

import json
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
from huggingface_hub import constants
from huggingface_hub.file_download import repo_folder_name

from nnsight import CONFIG
import torch
from torch.nn.modules import Module
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.generation.utils import GenerationMixin
from transformers.models.auto import modeling_auto
from transformers.models.llama.configuration_llama import LlamaConfig
from typing_extensions import Self

from ..intervention.envoy import Envoy
from ..intervention.tracing.tracer import InterleavingTracer
from ..util import WrapperModule
from .mixins import RemoteableMixin


class LanguageModel(RemoteableMixin):
    """LanguageModels are NNsight wrappers around transformers language models.

    Inputs can be in the form of:
        Prompt: (str)
        Prompts: (List[str])
        Batched prompts: (List[List[str]])
        Tokenized prompt: (Union[List[int], torch.Tensor])
        Tokenized prompts: (Union[List[List[int]], torch.Tensor])
        Direct input: (Dict[str,Any])

    If using a custom model, you also need to provide the tokenizer like ``LanguageModel(custom_model, tokenizer=tokenizer)``

    Calls to generate pass arguments downstream to :func:`GenerationMixin.generate`

    Attributes:
        config (PretrainedConfig): Huggingface config file loaded from repository or checkpoint.
        tokenizer (PreTrainedTokenizer): Tokenizer for LMs.
        automodel (Type): AutoModel type from transformer auto models.
        model (PreTrainedModel): Meta version of underlying auto model.

    """

    tokenizer: PreTrainedTokenizer

    def __init__(
        self,
        *args,
        config: Optional[PretrainedConfig] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        automodel: Type[AutoModel] = AutoModelForCausalLM,
        import_edits:Union[bool, str] = False,
        **kwargs,
    ) -> None:

        self.automodel = (
            automodel
            if not isinstance(automodel, str)
            else getattr(modeling_auto, automodel)
        )

        self.config = config
        self.tokenizer = tokenizer
        self.repo_id: str = args[0] if isinstance(args[0], str) else None

        super().__init__(*args, **kwargs)
        
        if import_edits:
            
            if isinstance(import_edits, str):
                
                self.import_edits(variant=import_edits)
                
            else:
            
                self.import_edits()
            

        self.generator: Envoy = WrapperModule()
        
    def export_edits(self, name:Optional[str] = None, export_dir: Optional[str] = None, variant: str = '__default__'):
        """TODO

        Args:
            name (Optional[str], optional): _description_. Defaults to None.
            export_dir (Optional[str], optional): _description_. Defaults to None.
            variant (str, optional): _description_. Defaults to '__default__'.
        """
        
        if name is None:
            name = repo_folder_name(repo_id=self.repo_id, repo_type='model')
                
            if export_dir is None:
                export_dir = os.path.join(constants.HF_HUB_CACHE, name, 'nnsight', 'exports')
                name = ""       
            
        super().export_edits(name, export_dir=export_dir, variant=variant)
        
    def import_edits(self, name:Optional[str] = None, export_dir: Optional[str] = None, variant: str = '__default__'):
        """TODO

        Args:
            name (Optional[str], optional): _description_. Defaults to None.
            export_dir (Optional[str], optional): _description_. Defaults to None.
            variant (str, optional): _description_. Defaults to '__default__'.
        """
        
        if name is None:
            name = repo_folder_name(repo_id=self.repo_id, repo_type='model')
                
            if export_dir is None:
                export_dir = os.path.join(constants.HF_HUB_CACHE, name, 'nnsight', 'exports')
                name = ""       
            
        super().import_edits(name, export_dir=export_dir, variant=variant)

    def __nnsight_generate__(self, *args, **kwargs):

        max_new_tokens = kwargs.get("max_new_tokens", None)

        if max_new_tokens is not None and self._interleaver is not None:
            self._interleaver.default_all = max_new_tokens

        output = self._model.generate(*args, **kwargs)

        if self._interleaver is not None:
            self._interleaver.default_all = None

        output = self.generator._module(output)

        return output

    def _load_config(self, repo_id: str, **kwargs):

        if self.config is None:

            self.config = AutoConfig.from_pretrained(repo_id, **kwargs)

    def _load_tokenizer(self, repo_id: str, **kwargs):

        if self.tokenizer is None:

            if "padding_side" not in kwargs:
                kwargs["padding_side"] = "left"

            self.tokenizer = AutoTokenizer.from_pretrained(
                repo_id, config=self.config, **kwargs
            )

            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_meta(
        self,
        repo_id: str,
        tokenizer_kwargs: Optional[Dict[str, Any]] = {},
        patch_llama_scan: bool = True,
        **kwargs,
    ) -> Module:

        self.repo_id = repo_id

        self._load_config(repo_id, **kwargs)

        self._load_tokenizer(repo_id, **tokenizer_kwargs)

        if (
            patch_llama_scan
            and isinstance(self.config, LlamaConfig)
            and isinstance(self.config.rope_scaling, dict)
            and "rope_type" in self.config.rope_scaling
        ):
            self.config.rope_scaling["rope_type"] = "default"

        model = self.automodel.from_config(self.config, trust_remote_code=True)

        return model

    def _load(
        self,
        repo_id: str,
        tokenizer_kwargs: Optional[Dict[str, Any]] = {},
        patch_llama_scan: bool = True,
        **kwargs,
    ) -> PreTrainedModel:

        self._load_config(repo_id, **kwargs)

        self._load_tokenizer(repo_id, **tokenizer_kwargs)

        if (
            patch_llama_scan
            and isinstance(self.config, LlamaConfig)
            and isinstance(self.config.rope_scaling, dict)
            and "rope_type" in self.config.rope_scaling
        ):
            self.config.rope_scaling["rope_type"] = "llama3"

        model = self.automodel.from_pretrained(repo_id, config=self.config, **kwargs)

        return model

    def _tokenize(
        self,
        inputs: Union[
            str,
            List[str],
            List[List[str]],
            List[int],
            List[List[int]],
            torch.Tensor,
            Dict[str, Any],
        ],
        **kwargs,
    ):

        if isinstance(inputs, str) or (
            isinstance(inputs, list) and isinstance(inputs[0], int)
        ):
            inputs = [inputs]

        if isinstance(inputs, torch.Tensor) and inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)

        if not isinstance(inputs[0], str):
            inputs = [{"input_ids": ids} for ids in inputs]
            return self.tokenizer.pad(inputs, return_tensors="pt", **kwargs)

        return self.tokenizer(inputs, return_tensors="pt", padding=True, **kwargs)

    def _prepare_input(
        self,
        *inputs: Tuple[
            Union[
                str,
                List[str],
                List[List[str]],
                List[int],
                List[List[int]],
                torch.Tensor,
                List[torch.Tensor],
                Dict[str, Any],
                BatchEncoding,
            ]
        ],
        input_ids: Union[
            List[int], List[List[int]], torch.Tensor, List[torch.Tensor]
        ] = None,
        labels: Any = None,
        **kwargs,
    ) -> Tuple[BatchEncoding, int]:

        if input_ids is not None:

            assert len(inputs) == 0

            inputs = (input_ids,)

        assert len(inputs) == 1

        inputs = inputs[0]

        if isinstance(inputs, dict):
            inputs = BatchEncoding(inputs)
        elif isinstance(inputs, BatchEncoding):
            pass
        else:

            inputs = self._tokenize(inputs, **kwargs)

            if labels is not None:
                labels = self._tokenize(labels, **kwargs)["input_ids"]

        return tuple(), {**inputs, "labels": labels}

    def _batch(
        self,
        batched_inputs: Optional[Tuple[Tuple[BatchEncoding], Dict[str, Any]]],
        **prepared_kwargs,
    ) -> Tuple[Dict[str, Any]]:

        if batched_inputs is None:
            return (tuple(), prepared_kwargs), len(prepared_kwargs["input_ids"])

        batched_inputs = batched_inputs[1]

        batched_labels = batched_inputs["labels"]

        attention_mask = batched_inputs["attention_mask"]

        batched_ids = [
            {"input_ids": ids}
            for ids in [
                *batched_inputs["input_ids"].tolist(),
                *prepared_kwargs["input_ids"].tolist(),
            ]
        ]
        new_batched_inputs = self.tokenizer.pad(batched_ids, return_tensors="pt")

        labels = prepared_kwargs.get("labels", None)

        if labels is not None:

            batched_labels = torch.cat((batched_labels, labels))

        if self.tokenizer.padding_side == "left":

            new_batched_inputs["attention_mask"][
                : attention_mask.shape[0], -attention_mask.shape[1] :
            ] = attention_mask

        else:

            new_batched_inputs["attention_mask"][
                : attention_mask.shape[0], : attention_mask.shape[1]
            ] = attention_mask

        batched_inputs.pop("input_ids", None)
        batched_inputs.pop("attention_mask", None)

        return (
            tuple(),
            {**new_batched_inputs, **batched_inputs, "labels": batched_labels},
        ), len(prepared_kwargs["input_ids"])

    def _remoteable_model_key(self) -> str:
        return json.dumps(
            {"repo_id": self.repo_id}  # , "torch_dtype": str(self._model.dtype)}
        )

    @classmethod
    def _remoteable_from_model_key(cls, model_key: str, **kwargs) -> Self:

        kwargs = {**json.loads(model_key), **kwargs}

        repo_id = kwargs.pop("repo_id")

        return LanguageModel(repo_id, **kwargs)


if TYPE_CHECKING:

    class LanguageModel(GenerationMixin, LanguageModel, PreTrainedModel):

        def generate(self, *args, **kwargs) -> Union[InterleavingTracer, Any]:
            return super().generate(*args, **kwargs)
