from __future__ import annotations

import collections
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from torch.utils.hooks import RemovableHandle
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BatchEncoding, PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizer)
from transformers.models.auto import modeling_auto
from .AbstractModel import AbstractModel


class LanguageModel(AbstractModel):
    """LanguageModels are nnsight wrappers around transformer auto models.

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
        automodel (type): AutoModel type from transformer auto models.
        meta_model (PreTrainedModel): Meta version of underlying auto model.
        local_model (PreTrainedModel): Local version of underlying auto model.

    """

    def __init__(self, *args, tokenizer=None, automodel=AutoModelForCausalLM, **kwargs) -> None:
        self.config: PretrainedConfig = None
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.meta_model: PreTrainedModel = None
        self.local_model: PreTrainedModel = None
        self.automodel = automodel if not isinstance(automodel, str) else getattr(modeling_auto, automodel)

        super().__init__(*args, **kwargs)

    def _register_increment_hook(self, hook: Callable) -> RemovableHandle:
        return self.local_model.register_forward_hook(hook)

    def _load_meta(self, repoid_or_path, *args, **kwargs) -> PreTrainedModel:
        self.config = AutoConfig.from_pretrained(repoid_or_path, *args, **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            repoid_or_path, config=self.config, padding_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.automodel.from_config(self.config, trust_remote_code=True)

    def _load_local(self, repoid_or_path, *args, **kwargs) -> PreTrainedModel:
        return self.automodel.from_pretrained(
            repoid_or_path, *args, config=self.config, **kwargs
        )

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
    ):
        if isinstance(inputs, BatchEncoding):
            return inputs

        if isinstance(inputs, str) or (
            isinstance(inputs, list) and isinstance(inputs[0], int)
        ):
            inputs = [inputs]

        if isinstance(inputs, torch.Tensor) and inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)

        if not isinstance(inputs[0], str):
            inputs = [{"input_ids": ids} for ids in inputs]

            return self.tokenizer.pad(inputs, return_tensors="pt")

        return self.tokenizer(inputs, return_tensors="pt", padding=True)

    def _prepare_inputs(
        self,
        inputs: Union[
            str,
            List[str],
            List[List[str]],
            List[int],
            List[List[int]],
            torch.Tensor,
            Dict[str, Any],
            BatchEncoding,
        ],
        labels: Any = None,
        **kwargs,
    ) -> BatchEncoding:
        if isinstance(inputs, dict):
            _inputs = self._tokenize(inputs["input_ids"])

            _inputs = self._tokenize(_inputs)

            if "labels" in inputs:
                labels = self._tokenize(inputs["labels"])
                labels = self._tokenize(labels)
                _inputs["labels"] = labels["input_ids"]

            return _inputs
        
        inputs = self._tokenize(inputs)

        if labels is not None:
            labels = self._tokenize(labels)

            inputs["labels"] = labels["input_ids"]

        return inputs

    def _batch_inputs(
        self, prepared_inputs: BatchEncoding, batched_inputs: Dict
    ) -> torch.Tensor:
        if batched_inputs is None:
            batched_inputs = {"input_ids": []}

            if "labels" in prepared_inputs:
                batched_inputs["labels"] = []

        batched_inputs["input_ids"].extend(prepared_inputs["input_ids"])

        if "labels" in prepared_inputs:
            batched_inputs["labels"].extend(prepared_inputs["labels"])

        return batched_inputs, len(prepared_inputs["input_ids"])

    def _example_input(self) -> Dict[str, torch.Tensor]:
        return BatchEncoding({"input_ids": torch.tensor([[0]]), "labels": torch.tensor([[0]])})

    def _scan(self, prepared_inputs, *args, **kwargs) -> None:
        # TODO
        # Actually use args and kwargs. Dont do this now because the args may be specific to _generation which throws unused args errors
        # Maybe inspect signature and filter out unused args.
        self.meta_model(**prepared_inputs.copy().to("meta"))

    def _forward(self, prepared_inputs, *args, **kwargs) -> Any:
        return self.local_model(
            *args, **prepared_inputs.to(self.local_model.device), **kwargs
        )

    def _generation(
        self, prepared_inputs, *args, max_new_tokens: int = 1, **kwargs
    ) -> Any:
        return self.local_model.generate(
            *args,
            **prepared_inputs.to(self.local_model.device),
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
