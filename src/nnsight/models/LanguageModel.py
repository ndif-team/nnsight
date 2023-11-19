from __future__ import annotations

import collections
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from torch.utils.hooks import RemovableHandle
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BatchEncoding, PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizer)

from .AbstractModel import AbstractModel


class LanguageModel(AbstractModel):
    """LanguageModels are nnsight wrappers around AutoModelForCausalLM models.

    Inputs can be in the form of:
        Prompt: (str)
        Prompts: (List[str])
        Batched prompts: (List[List[str]])
        Tokenized prompt: (Union[List[int], torch.Tensor])
        Tokenized prompts: (Union[List[List[int]], torch.Tensor])
        Direct input: (Dict[str,Any])

    If using a custom model, you also need to provide the tokenizer like ``LanguageModel(custom_model, tokenizer=tokenizer)``

    Calls to generate pass arguments downstream to :func:`AutoModelForCausalLM.generate`

    Attributes:
        config (PretrainedConfig): Huggingface config file loaded from repository or checkpoint.
        tokenizer (PreTrainedTokenizer): Tokenizer for LMs.
        meta_model (PreTrainedModel): Meta version of underlying AutoModelForCausalLM model.
        local_model (PreTrainedModel): Local version of underlying AutoModelForCausalLM model.

    """

    def __init__(self, *args, tokenizer=None, **kwargs) -> None:
        self.config: PretrainedConfig = None
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.meta_model: PreTrainedModel = None
        self.local_model: PreTrainedModel = None

        super().__init__(*args, **kwargs)

    def _register_increment_hook(self, hook: Callable) -> RemovableHandle:
        return self.local_model.register_forward_hook(hook)

    def _load_meta(self, repoid_or_path, *args, **kwargs) -> PreTrainedModel:
        self.config = AutoConfig.from_pretrained(repoid_or_path, *args, **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            repoid_or_path, config=self.config, padding_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)

    def _load_local(self, repoid_or_path, *args, **kwargs) -> PreTrainedModel:
        return AutoModelForCausalLM.from_pretrained(
            repoid_or_path, *args, config=self.config, **kwargs
        )

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
        ],
    ) -> BatchEncoding:
        if isinstance(inputs, collections.abc.Mapping):
            return BatchEncoding(inputs)

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

    def _batched_inputs(self, prepared_inputs: BatchEncoding) -> torch.Tensor:
        return prepared_inputs["input_ids"]

    def _example_input(self) -> Dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[0]])}

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
