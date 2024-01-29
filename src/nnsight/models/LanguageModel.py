from __future__ import annotations

from typing import Any, Dict, List, Union

import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BatchEncoding, PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizer)
from transformers.models.auto import modeling_auto

from .NNsightModel import NNsightModel


class LanguageModel(NNsightModel):
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

    def __init__(
        self, *args, tokenizer=None, automodel=AutoModelForCausalLM, **kwargs
    ) -> None:
        self.config: PretrainedConfig = None
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.meta_model: PreTrainedModel = None
        self.local_model: PreTrainedModel = None
        self.automodel = (
            automodel
            if not isinstance(automodel, str)
            else getattr(modeling_auto, automodel)
        )

        super().__init__(*args, **kwargs)

    def _load_meta(self, repoid_or_path, *args, **kwargs) -> PreTrainedModel:
        self.config = AutoConfig.from_pretrained(repoid_or_path, *args, **kwargs)

        if self.tokenizer is None:

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
        **kwargs,
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
            return self.tokenizer.pad(inputs, return_tensors="pt", **kwargs)

        return self.tokenizer(inputs, return_tensors="pt", padding=True, **kwargs)

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

            new_inputs = dict()

            tokenized_inputs = self._tokenize(inputs["input_ids"], **kwargs)

            new_inputs['input_ids'] = tokenized_inputs['input_ids']

            if "attention_mask" in inputs:
                for ai, attn_mask in enumerate(inputs["attention_mask"]):
                    tokenized_inputs["attention_mask"][ai, -len(attn_mask) :] = attn_mask

                new_inputs["attention_mask"] = tokenized_inputs["attention_mask"]

            if "labels" in inputs:
                labels = self._tokenize(inputs["labels"], **kwargs)

                new_inputs["labels"] = labels["input_ids"]

            return BatchEncoding(new_inputs)

        inputs = self._tokenize(inputs, **kwargs)

        if labels is not None:
            labels = self._tokenize(labels, **kwargs)

            inputs["labels"] = labels["input_ids"]

        return inputs

    def _batch_inputs(
        self, prepared_inputs: BatchEncoding, batched_inputs: Dict
    ) -> torch.Tensor:
        if batched_inputs is None:
            batched_inputs = {"input_ids": []}

            if "labels" in prepared_inputs:
                batched_inputs["labels"] = []

            if "attention_mask" in prepared_inputs:
                batched_inputs["attention_mask"] = []

        batched_inputs["input_ids"].extend(prepared_inputs["input_ids"])

        if "labels" in prepared_inputs:
            batched_inputs["labels"].extend(prepared_inputs["labels"])
        if "attention_mask" in prepared_inputs:
            batched_inputs["attention_mask"].extend(prepared_inputs["attention_mask"])

        return batched_inputs, len(prepared_inputs["input_ids"])

    def _example_input(self) -> Dict[str, torch.Tensor]:
        return BatchEncoding(
            {"input_ids": torch.tensor([[0]]), "labels": torch.tensor([[0]])}
        )

    def _generation(
        self, prepared_inputs, *args, max_new_tokens: int = 1, **kwargs
    ) -> Any:
        return super()._generation(
            prepared_inputs, *args, max_new_tokens=max_new_tokens, **kwargs
        )
