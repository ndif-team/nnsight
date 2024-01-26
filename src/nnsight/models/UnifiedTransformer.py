from __future__ import annotations

from typing import Any, Dict, List, Union

import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BatchEncoding, PretrainedConfig, PreTrainedModel,
                          PreTrainedTokenizer)
from transformers.models.auto import modeling_auto

from .NNsightModel import NNsightModel


class UnifiedTransformer(NNsightModel):
    """UnifiedTransformer is an nnsight wrapper around TransformerLens's HookedTransformer.

    Inputs can be in the form of:
        Prompt: (str)
        Prompts: (List[str])
        Batched prompts: (List[List[str]])
        Tokenized prompt: (Union[List[int], torch.Tensor])
        Tokenized prompts: (Union[List[List[int]], torch.Tensor])
        Direct input: (Dict[str,Any])

    If using a custom model, you also need to provide the tokenizer like ``UnifiedTransformer(custom_model, tokenizer=tokenizer)``

    Calls to generate pass arguments downstream to :func:`GenerationMixin.generate`

    Attributes:
        config (HookedTransformerConfig): Huggingface config file loaded from repository or checkpoint.
        tokenizer (PreTrainedTokenizer): Tokenizer for LMs.
        meta_model (PreTrainedModel): Meta version of underlying auto model.
        local_model (HookedTransformer): Local version of underlying HookedTransformer.

    """

    def __init__(
        self, 
        *args, 
        device: str, 
        **kwargs
    ) -> None:
        self.meta_model: PreTrainedModel = None
        self.local_model: PreTrainedModel = None

        super().__init__(*args, **kwargs)

        self.tokenizer = self.local_model.tokenizer
        self.config = self.local_model.cfg
        self.local_model.device = device

    def _load_meta(self, repoid_or_path, *args, **kwargs) -> PreTrainedModel:
        raise NotImplementedError("This subclass does not implement this method.")

    def _load_local(self, repoid_or_path, *args, **kwargs) -> PreTrainedModel:
        raise NotImplementedError("This subclass does not implement this method.")

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
        **kwargs,
    ) -> BatchEncoding:
        if isinstance(inputs, dict):

            new_inputs = dict()

            tokenized_inputs = self._tokenize(inputs["input"], **kwargs)

            new_inputs['input'] = tokenized_inputs['input_ids']

            if "attention_mask" in inputs:
                for ai, attn_mask in enumerate(inputs["attention_mask"]):
                    tokenized_inputs["attention_mask"][ai, -len(attn_mask) :] = attn_mask

                new_inputs["attention_mask"] = tokenized_inputs["attention_mask"]

            return BatchEncoding(new_inputs)

        inputs = self._tokenize(inputs, **kwargs)
        
        if "input_ids" in inputs:
            inputs["input"] = inputs.pop("input_ids")

        return inputs

    def _batch_inputs(
        self, prepared_inputs: BatchEncoding, batched_inputs: Dict
    ) -> torch.Tensor:
        if batched_inputs is None:
            batched_inputs = {"input": []}

            if "attention_mask" in prepared_inputs:
                batched_inputs["attention_mask"] = []

        batched_inputs["input"].extend(prepared_inputs["input"])

        if "attention_mask" in prepared_inputs:
            batched_inputs["attention_mask"].extend(prepared_inputs["attention_mask"])

        return batched_inputs, len(prepared_inputs["input"])

    def _example_input(self) -> Dict[str, torch.Tensor]:
        return BatchEncoding(
            {"input": torch.tensor([[0]])}
        )

    def _generation(
        self, prepared_inputs, *args, max_new_tokens: int = 1, **kwargs
    ) -> Any:

        # HookedTransformer uses attention_mask in forward, but not generate
        if "attention_mask" in prepared_inputs:
            prepared_inputs.pop("attention_mask")

        return super()._generation(
            prepared_inputs, *args, max_new_tokens=max_new_tokens, **kwargs
        )
