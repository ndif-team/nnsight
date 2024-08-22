from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union, Optional

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformers import BatchEncoding, PreTrainedTokenizer

from .LanguageModel import LanguageModel


class UnifiedTransformer(LanguageModel):
    """UnifiedTransformer is an nnsight wrapper around TransformerLens's HookedTransformer.

    Inputs can be in the form of:
        Prompt: (str)
        Prompts: (List[str])
        Batched prompts: (List[List[str]])
        Tokenized prompt: (Union[List[int], torch.Tensor])
        Tokenized prompts: (Union[List[List[int]], torch.Tensor])
        Direct input: (Dict[str,Any])

    TransformerLens processing arguments can be passed as kwargs to the constructor.
    Pass `processing=False` to call `from_pretrained_no_processing` instead of `from_pretrained`.

    Calls to generate pass arguments downstream to :func:`GenerationMixin.generate`

    """

    def __init__(
        self, model: str, *args, processing: bool = True, tokenizer: Optional[PreTrainedTokenizer] = None, **kwargs
    ) -> None:
        if processing:
            hooked_model = HookedTransformer.from_pretrained(model, *args, tokenizer=tokenizer, **kwargs)
        else:
            hooked_model = HookedTransformer.from_pretrained_no_processing(
                model, *args, tokenizer=tokenizer, **kwargs
            )

        self.tokenizer = hooked_model.tokenizer
        self._model: HookedTransformer = None

        super().__init__(hooked_model, tokenizer=self.tokenizer, *args, **kwargs)

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
    ) -> Tuple[BatchEncoding, int]:
        if isinstance(inputs, dict):

            new_inputs = dict()

            tokenized_inputs = self._tokenize(inputs["input"], **kwargs)

            new_inputs["input"] = tokenized_inputs["input_ids"]

            if "attention_mask" in inputs:
                for ai, attn_mask in enumerate(inputs["attention_mask"]):
                    tokenized_inputs["attention_mask"][
                        ai, -len(attn_mask) :
                    ] = attn_mask

                new_inputs["attention_mask"] = tokenized_inputs["attention_mask"]

            return (BatchEncoding(new_inputs),), len(new_inputs["input"])

        inputs = self._tokenize(inputs, **kwargs)

        if "input_ids" in inputs:
            inputs["input"] = inputs.pop("input_ids")

        return (inputs,), len(inputs["input"])

    def _batch_inputs(
        self,
        batched_inputs: Dict,
        prepared_inputs: BatchEncoding,
    ) -> torch.Tensor:
        if batched_inputs is None:
            batched_inputs = {"input": []}

            if "attention_mask" in prepared_inputs:
                batched_inputs["attention_mask"] = []
        
        else:
            batched_inputs = batched_inputs[0]

        batched_inputs["input"].extend(prepared_inputs["input"])

        if "attention_mask" in prepared_inputs:
            batched_inputs["attention_mask"].extend(prepared_inputs["attention_mask"])

        return (batched_inputs,)

    def _execute_forward(self, prepared_inputs, *args, **kwargs) -> Any:

        # HookedTransformer uses attention_mask in forward but not in generate.
        if "attention_mask" in prepared_inputs:
            prepared_inputs.pop("attention_mask")

        return super()._execute_forward(prepared_inputs, *args, **kwargs)

    def _execute_generate(self, prepared_inputs, *args, **kwargs) -> Any:

        # HookedTransformer uses attention_mask in forward but not in generate.
        if "attention_mask" in prepared_inputs:
            prepared_inputs.pop("attention_mask")

        return super()._execute_generate(prepared_inputs, *args, **kwargs)
