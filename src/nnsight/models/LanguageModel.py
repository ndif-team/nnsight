from __future__ import annotations

import json
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.models.auto import modeling_auto
from transformers.models.llama.configuration_llama import LlamaConfig
from typing_extensions import Self

from nnsight.envoy import Envoy

from ..intervention import InterventionProxy
from ..util import WrapperModule
from . import NNsight
from .mixins import GenerationMixin, RemoteableMixin


class TokenIndexer:
    """Helper class to directly access token indices of hidden states.
    Directly indexes the second dimension of tensors.
    Makes positive indices negative as tokens are padded on the left.

    Args:
        proxy (InterventionProxy): Proxy to aid in token indexing.
    """

    def __init__(self, proxy: InterventionProxy) -> None:
        self.proxy = proxy

    def convert_idx(self, idx: int):
        if idx >= 0:
            n_tokens = self.proxy.node.proxy_value.shape[1]
            idx = -(n_tokens - idx)

        return idx

    def __getitem__(self, key: int) -> LanguageModelProxy:
        key = self.convert_idx(key)

        return self.proxy[:, key]

    def __setitem__(self, key: int, value: Union[LanguageModelProxy, Any]) -> None:
        key = self.convert_idx(key)

        self.proxy[:, key] = value


class LanguageModelProxy(InterventionProxy):
    """

    Indexing by token of hidden states can easily done using ``.token[<idx>]`` or ``.t[<idx>]``

    .. code-block:: python

        with runner.invoke('The Eiffel Tower is in the city of') as invoker:
            logits = model.lm_head.output.t[0].save()

        print(logits.value)

    This would save only the first token of the output for this module.
    This should be used when using multiple invokes as the batching and padding of multiple inputs could mean the indices for tokens shifts around and this take care of that.

    Args:
        InterventionProxy (_type_): _description_

    Returns:
        _type_: _description_
    """

    @property
    def token(self) -> TokenIndexer:
        """Property used to do token based indexing on a proxy.
        Directly indexes the second dimension of tensors.
        Makes positive indices negative as tokens are padded on the left.

        Example:

            .. code-block:: python

                model.transformer.h[0].mlp.output.token[0]

            Is equivalent to:

            .. code-block:: python

                model.transformer.h[0].mlp.output.token[:,-3]

            For a proxy tensor with 3 tokens.

        Returns:
            TokenIndexer: Object to do token based indexing.
        """
        return TokenIndexer(self)

    @property
    def t(self) -> TokenIndexer:
        """Property as alias for InterventionProxy.token"""
        return self.token


class LanguageModel(GenerationMixin, RemoteableMixin, NNsight):
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

    proxy_class = LanguageModelProxy

    def __new__(cls, *args, **kwargs) -> Self | Envoy:
        return object.__new__(cls)

    def __init__(
        self,
        model_key: Union[str, torch.nn.Module],
        *args,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        automodel: Type[AutoModel] = AutoModelForCausalLM,
        **kwargs,
    ) -> None:
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self._model: PreTrainedModel = None
        self.automodel = (
            automodel
            if not isinstance(automodel, str)
            else getattr(modeling_auto, automodel)
        )

        if isinstance(model_key, torch.nn.Module):

            setattr(model_key, "generator", WrapperModule())

        super().__init__(model_key, *args, **kwargs)

    def _load(
        self,
        repo_id: str,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        patch_llama_scan: bool = True,
        **kwargs,
    ) -> PreTrainedModel:

        config = kwargs.pop("config", None) or AutoConfig.from_pretrained(
            repo_id, **kwargs
        )

        if self.tokenizer is None:
            if tokenizer_kwargs is None:
                tokenizer_kwargs = {}

            if "padding_side" not in tokenizer_kwargs:
                tokenizer_kwargs["padding_side"] = "left"

            self.tokenizer = AutoTokenizer.from_pretrained(
                repo_id, config=config, **tokenizer_kwargs
            )

            if not hasattr(self.tokenizer.pad_token, "pad_token"):
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        if self._model is None:

            if (
                patch_llama_scan
                and isinstance(config, LlamaConfig)
                and isinstance(config.rope_scaling, dict)
                and "rope_type" in config.rope_scaling
            ):
                config.rope_scaling["rope_type"] = "default"

            model = self.automodel.from_config(config, trust_remote_code=True)

            setattr(model, "generator", WrapperModule())

            return model

        if (
            patch_llama_scan
            and isinstance(config, LlamaConfig)
            and isinstance(config.rope_scaling, dict)
            and "rope_type" in config.rope_scaling
        ):
            config.rope_scaling["rope_type"] = "llama3"

        model = self.automodel.from_pretrained(repo_id, config=config, **kwargs)

        setattr(model, "generator", WrapperModule())

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
    ) -> Tuple[BatchEncoding, int]:
        if isinstance(inputs, dict):

            new_inputs = dict()

            tokenized_inputs = self._tokenize(inputs["input_ids"], **kwargs)

            new_inputs["input_ids"] = tokenized_inputs["input_ids"]

            if "attention_mask" in inputs:
                for ai, attn_mask in enumerate(inputs["attention_mask"]):
                    tokenized_inputs["attention_mask"][
                        ai, -len(attn_mask) :
                    ] = attn_mask

                new_inputs["attention_mask"] = tokenized_inputs["attention_mask"]

            if "labels" in inputs:
                labels = self._tokenize(inputs["labels"], **kwargs)

                new_inputs["labels"] = labels["input_ids"]

            return (BatchEncoding(new_inputs),), len(new_inputs["input_ids"])

        inputs = self._tokenize(inputs, **kwargs)

        if labels is not None:
            labels = self._tokenize(labels, **kwargs)

            inputs["labels"] = labels["input_ids"]

        return (inputs,), len(inputs["input_ids"])

    def _batch_inputs(
        self,
        batched_inputs: Optional[Dict[str, Any]],
        prepared_inputs: BatchEncoding,
    ) -> Tuple[Dict[str, Any]]:

        if batched_inputs is None:
            batched_inputs = {"input_ids": []}

            if "labels" in prepared_inputs:
                batched_inputs["labels"] = []

            if "attention_mask" in prepared_inputs:
                batched_inputs["attention_mask"] = []

        else:

            batched_inputs = batched_inputs[0]

        batched_inputs["input_ids"].extend(prepared_inputs["input_ids"])

        if "labels" in prepared_inputs:
            batched_inputs["labels"].extend(prepared_inputs["labels"])
        if "attention_mask" in prepared_inputs:
            batched_inputs["attention_mask"].extend(prepared_inputs["attention_mask"])

        return (batched_inputs,)

    def _execute_forward(self, prepared_inputs: Any, *args, **kwargs):

        device = next(self._model.parameters()).device

        return self._model(
            *args,
            **prepared_inputs.to(device),
            **kwargs,
        )

    def _execute_generate(
        self, prepared_inputs: Any, *args, max_new_tokens=1, **kwargs
    ):

        device = next(self._model.parameters()).device

        output = self._model.generate(
            *args,
            **prepared_inputs.to(device),
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        self._model.generator(output)

        return output

    def _remoteable_model_key(self) -> str:
        return json.dumps(
            {"repo_id": self._model_key}  # , "torch_dtype": str(self._model.dtype)}
        )

    @classmethod
    def _remoteable_from_model_key(cls, model_key: str, **kwargs) -> Self:

        kwargs = {**json.loads(model_key), **kwargs}

        repo_id = kwargs.pop("repo_id")

        return LanguageModel(repo_id, **kwargs)
