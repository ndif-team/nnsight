from __future__ import annotations


import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizer,
)

from ..intervention.envoy import Envoy
from ..util import WrapperModule
from .transformers import TransformersModel


class LanguageModel(TransformersModel):
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

    class Generator(WrapperModule):
        """Wrapper module that captures the final generation output.

        Contains a :class:`Streamer` submodule that receives tokens
        during generation. The generator output can be accessed via
        ``model.generator.output`` inside a trace, though
        ``tracer.result`` is preferred for new code.
        """

        class Streamer(WrapperModule):
            """Streamer that receives tokens during generation and passes them through as a module call."""

            def put(self, *args):
                return self(*args)

            def end(self):
                pass

        def __init__(self) -> None:

            super().__init__()

            self.streamer = LanguageModel.Generator.Streamer()

    def __init__(
        self,
        *args,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        automodel: Type[AutoModel] = AutoModelForCausalLM,
        **kwargs,
    ) -> None:

        self.tokenizer: PreTrainedTokenizer = tokenizer

        super().__init__(*args, automodel=automodel, **kwargs)

        self.generator: Envoy = LanguageModel.Generator()

    def _load_meta(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = {},
        **kwargs,
    ):

        self._load_config(repo_id, revision=revision, **kwargs)

        self._load_tokenizer(repo_id, revision=revision, **tokenizer_kwargs)

        model = super()._load_meta(repo_id, revision=revision, **kwargs)

        self._patch_generation_config(model)

        return model

    def _load(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = {},
        **kwargs,
    ):

        self._load_config(repo_id, revision=revision, **kwargs)

        self._load_tokenizer(repo_id, revision=revision, **tokenizer_kwargs)

        model = super()._load(repo_id, revision=revision, **kwargs)

        self._patch_generation_config(model)

        return model

    # Some transformer models compile on first generation. As of 0.5.0.dev7 this not not work with nnsight if fullgraph is True
    def _patch_generation_config(self, model: torch.nn.Module):

        if getattr(model, "generation_config", None) is not None:

            warnings.filterwarnings("ignore", message="The CUDA Graph is empty")

            generation_config = model.generation_config

            compile_config = getattr(generation_config, "compile_config", None)

            if compile_config is None:

                from transformers.generation.configuration_utils import CompileConfig

                compile_config = CompileConfig()

            compile_config.fullgraph = False
            compile_config.dynamic = True

            setattr(generation_config, "compile_config", compile_config)

    def __nnsight_generate__(self, *args, **kwargs):
        """Custom generation entry point used when ``.generate()`` is called as a tracing context.

        Sets up iteration tracking via ``max_new_tokens``, injects a
        streamer for token-by-token access, and wraps the final output
        through the :attr:`generator` module.
        """

        max_new_tokens = kwargs.get("max_new_tokens", None)

        if max_new_tokens is not None and self._interleaver is not None:
            self._interleaver.default_all = max_new_tokens

        streamer = kwargs.pop("streamer", self.generator.streamer._module)

        output = self._model.generate(*args, streamer=streamer, **kwargs)

        if self._interleaver is not None:
            self._interleaver.default_all = None

        output = self.generator(output, hook=True)

        return output

    def _load_tokenizer(self, repo_id: str, **kwargs):

        if self.tokenizer is None:

            if "padding_side" not in kwargs:
                kwargs["padding_side"] = "left"

            self.tokenizer = AutoTokenizer.from_pretrained(
                repo_id, config=self.config, **kwargs
            )

            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

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
        """
        Tokenizes the inputs.
        """

        if isinstance(inputs, torch.Tensor):
            if inputs.ndim == 1:
                inputs = inputs.unsqueeze(0)
            return BatchEncoding({"input_ids": inputs})

        if self.tokenizer is None:
            if self.repo_id is not None:
                self._load_tokenizer(self.repo_id, **kwargs)
            else:
                raise AttributeError(
                    "Tokenizer not found. If you passed a pre-loaded model to `LanguageModel`, you need to provide a tokenizer when initializing: `LanguageModel(model, tokenizer=tokenizer)`."
                )

        if isinstance(inputs, str) or (
            isinstance(inputs, list) and isinstance(inputs[0], int)
        ):
            inputs = [inputs]

        if not isinstance(inputs[0], str):
            inputs = [{"input_ids": ids} for ids in inputs]
            return self.tokenizer.pad(inputs, return_tensors="pt", **kwargs)

        return self.tokenizer(inputs, return_tensors="pt", padding=True, **kwargs)

    _TOKENIZER_KWARGS = {
        "text_pair",
        "text_target",
        "text_pair_target",
        "add_special_tokens",
        "padding",
        "truncation",
        "max_length",
        "stride",
        "is_split_into_words",
        "pad_to_multiple_of",
        "padding_side",
        "return_tensors",
        "return_token_type_ids",
        "return_attention_mask",
        "return_overflowing_tokens",
        "return_special_tokens_mask",
        "return_offsets_mapping",
        "return_length",
        "verbose",
    }

    def _prepare_input(
        self,
        *inputs: Union[
            str,
            List[str],
            List[List[str]],
            List[int],
            List[List[int]],
            torch.Tensor,
            List[torch.Tensor],
            Dict[str, Any],
            BatchEncoding,
        ],
        input_ids: Union[
            List[int], List[List[int]], torch.Tensor, List[torch.Tensor]
        ] = None,
        labels: Any = None,
        attention_mask: Any = None,
        **kwargs,
    ) -> tuple[tuple[Any], dict[str, Any], int]:
        """Normalize user inputs into a ``(args, kwargs, batch_size)`` tuple.

        Handles tokenization of strings, tensor reshaping, and
        separation of tokenizer-specific kwargs from model kwargs.

        Returns:
            Tuple of ``(args, kwargs, batch_size)`` ready for the
            model's forward pass or for :meth:`_batch`.
        """

        tokenizer_kwargs = {}
        remaining_kwargs = {}
        for k, v in kwargs.items():
            if k in self._TOKENIZER_KWARGS:
                tokenizer_kwargs[k] = v
            else:
                remaining_kwargs[k] = v

        if input_ids is not None:
            assert len(inputs) == 0
            inputs = (input_ids,)

        if len(inputs) == 0:
            return tuple(), remaining_kwargs, 0

        assert len(inputs) == 1
        inputs = inputs[0]

        if isinstance(inputs, dict):
            inputs = BatchEncoding(inputs)
        elif isinstance(inputs, BatchEncoding):
            pass
        else:

            inputs = self._tokenize(inputs, **tokenizer_kwargs)

            if labels is not None:
                labels = self._tokenize(labels, **tokenizer_kwargs)["input_ids"]

        if attention_mask is not None:
            inputs["attention_mask"] = attention_mask

        return (
            tuple(),
            {**inputs, "labels": labels, **remaining_kwargs},
            len(inputs["input_ids"]),
        )

    def _batch(
        self,
        batched_inputs: Optional[Tuple[Tuple[BatchEncoding], Dict[str, Any]]],
        **prepared_kwargs,
    ) -> tuple[tuple[Any], dict[str, Any]]:
        """Combine multiple invokes' prepared inputs into a single padded batch.

        Re-pads token sequences so they share a common length and
        preserves attention masks from earlier invokes.

        Returns:
            Tuple of ``(args, kwargs)`` representing the combined batch.
        """
        batched_inputs = batched_inputs[1].copy()

        if "input_ids" not in batched_inputs:
            return tuple(), {**prepared_kwargs, **batched_inputs}

        batched_labels = batched_inputs["labels"]

        attention_mask = batched_inputs.get("attention_mask", torch.ones_like(batched_inputs["input_ids"]))

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

        new_attention_mask = prepared_kwargs.get("attention_mask", None)
        n_old = attention_mask.shape[0] if attention_mask is not None else 0
        left = self.tokenizer.padding_side == "left"

        combined_mask = torch.zeros_like(new_batched_inputs["attention_mask"])

        for row_start, mask in [(0, attention_mask), (n_old, new_attention_mask)]:
            if mask is not None:
                if left:
                    combined_mask[row_start : row_start + mask.shape[0], -mask.shape[1] :] = mask
                else:
                    combined_mask[row_start : row_start + mask.shape[0], : mask.shape[1]] = mask

        new_batched_inputs["attention_mask"] = combined_mask

        batched_inputs.pop("input_ids", None)
        batched_inputs.pop("attention_mask", None)

        return (
            tuple(),
            {**new_batched_inputs, **batched_inputs, "labels": batched_labels},
        )

    def _remoteable_model_key(self) -> str:
        return super()._remoteable_model_key()

    def _remoteable_persistent_objects(self) -> dict:
        persistent_objects = super()._remoteable_persistent_objects()
        persistent_objects["Tokenizer"] = self.tokenizer
        return persistent_objects

    def __getstate__(self):
        state = super().__getstate__()
        self.tokenizer._persistent_id = "Tokenizer"
        state["tokenizer"] = self.tokenizer
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.tokenizer = state["tokenizer"]
