"""Megatron-Core backend for nnsight. v0: TP=EP=PP=1, single GPU, trace-only.

Pinned against megatron-core==0.16.1. Same-process execution: the mcore model
lives in the user's process and the inherited Envoy trace path runs it, exactly
like TransformersModel. Only the loading, the HF->mcore calling convention, and
the [seq, batch, hidden] activation layout are new; the pipeline machinery of
TransformersModel is bypassed entirely (an mcore GPTModel is not a HF
PreTrainedModel and cannot ride a transformers pipeline).
"""

import os
import socket
from typing import Any, Optional

import torch

from ...intervention.batching import Batcher
from ..transformers import TransformersModel
from . import loading


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _ensure_megatron_init(seed: int = 1234):
    """Single-process megatron init. Idempotent.

    parallel_state.initialize_model_parallel asserts torch.distributed is
    initialized even at world size 1, and DotProductAttention forks the CUDA
    RNG tracker unconditionally (even at dropout 0.0), so the seed call is
    required before any forward.
    """

    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )

    model_parallel_cuda_manual_seed(seed)


class MegatronCausalLM(torch.nn.Module):
    """HF-style calling convention around a megatron.core GPTModel.

    Absorbs the kwargs the batched-input path produces (input_ids,
    attention_mask, labels) and translates the HF 2D mask (1 = keep) into
    mcore's bool [b, 1, sq, sk] mask (True = masked out), OR'd with causal.
    Returns logits [b, s, vocab].
    """

    def __init__(self, gpt: torch.nn.Module):
        super().__init__()
        self.gpt = gpt

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        b, s = input_ids.shape
        device = input_ids.device

        if position_ids is None:
            position_ids = torch.arange(s, device=device).unsqueeze(0).expand(b, s)

        causal = torch.triu(
            torch.ones(s, s, dtype=torch.bool, device=device), diagonal=1
        )
        if attention_mask is not None:
            pad = (attention_mask == 0)[:, None, None, :]
            mask4d = causal[None, None] | pad
        else:
            mask4d = causal[None, None].expand(b, 1, s, s)

        return self.gpt(
            input_ids=input_ids, position_ids=position_ids, attention_mask=mask4d
        )


class MegatronBatcher(Batcher):
    """Batcher aware of mcore's [seq, batch, hidden] decoder-internal layout.

    Wrapper-boundary tensors (input_ids, logits) are batch-first; decoder
    activations carry the batch on dim 1. Detect by matching the combined batch
    size, preferring dim 0 on ambiguity (same heuristic class as the stock
    batcher's shape[0] check).
    """

    def _dim(self, tensor: torch.Tensor) -> Optional[int]:
        if tensor.dim() >= 1 and tensor.shape[0] == self.total:
            return 0
        if tensor.dim() >= 2 and tensor.shape[1] == self.total:
            return 1
        return None

    def _narrow_tensor(self, tensor: torch.Tensor, group: list) -> torch.Tensor:
        dim = self._dim(tensor)
        if dim is None:
            return tensor
        start, size = group
        view = tensor.narrow(dim, start, size)
        # Same marker as Batcher._narrow_tensor: lets a `.backward()` grad hook
        # redirect from this out-of-graph view to its storage-owning base. The
        # redirection recovers the slice from the view's own strided geometry,
        # so it is dimension-agnostic.
        view._nnsight_batch = True
        return view

    def _widen_tensor(self, full: torch.Tensor, group: list, edited: torch.Tensor) -> torch.Tensor:
        dim = self._dim(full)
        if dim is None:
            return full
        start, size = group
        # cat (not in-place) keeps autograd correct for leaves/views and avoids
        # aliasing when `edited` is a narrowed view of `full`.
        pre = full.narrow(dim, 0, start)
        post = full.narrow(dim, start + size, full.shape[dim] - start - size)
        return torch.cat([pre, edited, post], dim=dim)


class MegatronLM(TransformersModel):
    """nnsight wrapper backing a HF repo id with a megatron.core GPTModel.

    v0 scope: single GPU, no parallelism, .trace() only (no generate/pipe).
    Envoy paths follow the mcore tree: model.gpt.decoder.layers[i].self_attention...
    """

    _batcher_class = MegatronBatcher

    def _load_meta(self, repo_id: str, *args: Any, dtype: torch.dtype = torch.float32, **kwargs: Any):
        self._load_hf_config(repo_id)
        self._load_tokenizer(repo_id)
        return MegatronCausalLM(self._build_gpt(dtype))

    def _load(self, repo_id: str, *args: Any, dtype: torch.dtype = torch.float32, **kwargs: Any):
        self._load_hf_config(repo_id)
        self._load_tokenizer(repo_id)
        gpt = self._build_gpt(dtype)
        loading.convert(gpt, self.config, repo_id, self.revision, dtype)
        return MegatronCausalLM(gpt).cuda()

    def _load_hf_config(self, repo_id: str):
        from transformers import AutoConfig

        # __dict__ write: Envoy.__setattr__ mirrors plain attribute writes onto
        # the wrapped module, and the adapter module has no use for a config.
        self.__dict__["config"] = AutoConfig.from_pretrained(
            repo_id, revision=self.revision
        )

    def _load_tokenizer(self, repo_id: str):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            # Right padding: mcore's rope path is computed from arange(seq_len)
            # and ignores position_ids, so real tokens must sit at 0..len-1.
            self.__dict__["tokenizer"] = AutoTokenizer.from_pretrained(
                repo_id, revision=self.revision, padding_side="right"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_gpt(self, dtype: torch.dtype):
        try:
            from megatron.core.models.gpt import GPTModel
            from megatron.core.models.gpt.gpt_layer_specs import (
                get_gpt_layer_local_spec,
            )
        except ImportError as e:
            raise ImportError(
                "The Megatron backend requires megatron-core: "
                "pip install megatron-core==0.16.1"
            ) from e

        _ensure_megatron_init()

        return GPTModel(
            config=loading.mcore_config_from_hf(self.config, dtype),
            transformer_layer_spec=get_gpt_layer_local_spec(normalization="RMSNorm"),
            vocab_size=self.config.vocab_size,
            max_sequence_length=self.config.max_position_embeddings,
            position_embedding_type="rope",
            # transformers 5.x: rope settings live in config.rope_parameters
            rotary_base=int(self.config.rope_parameters["rope_theta"]),
            share_embeddings_and_output_weights=self.config.tie_word_embeddings,
            parallel_output=False,
        )

    def _preprocess_invoke(self, data: Any, kwargs: dict) -> tuple:
        # Text-only override of the base: the tail of the base method tokenizes
        # through the task pipeline, which this backend does not build.
        if self._is_opaque(data, kwargs):
            return None, kwargs
        if self._is_pretokenized(data, kwargs):
            return self._encode_pretokenized(data, kwargs)
        inputs = list(data) if isinstance(data, (list, tuple)) else [data]
        rows = [dict(self.tokenizer(one, return_tensors="pt")) for one in inputs]
        return rows, kwargs

    def generate(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "The v0 Megatron backend supports .trace() only; GPTModel has no generate."
        )

    def pipe(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "The v0 Megatron backend has no task pipeline; use .trace()."
        )

    def _remoteable_model_key(self) -> str:
        raise NotImplementedError(
            "The v0 Megatron backend is local-only; remote=True is not supported."
        )
