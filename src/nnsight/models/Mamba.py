from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Union

import causal_conv1d_cuda
import mamba_ssm
import selective_scan_cuda
import torch
from mamba_ssm import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from transformers import AutoTokenizer, BatchEncoding, PreTrainedModel

from ..patching import Patch, Patcher
from .LanguageModel import LanguageModel


class Mamba(LanguageModel):
    def _load_meta(
        self, repoid_or_path, *args, device=None, **kwargs
    ) -> PreTrainedModel:
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        config_data = load_config_hf(repoid_or_path)
        self.config = MambaConfig(**config_data)
        return MambaLMHeadModel(self.config, device="meta", dtype=None, **kwargs)

    def _load_local(self, repoid_or_path, *args, **kwargs) -> PreTrainedModel:
        model = MambaLMHeadModel(self.config, **kwargs)
        model.load_state_dict(load_state_dict_hf(repoid_or_path, **kwargs))
        return model

    def _example_input(self) -> Dict[str, torch.Tensor]:
        return BatchEncoding({"input_ids": torch.tensor([[0]])})

    def _scan(self, prepared_inputs, *args, **kwargs) -> None:
        def blah(hs, *args, residual=None, **kwargs):
            return hs, residual

        def blah1(hs, *args, **kwargs):
            return hs

        def blah2(hs, *args, **kwargs):
            return hs

        def blah3(conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus):
            return (
                conv1d_out,
                torch.zeros((*conv1d_out.shape, A.shape[1] * 2), device="meta"),
                conv1d_out,
            )

        with Patcher() as patcher:
            patcher.add(Patch(mamba_ssm.modules.mamba_simple, blah, "rms_norm_fn"))
            patcher.add(Patch(mamba_ssm.models.mixer_seq_simple, blah1, "rms_norm_fn"))
            patcher.add(Patch(causal_conv1d_cuda, blah2, "causal_conv1d_fwd"))
            patcher.add(Patch(selective_scan_cuda, blah3, "fwd"))

            self.meta_model(prepared_inputs.copy()["input_ids"].to("meta"))

    def _forward(self, prepared_inputs, *args, **kwargs) -> Any:
        return self.local_model(
            prepared_inputs["input_ids"].to(next(self.local_model.parameters()).device),
            *args,
            **kwargs,
        )

    def _generation(self, prepared_inputs, *args, max_length: int = 1, **kwargs) -> Any:
        return self.local_model.generate(
            prepared_inputs["input_ids"].to(next(self.local_model.parameters()).device),
            *args,
            max_length=max_length,
            **kwargs,
        )
