from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Union

import accelerate
import causal_conv1d_cuda
import mamba_ssm
import selective_scan_cuda
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from transformers import AutoTokenizer, BatchEncoding, PreTrainedModel

from nnsight.util import WrapperModule

from ..patching import Patch, Patcher
from .LanguageModel import LanguageModel

from torch._guards import detect_fake_mode

class Mamba(LanguageModel):

    def _load(self, repo_id: str, device='cpu', **kwargs) -> PreTrainedModel:

        config = MambaConfig(**load_config_hf(repo_id))

        if self.tokenizer is None:

            self.tokenizer = AutoTokenizer.from_pretrained(
                repo_id, config=config, padding_side="left"
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self._model is None:
            
            model = MambaLMHeadModel(config, device="meta", dtype=None, **kwargs)
            
            setattr(model, 'generator', WrapperModule())

            return model

        model = MambaLMHeadModel(config, device=device, **kwargs)
        model.load_state_dict(load_state_dict_hf(repo_id, device=device, **kwargs))
        
        setattr(model, 'generator', WrapperModule())

        return model
    
    def _execute_forward(self, prepared_inputs: Any, *args, **kwargs):

        device = next(self._model.parameters()).device

        patcher = None

        with Patcher() as patcher:

            if detect_fake_mode(prepared_inputs):

                def blah(hs, *args, residual=None, **kwargs):
                    return hs, residual or torch.rand_like(hs)

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

                
                patcher.add(Patch(mamba_ssm.modules.mamba_simple, blah, "rms_norm_fn"))
                patcher.add(
                    Patch(mamba_ssm.models.mixer_seq_simple, blah1, "rms_norm_fn")
                )
                patcher.add(Patch(causal_conv1d_cuda, blah2, "causal_conv1d_fwd"))
                patcher.add(Patch(selective_scan_cuda, blah3, "fwd"))


            return self._model(
                prepared_inputs["input_ids"].to(device),
                *args,
                **kwargs,
            )

    def _execute_generate(
        self, prepared_inputs: Any, *args, max_length=1, **kwargs
    ):

        device = next(self._model.parameters()).device

        patcher = None

        with Patcher() as patcher:

            if detect_fake_mode(prepared_inputs):

                def blah(hs, *args, residual=None, **kwargs):
                    return hs, residual or torch.rand_like(hs)

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

                
                patcher.add(Patch(mamba_ssm.modules.mamba_simple, blah, "rms_norm_fn"))
                patcher.add(
                    Patch(mamba_ssm.models.mixer_seq_simple, blah1, "rms_norm_fn")
                )
                patcher.add(Patch(causal_conv1d_cuda, blah2, "causal_conv1d_fwd"))
                patcher.add(Patch(selective_scan_cuda, blah3, "fwd"))

            output = self._model.generate(
                prepared_inputs["input_ids"].to(device),
                *args,
                max_length=max_length,
                **kwargs,
            )

        self._model.generator(output)

        return output

class SSM(torch.nn.Module):
    class DiscA(torch.nn.Module):
        def forward(self, delta, A):
            return torch.exp(torch.einsum("bdl,dn->bdln", delta, A))

    class DiscB(torch.nn.Module):
        def forward(self, delta, B):
            return torch.einsum("bdl,bnl->bdln", delta, B)

    class Hx(torch.nn.Module):
        class Bx(torch.nn.Module):
            def forward(self, deltaB: torch.Tensor, x: torch.Tensor):
                return torch.einsum("bdn,bd->bdn", deltaB, x)

        class Ah(torch.nn.Module):
            def forward(self, deltaA: torch.Tensor, h: torch.Tensor):
                return deltaA * h

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

            self.bx = SSM.Hx.Bx()
            self.ah = SSM.Hx.Ah()

        def forward(
            self,
            deltaA: torch.Tensor,
            deltaB: torch.Tensor,
            x: torch.Tensor,
            h: torch.Tensor,
        ):
            return self.ah(deltaA, h) + self.bx(deltaB, x)

    class Yh(torch.nn.Module):
        def forward(self, h, C):
            y = torch.einsum("bdn,bn->bd", h, C)

            if y.is_complex():
                y = y.real * 2

            return y

    def __init__(self):
        super().__init__()

        self.discA = SSM.DiscA()
        self.discB = SSM.DiscB()
        self.hx = SSM.Hx()
        self.yh = SSM.Yh()

    def forward(self, x, delta, A, B, C, D=None, z=None, return_last_state=False):
        dtype_in = x.dtype

        x = x.float()
        delta = delta.float()

        batch, dim, dstate = x.shape[0], A.shape[0], A.shape[1]

        if A.is_complex():
            B = torch.view_as_complex(
                rearrange(B.float(), "... (L two) -> ... L two", two=2)
            )
            C = torch.view_as_complex(
                rearrange(C.float(), "... (L two) -> ... L two", two=2)
            )
        else:
            B = B.float()
            C = C.float()

        deltaA = self.discA(delta, A)

        deltaB = self.discB(delta, B)

        last_state = None

        h = A.new_zeros((batch, dim, dstate))

        ys = []

        # Main recurrence loop
        for token_idx in range(x.shape[2]):
            h = self.hx(
                deltaA[:, :, token_idx], deltaB[:, :, token_idx], x[:, :, token_idx], h
            )

            y = self.yh(h, C[:, :, token_idx])

            if token_idx == x.shape[2] - 1:
                last_state = h

            ys.append(y)

        y = torch.stack(ys, dim=2)  # (batch dim L)

        out = y if D is None else y + x * rearrange(D, "d -> d 1")

        if z is not None:
            out = out * F.silu(z)

        out = out.to(dtype=dtype_in)

        return out if not return_last_state else (out, last_state)


class MambaModuleInterp(mamba_ssm.modules.mamba_simple.Mamba):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dt = WrapperModule()
        self.B = WrapperModule()
        self.C = WrapperModule()

        self.ssm = SSM()

        self.delta_softplus = torch.nn.Softplus()

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj(hidden_states),
            "b l d -> b d l",
            l=seqlen,
        )

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        x, z = xz.chunk(2, dim=1)

        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(
                F.pad(x, (self.d_conv - x.shape[-1], 0))
            )  # Update state (B D W)

        x = self.act(self.conv1d(x)[..., :seqlen])

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_proj(dt).t()

        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        dt = self.dt(dt)
        B = self.B(B)
        C = self.C(C)

        dt = self.delta_softplus(dt)

        y = self.ssm(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)

        y = rearrange(y, "b d l -> b l d")

        out = self.out_proj(y)

        return out


class MambaInterp(Mamba):
    def __init__(self, *args, **kwargs):
        patcher = Patcher()

        patcher.add(
            Patch(
                mamba_ssm.models.mixer_seq_simple,
                MambaModuleInterp,
                "Mamba",
            )
        )

        patcher.__enter__()

        super().__init__(*args, **kwargs)
