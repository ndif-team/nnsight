"""HF checkpoint -> Megatron-Core weight conversion.

Layouts verified against megatron-core==0.16.1 source:
- linear_qkv fuses per query group [q_g1..q_gN, k_g, v_g]
  (megatron/core/transformer/attention.py, get_query_key_value_tensors).
- linear_fc1 is [gate; up] with gate rows first
  (fused_bias_swiglu.py: chunk(2)[0] is silu-activated).
"""

import glob
import os
from typing import Any, Callable, Dict

import torch
import torch.nn.functional as F

CONVERTERS: Dict[str, Callable] = {}


def register(model_type: str):
    def deco(fn):
        CONVERTERS[model_type] = fn
        return fn

    return deco


def mcore_config_from_hf(hf_cfg: Any, dtype: torch.dtype):
    """Derive a TransformerConfig for the local (unfused, TE/apex-free) spec."""

    from megatron.core.transformer import TransformerConfig

    return TransformerConfig(
        num_layers=hf_cfg.num_hidden_layers,
        hidden_size=hf_cfg.hidden_size,
        num_attention_heads=hf_cfg.num_attention_heads,
        num_query_groups=hf_cfg.num_key_value_heads,
        ffn_hidden_size=hf_cfg.intermediate_size,
        gated_linear_unit=True,
        activation_func=F.silu,
        normalization="RMSNorm",
        layernorm_epsilon=hf_cfg.rms_norm_eps,
        add_bias_linear=False,
        add_qkv_bias=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_softmax_in_fp32=True,
        masked_softmax_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        gradient_accumulation_fusion=False,
        params_dtype=dtype,
        perform_initialization=False,
        use_cpu_initialization=True,
    )


def load_hf_state(repo_id: str, revision: str = None) -> Dict[str, torch.Tensor]:
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = snapshot_download(
        repo_id, revision=revision, allow_patterns=["*.safetensors"]
    )
    state = {}
    for file in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(file, framework="pt") as st:
            for key in st.keys():
                state[key] = st.get_tensor(key)
    return state


def merge_qkv_weight(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ng: int, hn: int
) -> torch.Tensor:
    """[np*hn, h], [ng*hn, h], [ng*hn, h] -> fused [(np + 2*ng)*hn, h], grouped per query group."""

    h = q.shape[1]
    q = q.reshape(ng, -1, hn, h)
    k = k.reshape(ng, 1, hn, h)
    v = v.reshape(ng, 1, hn, h)
    return torch.cat([q, k, v], dim=1).reshape(-1, h)


def merge_qkv_bias(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ng: int, hn: int
) -> torch.Tensor:
    q = q.reshape(ng, -1, hn)
    k = k.reshape(ng, 1, hn)
    v = v.reshape(ng, 1, hn)
    return torch.cat([q, k, v], dim=1).reshape(-1)


@register("qwen2")
def convert_qwen2(gpt: torch.nn.Module, hf: Dict[str, torch.Tensor], hf_cfg: Any, dtype: torch.dtype):
    """Load a HF Qwen2ForCausalLM state dict into an mcore GPTModel (local spec), strictly."""

    ng = hf_cfg.num_key_value_heads
    hn = hf_cfg.hidden_size // hf_cfg.num_attention_heads

    out = {"embedding.word_embeddings.weight": hf.pop("model.embed_tokens.weight")}

    for i in range(hf_cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        m = f"decoder.layers.{i}."
        out[m + "input_layernorm.weight"] = hf.pop(p + "input_layernorm.weight")
        out[m + "self_attention.linear_qkv.weight"] = merge_qkv_weight(
            hf.pop(p + "self_attn.q_proj.weight"),
            hf.pop(p + "self_attn.k_proj.weight"),
            hf.pop(p + "self_attn.v_proj.weight"),
            ng, hn,
        )
        out[m + "self_attention.linear_qkv.bias"] = merge_qkv_bias(
            hf.pop(p + "self_attn.q_proj.bias"),
            hf.pop(p + "self_attn.k_proj.bias"),
            hf.pop(p + "self_attn.v_proj.bias"),
            ng, hn,
        )
        out[m + "self_attention.linear_proj.weight"] = hf.pop(p + "self_attn.o_proj.weight")
        out[m + "pre_mlp_layernorm.weight"] = hf.pop(p + "post_attention_layernorm.weight")
        out[m + "mlp.linear_fc1.weight"] = torch.cat(
            [hf.pop(p + "mlp.gate_proj.weight"), hf.pop(p + "mlp.up_proj.weight")], dim=0
        )
        out[m + "mlp.linear_fc2.weight"] = hf.pop(p + "mlp.down_proj.weight")

    out["decoder.final_layernorm.weight"] = hf.pop("model.norm.weight")

    # Tied checkpoints may or may not materialize lm_head.weight; mcore reuses
    # the embedding weight (share_embeddings_and_output_weights=True), so it is
    # dropped either way.
    hf.pop("lm_head.weight", None)

    if hf:
        raise ValueError(f"Unconsumed HF tensors (mapping drift?): {sorted(hf)[:8]}")

    params = dict(gpt.named_parameters())
    missing = set(params) - set(out)
    extra = set(out) - set(params)
    if missing or extra:
        raise ValueError(
            f"Strict load mismatch. Unfilled mcore params: {sorted(missing)[:8]}; "
            f"unmatched converted tensors: {sorted(extra)[:8]}"
        )

    with torch.no_grad():
        for name, tensor in out.items():
            if params[name].shape != tensor.shape:
                raise ValueError(f"{name}: shape {tuple(params[name].shape)} vs converted {tuple(tensor.shape)}")
            params[name].copy_(tensor.to(dtype))


def convert(gpt: torch.nn.Module, hf_cfg: Any, repo_id: str, revision: str, dtype: torch.dtype):
    model_type = hf_cfg.model_type
    if model_type not in CONVERTERS:
        raise NotImplementedError(
            f"No HF->mcore converter registered for model_type={model_type!r}. "
            f"Available: {sorted(CONVERTERS)}"
        )
    CONVERTERS[model_type](gpt, load_hf_state(repo_id, revision), hf_cfg, dtype)
