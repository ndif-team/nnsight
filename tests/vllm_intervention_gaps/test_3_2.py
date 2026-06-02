"""
Gap 3.2: Attention weights are inaccessible -- PagedAttention is a fused CUDA kernel

DOCUMENTED PATTERN (CLAUDE.md "Attention Pattern Extraction"):
    attn_weights = model.transformer.h[0].attn.source.attention_interface_0.output[0].save()
The docs show extracting attention weights through source tracing into the
attention computation, enabling attention visualization and analysis.

ON HF (expected behavior):
    self_attn.output is a tuple (attn_output, attn_weights_or_none). When
    output_attentions=True, attention weights are accessible as a [batch, heads,
    seq, seq] tensor. Source tracing into the attention module reveals the SDPA
    call and its inputs/outputs. Attention pattern visualization, head pruning,
    and attention knockout experiments all work.

ON vLLM (the gap):
    PagedAttention is a fused CUDA kernel that computes attention entirely in
    C/CUDA -- no Python-level attention weight tensor is ever produced.
    self_attn.output is a single tensor (just the attention output, no weights).
    Source tracing into the attention module shows only a delegate call to the
    C kernel. There is no attention weight matrix to extract at any level.

WHY THIS MATTERS:
    Attention weight analysis is foundational to interpretability: attention
    visualization, attention head importance scoring, induction head detection,
    attention knockout, and attention-based feature attribution all require the
    [heads, seq, seq] weight matrix. With PagedAttention, none of these
    workflows are possible. This is a fundamental architectural limitation,
    not just an interface mismatch.

VALIDATION: Check whether self_attn.output contains attention weights (HF tuple
with optional weights) or is a single tensor (vLLM, no weights produced).
"""

import argparse
import json

import torch


def run_vllm(model_name, prompt):
    from nnsight.modeling.vllm import VLLM

    model = VLLM(model_name, gpu_memory_utilization=0.05, dispatch=True)

    with model.trace(prompt, temperature=0.0):
        vllm_attn_out = model.model.layers[0].self_attn.output.save()

    print("vLLM self_attn.output:")
    vllm_is_tuple = isinstance(vllm_attn_out, tuple)
    if vllm_is_tuple:
        print(f"  IS TUPLE, length={len(vllm_attn_out)}")
        for i, t in enumerate(vllm_attn_out):
            if isinstance(t, torch.Tensor):
                print(f"  [{i}]: shape={t.shape}, dtype={t.dtype}")
            elif t is None:
                print(f"  [{i}]: None")
            else:
                print(f"  [{i}]: type={type(t)}")
    elif isinstance(vllm_attn_out, torch.Tensor):
        print(f"  IS TENSOR, shape={vllm_attn_out.shape}")
    else:
        print(f"  type={type(vllm_attn_out)}")

    vllm_has_attn = hasattr(model.model.layers[0].self_attn, "attn")
    print(f"\n  has self_attn.attn (PagedAttention): {vllm_has_attn}")
    if vllm_has_attn:
        print(f"  self_attn.attn type: {type(model.model.layers[0].self_attn.attn._module)}")

    if vllm_is_tuple:
        output_desc = f"tuple len={len(vllm_attn_out)}"
    elif isinstance(vllm_attn_out, torch.Tensor):
        output_desc = f"tensor shape={list(vllm_attn_out.shape)}"
    else:
        output_desc = f"type={type(vllm_attn_out)}"

    # Gap is confirmed when output is a single tensor (no attention weights).
    # If it's a tuple, the gap is not reproduced (weights might be present).
    is_single_tensor = isinstance(vllm_attn_out, torch.Tensor)
    status = "CONFIRMED" if is_single_tensor else "NOT_REPRODUCED"
    detail = (
        f"vLLM attn output: {output_desc}; "
        f"PagedAttention fuses attention — no weights accessible"
    )
    return {
        "backend": "vllm",
        "status": status,
        "detail": detail,
        "is_tuple": vllm_is_tuple,
        "output_desc": output_desc,
    }


def run_hf(model_name, prompt):
    from nnsight import LanguageModel

    model = LanguageModel(model_name, device_map="cuda", dispatch=True)

    with model.trace(prompt):
        hf_attn_out = model.model.layers[0].self_attn.output.save()

    print("HF self_attn.output:")
    hf_is_tuple = isinstance(hf_attn_out, tuple)
    if hf_is_tuple:
        print(f"  IS TUPLE, length={len(hf_attn_out)}")
        for i, t in enumerate(hf_attn_out):
            if isinstance(t, torch.Tensor):
                print(f"  [{i}]: shape={t.shape}, dtype={t.dtype}")
            elif t is None:
                print(f"  [{i}]: None")
            else:
                print(f"  [{i}]: type={type(t)}")
    elif isinstance(hf_attn_out, torch.Tensor):
        print(f"  IS TENSOR, shape={hf_attn_out.shape}")
    else:
        print(f"  type={type(hf_attn_out)}")

    if hf_is_tuple:
        output_desc = f"tuple len={len(hf_attn_out)}"
    elif isinstance(hf_attn_out, torch.Tensor):
        output_desc = f"tensor shape={list(hf_attn_out.shape)}"
    else:
        output_desc = f"type={type(hf_attn_out)}"

    # HF: self_attn.output should be a tuple (attn_output, attn_weights_or_none)
    status = "NO_GAP" if hf_is_tuple else "UNEXPECTED"
    detail = f"HF attn output: {output_desc}"
    return {
        "backend": "hf",
        "status": status,
        "detail": detail,
        "is_tuple": hf_is_tuple,
        "output_desc": output_desc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--backend", choices=["vllm", "hf"], required=True)
    args = parser.parse_args()

    prompt = "The Eiffel Tower is in"

    if args.backend == "vllm":
        result = run_vllm(args.model, prompt)
    else:
        result = run_hf(args.model, prompt)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
