"""
Gap 2.2: q_proj, k_proj, v_proj do not exist -- merged into qkv_proj

DOCUMENTED PATTERN (CLAUDE.md "Accessing Outputs" implies per-submodule access):
    q = model.model.layers[0].self_attn.q_proj.output.save()
    k = model.model.layers[0].self_attn.k_proj.output.save()
    v = model.model.layers[0].self_attn.v_proj.output.save()
Users expect to access individual Q, K, V projections for attention head analysis,
key-value editing, or selective ablation.

ON HF (expected behavior):
    self_attn has separate q_proj, k_proj, v_proj (each a Linear module) and
    o_proj. Each projection is independently hookable. Researchers can read query
    vectors for attention pattern analysis, edit key/value representations for
    factual recall experiments, or ablate individual projections.

ON vLLM (the gap):
    q_proj, k_proj, and v_proj DO NOT EXIST. They are merged into a single
    qkv_proj (QKVParallelLinear) whose output is a concatenated tensor of all
    three projections. o_proj is a RowParallelLinear (returns tuple, see Gap 2.3).
    Accessing model.model.layers[0].self_attn.q_proj raises AttributeError.

WHY THIS MATTERS:
    Attention head analysis, knowledge editing (modifying keys/values to change
    factual associations), induction head detection, and Q/K/V-specific SAE
    training all require separate access to individual projections. The merged
    module forces users to manually split the concatenated tensor and know the
    exact head dimensions, which is error-prone and model-specific.

VALIDATION: Check for existence of q_proj, k_proj, v_proj, and qkv_proj
attributes on the attention module.
"""

import argparse
import json

import torch


def run_vllm(model_name, prompt):
    from nnsight.modeling.vllm import VLLM

    model = VLLM(model_name, gpu_memory_utilization=0.05, dispatch=True)

    print("vLLM Attention structure:")
    vllm_attn = model.model.layers[0].self_attn
    print(f"  {vllm_attn}")

    vllm_has_q = hasattr(vllm_attn, "q_proj")
    vllm_has_k = hasattr(vllm_attn, "k_proj")
    vllm_has_v = hasattr(vllm_attn, "v_proj")
    vllm_has_qkv = hasattr(vllm_attn, "qkv_proj")
    vllm_has_o = hasattr(vllm_attn, "o_proj")

    print(f"  q_proj: {vllm_has_q}")
    print(f"  k_proj: {vllm_has_k}")
    print(f"  v_proj: {vllm_has_v}")
    print(f"  qkv_proj: {vllm_has_qkv}")
    print(f"  o_proj: {vllm_has_o}")

    with model.trace(prompt, temperature=0.0):
        if vllm_has_qkv:
            qkv_out = model.model.layers[0].self_attn.qkv_proj.output.save()
        o_out = model.model.layers[0].self_attn.o_proj.output.save()

    if vllm_has_qkv:
        if isinstance(qkv_out, tuple):
            print(f"\n  qkv_proj.output: tuple, [0].shape={qkv_out[0].shape}")
        else:
            print(f"\n  qkv_proj.output: shape={qkv_out.shape}")

    if isinstance(o_out, tuple):
        print(f"  o_proj.output: tuple, [0].shape={o_out[0].shape}")
    else:
        print(f"  o_proj.output: shape={o_out.shape}")

    gap_confirmed = (not vllm_has_q) and vllm_has_qkv
    status = "CONFIRMED" if gap_confirmed else "NOT_REPRODUCED"
    detail = (
        f"vLLM: q_proj={vllm_has_q}, k_proj={vllm_has_k}, v_proj={vllm_has_v}, "
        f"qkv_proj={vllm_has_qkv}"
    )
    return {
        "backend": "vllm",
        "status": status,
        "detail": detail,
        "has_q_proj": vllm_has_q,
        "has_k_proj": vllm_has_k,
        "has_v_proj": vllm_has_v,
        "has_qkv_proj": vllm_has_qkv,
    }


def run_hf(model_name, prompt):
    from nnsight import LanguageModel

    model = LanguageModel(model_name, device_map="cuda", dispatch=True)

    print("HF Attention structure:")
    hf_attn = model.model.layers[0].self_attn
    print(f"  {hf_attn}")

    hf_has_q = hasattr(hf_attn, "q_proj")
    hf_has_k = hasattr(hf_attn, "k_proj")
    hf_has_v = hasattr(hf_attn, "v_proj")
    hf_has_qkv = hasattr(hf_attn, "qkv_proj")
    hf_has_o = hasattr(hf_attn, "o_proj")

    print(f"  q_proj: {hf_has_q}")
    print(f"  k_proj: {hf_has_k}")
    print(f"  v_proj: {hf_has_v}")
    print(f"  qkv_proj: {hf_has_qkv}")
    print(f"  o_proj: {hf_has_o}")

    # HF: separate q/k/v projections, no merged qkv_proj
    status = "NO_GAP" if hf_has_q and hf_has_k and hf_has_v and not hf_has_qkv else "UNEXPECTED"
    detail = (
        f"HF: q_proj={hf_has_q}, k_proj={hf_has_k}, v_proj={hf_has_v}, "
        f"qkv_proj={hf_has_qkv}"
    )
    return {
        "backend": "hf",
        "status": status,
        "detail": detail,
        "has_q_proj": hf_has_q,
        "has_k_proj": hf_has_k,
        "has_v_proj": hf_has_v,
        "has_qkv_proj": hf_has_qkv,
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
