"""
Gap 1.4: LayerNorm output is a (normalized, residual) tuple instead of a single tensor

DOCUMENTED PATTERN (CLAUDE.md "Common Patterns > Logit Lens"):
    hs = model.transformer.h[i].output[0]
    logits = model.lm_head(model.transformer.ln_f(hs))  # Expects single tensor
Users treat layernorm outputs as single tensors throughout -- applying the final
layernorm and then the unembedding head for logit lens, or reading intermediate
normalized representations for analysis.

ON HF (expected behavior):
    input_layernorm.output is a single tensor: RMSNorm(hidden_states).
    post_attention_layernorm.output is also a single tensor. Code that passes
    layernorm output directly to another module or applies arithmetic works as
    expected.

ON vLLM (the gap):
    input_layernorm.output is a 2-TUPLE: (normalized, new_residual). This is
    because vLLM uses fused_add_rms_norm which performs residual addition AND
    normalization in a single kernel call. The same applies to
    post_attention_layernorm. Code that treats the output as a single tensor --
    e.g., model.lm_head(layernorm.output) -- gets a tuple instead.

WHY THIS MATTERS:
    Logit lens and any analysis pipeline that reads layernorm outputs to understand
    the normalized representation entering attention or MLP will fail or silently
    produce wrong results. The tuple is not indexable the same way as a tensor,
    and passing it to downstream modules causes shape mismatches or type errors.

VALIDATION: Check whether input_layernorm.output and post_attention_layernorm.output
are tuples (vLLM) or single tensors (HF).
"""

import argparse
import json

import torch


def run_vllm(model_name, prompt, layer_idx):
    from nnsight.modeling.vllm import VLLM

    model = VLLM(model_name, gpu_memory_utilization=0.05, dispatch=True)

    with model.trace(prompt, temperature=0.0):
        vllm_ln_out = model.model.layers[layer_idx].input_layernorm.output.save()
        vllm_post_ln_out = model.model.layers[layer_idx].post_attention_layernorm.output.save()

    print("vLLM input_layernorm.output:")
    vllm_ln_is_tuple = isinstance(vllm_ln_out, tuple)
    if vllm_ln_is_tuple:
        print(f"  IS TUPLE, length={len(vllm_ln_out)}")
        for i, t in enumerate(vllm_ln_out):
            if isinstance(t, torch.Tensor):
                print(f"  [{i}]: shape={t.shape}, dtype={t.dtype}")
            else:
                print(f"  [{i}]: type={type(t)}")
    elif isinstance(vllm_ln_out, torch.Tensor):
        print(f"  IS TENSOR, shape={vllm_ln_out.shape}, dtype={vllm_ln_out.dtype}")
    else:
        print(f"  type={type(vllm_ln_out)}")

    print("\nvLLM post_attention_layernorm.output:")
    vllm_post_is_tuple = isinstance(vllm_post_ln_out, tuple)
    if vllm_post_is_tuple:
        print(f"  IS TUPLE, length={len(vllm_post_ln_out)}")
        for i, t in enumerate(vllm_post_ln_out):
            if isinstance(t, torch.Tensor):
                print(f"  [{i}]: shape={t.shape}, dtype={t.dtype}")
            else:
                print(f"  [{i}]: type={type(t)}")
    elif isinstance(vllm_post_ln_out, torch.Tensor):
        print(f"  IS TENSOR, shape={vllm_post_ln_out.shape}")

    status = "CONFIRMED" if vllm_ln_is_tuple else "NOT_REPRODUCED"
    detail = (
        f"vLLM input_layernorm.output is_tuple={vllm_ln_is_tuple}; "
        f"post_attention_layernorm.output is_tuple={vllm_post_is_tuple}"
    )
    return {
        "backend": "vllm",
        "status": status,
        "detail": detail,
        "ln_is_tuple": vllm_ln_is_tuple,
        "post_ln_is_tuple": vllm_post_is_tuple,
    }


def run_hf(model_name, prompt, layer_idx):
    from nnsight import LanguageModel

    model = LanguageModel(model_name, device_map="cuda", dispatch=True)

    with model.trace(prompt):
        hf_ln_out = model.model.layers[layer_idx].input_layernorm.output.save()
        hf_post_ln_out = model.model.layers[layer_idx].post_attention_layernorm.output.save()

    print("HF input_layernorm.output:")
    hf_ln_is_tuple = isinstance(hf_ln_out, tuple)
    if hf_ln_is_tuple:
        print(f"  IS TUPLE, length={len(hf_ln_out)}")
    elif isinstance(hf_ln_out, torch.Tensor):
        print(f"  IS TENSOR, shape={hf_ln_out.shape}, dtype={hf_ln_out.dtype}")
    else:
        print(f"  type={type(hf_ln_out)}")

    print("\nHF post_attention_layernorm.output:")
    hf_post_is_tuple = isinstance(hf_post_ln_out, tuple)
    if hf_post_is_tuple:
        print(f"  IS TUPLE, length={len(hf_post_ln_out)}")
    elif isinstance(hf_post_ln_out, torch.Tensor):
        print(f"  IS TENSOR, shape={hf_post_ln_out.shape}, dtype={hf_post_ln_out.dtype}")
    else:
        print(f"  type={type(hf_post_ln_out)}")

    # HF: layernorm output should be a single tensor
    ln_is_tensor = isinstance(hf_ln_out, torch.Tensor)
    status = "NO_GAP" if ln_is_tensor else "UNEXPECTED"
    detail = (
        f"HF input_layernorm.output is_tensor={ln_is_tensor}; "
        f"post_attention_layernorm.output is_tensor={isinstance(hf_post_ln_out, torch.Tensor)}"
    )
    return {
        "backend": "hf",
        "status": status,
        "detail": detail,
        "ln_is_tuple": hf_ln_is_tuple,
        "post_ln_is_tuple": hf_post_is_tuple,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--backend", choices=["vllm", "hf"], required=True)
    args = parser.parse_args()

    prompt = "The Eiffel Tower is in"
    layer_idx = 5

    if args.backend == "vllm":
        result = run_vllm(args.model, prompt, layer_idx)
    else:
        result = run_hf(args.model, prompt, layer_idx)

    print(json.dumps(result))


if __name__ == "__main__":
    main()
