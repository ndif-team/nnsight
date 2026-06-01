"""
Gap 2.3: down_proj.output and o_proj.output are (tensor, bias) tuples instead of tensors

DOCUMENTED PATTERN (CLAUDE.md "Accessing Outputs" and "Modifying Activations"):
    mlp_out = model.model.layers[0].mlp.down_proj.output.save()
    # User then does: mlp_out * 2, or uses it for analysis
Users expect .output from a linear layer to be a single tensor, consistent with
nn.Linear behavior and all HF models.

ON HF (expected behavior):
    down_proj.output and o_proj.output are single tensors (nn.Linear returns a
    tensor). Code like `down_proj.output * 2` or `down_proj.output[:] = 0` works
    directly.

ON vLLM (the gap):
    vLLM's RowParallelLinear.forward() returns (output, output_bias) -- a 2-tuple.
    Both mlp.down_proj and self_attn.o_proj use RowParallelLinear. Code that
    treats .output as a tensor -- e.g., `down_proj.output * 2` -- will attempt to
    multiply a tuple, causing a TypeError or silently wrong behavior if the tuple
    is unpacked incorrectly.

WHY THIS MATTERS:
    Any intervention that reads or modifies down_proj or o_proj outputs needs to
    know about the tuple wrapper. This affects MLP output ablation, attention
    output steering, and any code that saves these outputs for analysis. The
    extra bias element is usually None, but its presence breaks the interface
    contract that users rely on.

VALIDATION: Check whether down_proj.output and o_proj.output are tuples (vLLM)
or single tensors (HF).
"""

import argparse
import json

import torch


def run_vllm(model_name, prompt):
    from nnsight.modeling.vllm import VLLM

    model = VLLM(model_name, gpu_memory_utilization=0.05, dispatch=True)

    # Forward order: self_attn (o_proj) runs before mlp (down_proj)
    with model.trace(prompt, temperature=0.0):
        o_out = model.model.layers[0].self_attn.o_proj.output.save()
        down_out = model.model.layers[0].mlp.down_proj.output.save()

    down_is_tuple = isinstance(down_out, tuple)
    o_is_tuple = isinstance(o_out, tuple)

    print("vLLM mlp.down_proj.output:")
    if down_is_tuple:
        print(f"  IS TUPLE, length={len(down_out)}")
        for i, t in enumerate(down_out):
            if isinstance(t, torch.Tensor):
                print(f"  [{i}]: shape={t.shape}")
            else:
                print(f"  [{i}]: {t}")
    else:
        print(f"  IS TENSOR, shape={down_out.shape}")

    print("\nvLLM self_attn.o_proj.output:")
    if o_is_tuple:
        print(f"  IS TUPLE, length={len(o_out)}")
        for i, t in enumerate(o_out):
            if isinstance(t, torch.Tensor):
                print(f"  [{i}]: shape={t.shape}")
            else:
                print(f"  [{i}]: {t}")
    else:
        print(f"  IS TENSOR, shape={o_out.shape}")

    gap_confirmed = down_is_tuple or o_is_tuple
    status = "CONFIRMED" if gap_confirmed else "NOT_REPRODUCED"
    detail = f"vLLM: down_proj tuple={down_is_tuple}, o_proj tuple={o_is_tuple}"
    return {
        "backend": "vllm",
        "status": status,
        "detail": detail,
        "down_is_tuple": down_is_tuple,
        "o_is_tuple": o_is_tuple,
    }


def run_hf(model_name, prompt):
    from nnsight import LanguageModel

    model = LanguageModel(model_name, device_map="cuda", dispatch=True)

    # HF forward order: self_attn runs before mlp
    # Inside self_attn: q_proj, k_proj, v_proj, attention, o_proj
    # Inside mlp: gate_proj, up_proj, act_fn, down_proj
    with model.trace(prompt):
        o_out = model.model.layers[0].self_attn.o_proj.output.save()
        down_out = model.model.layers[0].mlp.down_proj.output.save()

    down_is_tuple = isinstance(down_out, tuple)
    o_is_tuple = isinstance(o_out, tuple)

    print("HF mlp.down_proj.output:")
    if down_is_tuple:
        print(f"  IS TUPLE, length={len(down_out)}")
    elif isinstance(down_out, torch.Tensor):
        print(f"  IS TENSOR, shape={down_out.shape}")
    else:
        print(f"  type={type(down_out)}")

    print("\nHF self_attn.o_proj.output:")
    if o_is_tuple:
        print(f"  IS TUPLE, length={len(o_out)}")
    elif isinstance(o_out, torch.Tensor):
        print(f"  IS TENSOR, shape={o_out.shape}")
    else:
        print(f"  type={type(o_out)}")

    # HF: both should be single tensors (nn.Linear returns tensor)
    both_tensor = not down_is_tuple and not o_is_tuple
    status = "NO_GAP" if both_tensor else "UNEXPECTED"
    detail = f"HF: down_proj tuple={down_is_tuple}, o_proj tuple={o_is_tuple}"
    return {
        "backend": "hf",
        "status": status,
        "detail": detail,
        "down_is_tuple": down_is_tuple,
        "o_is_tuple": o_is_tuple,
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
