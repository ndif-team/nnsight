"""
Gap 1.5: LayerNorm takes 2 args (raw_output, residual) instead of 1 (hidden_states)

DOCUMENTED PATTERN (related to Gap 1.4 -- CLAUDE.md "Accessing Inputs"):
    layer_input = model.transformer.h[0].input.save()
    # .input is the first positional arg; .inputs gives (args_tuple, kwargs_dict)
On HF, input_layernorm takes a single hidden_states tensor. Users who hook into
layernorm inputs to read or modify the pre-normalization hidden state expect one
tensor argument.

ON HF (expected behavior):
    input_layernorm.input is a single float tensor (hidden_states). The inputs
    tuple has 1 positional arg. Code that reads or replaces the layernorm input
    to inject modified hidden states works directly.

ON vLLM (the gap):
    input_layernorm.input is the FIRST of TWO positional args. The fused kernel
    signature is fused_add_rms_norm(raw_output, residual), so .inputs has 2 args.
    .input gives only arg[0] (the raw sub-layer output), missing the residual
    entirely. Code that reads .input expecting the full hidden state gets an
    incomplete value.

WHY THIS MATTERS:
    Any code that hooks into layernorm inputs to read pre-normalization hidden
    states, modify them (e.g., adding steering vectors before normalization), or
    replace them for causal tracing experiments will operate on the wrong tensor.
    The user gets the raw attention/MLP output without the residual connection,
    not the full hidden state that would be normalized.

VALIDATION: Count positional args to input_layernorm and post_attention_layernorm.
HF has 1, vLLM has 2.
"""

import argparse
import json

import torch


def run_vllm(model_name, prompt, layer_idx):
    from nnsight.modeling.vllm import VLLM

    model = VLLM(model_name, gpu_memory_utilization=0.05, dispatch=True)

    with model.trace(prompt, temperature=0.0):
        vllm_ln_input = model.model.layers[layer_idx].input_layernorm.input.save()
        vllm_ln_inputs = model.model.layers[layer_idx].input_layernorm.inputs.save()
        vllm_post_ln_input = model.model.layers[layer_idx].post_attention_layernorm.input.save()
        vllm_post_ln_inputs = model.model.layers[layer_idx].post_attention_layernorm.inputs.save()

    print("vLLM input_layernorm.input:")
    if isinstance(vllm_ln_input, torch.Tensor):
        print(f"  IS TENSOR, shape={vllm_ln_input.shape}, dtype={vllm_ln_input.dtype}")
    else:
        print(f"  type={type(vllm_ln_input)}")

    print("\nvLLM input_layernorm.inputs (args, kwargs):")
    vllm_args, vllm_kwargs = vllm_ln_inputs
    for i, a in enumerate(vllm_args):
        if isinstance(a, torch.Tensor):
            print(f"  arg[{i}]: shape={a.shape}, dtype={a.dtype}")
        else:
            print(f"  arg[{i}]: type={type(a)}")

    print("\nvLLM post_attention_layernorm.input:")
    if isinstance(vllm_post_ln_input, torch.Tensor):
        print(f"  IS TENSOR, shape={vllm_post_ln_input.shape}")
    else:
        print(f"  type={type(vllm_post_ln_input)}")

    print("\nvLLM post_attention_layernorm.inputs (args, kwargs):")
    vllm_post_args, vllm_post_kwargs = vllm_post_ln_inputs
    for i, a in enumerate(vllm_post_args):
        if isinstance(a, torch.Tensor):
            print(f"  arg[{i}]: shape={a.shape}, dtype={a.dtype}")
        else:
            print(f"  arg[{i}]: type={type(a)}")

    vllm_num_args = len(vllm_args)
    status = "CONFIRMED" if vllm_num_args >= 2 else "NOT_REPRODUCED"
    detail = f"vLLM input_layernorm num_args={vllm_num_args}"
    return {
        "backend": "vllm",
        "status": status,
        "detail": detail,
        "num_args": vllm_num_args,
    }


def run_hf(model_name, prompt, layer_idx):
    from nnsight import LanguageModel

    model = LanguageModel(model_name, device_map="cuda", dispatch=True)

    with model.trace(prompt):
        hf_ln_input = model.model.layers[layer_idx].input_layernorm.input.save()
        hf_ln_inputs = model.model.layers[layer_idx].input_layernorm.inputs.save()
        hf_post_ln_input = model.model.layers[layer_idx].post_attention_layernorm.input.save()
        hf_post_ln_inputs = model.model.layers[layer_idx].post_attention_layernorm.inputs.save()

    print("HF input_layernorm.input:")
    if isinstance(hf_ln_input, torch.Tensor):
        print(f"  IS TENSOR, shape={hf_ln_input.shape}, dtype={hf_ln_input.dtype}")
    else:
        print(f"  type={type(hf_ln_input)}")

    print("\nHF input_layernorm.inputs (args, kwargs):")
    hf_args, hf_kwargs = hf_ln_inputs
    for i, a in enumerate(hf_args):
        if isinstance(a, torch.Tensor):
            print(f"  arg[{i}]: shape={a.shape}, dtype={a.dtype}")
        elif a is None:
            print(f"  arg[{i}]: None")
        else:
            print(f"  arg[{i}]: type={type(a)}")

    print("\nHF post_attention_layernorm.input:")
    if isinstance(hf_post_ln_input, torch.Tensor):
        print(f"  IS TENSOR, shape={hf_post_ln_input.shape}, dtype={hf_post_ln_input.dtype}")
    else:
        print(f"  type={type(hf_post_ln_input)}")

    print("\nHF post_attention_layernorm.inputs (args, kwargs):")
    hf_post_args, hf_post_kwargs = hf_post_ln_inputs
    for i, a in enumerate(hf_post_args):
        if isinstance(a, torch.Tensor):
            print(f"  arg[{i}]: shape={a.shape}, dtype={a.dtype}")
        elif a is None:
            print(f"  arg[{i}]: None")
        else:
            print(f"  arg[{i}]: type={type(a)}")

    hf_num_args = len(hf_args)
    # HF: layernorm takes 1 positional arg
    status = "NO_GAP" if hf_num_args == 1 else "UNEXPECTED"
    detail = f"HF input_layernorm num_args={hf_num_args}"
    return {
        "backend": "hf",
        "status": status,
        "detail": detail,
        "num_args": hf_num_args,
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
