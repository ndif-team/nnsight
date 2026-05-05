"""
Gap 1.3: layer.input returns position IDs instead of hidden states

DOCUMENTED PATTERN (CLAUDE.md "Accessing Inputs"):
    layer_input = model.transformer.h[0].input.save()  # Users expect: hidden states
The docs define .input as "the first positional argument to the module." Users
expect this to be the hidden state tensor flowing into the layer, which is what
they would analyze, log, or use as a patching source.

ON HF (expected behavior):
    layer.input is a float tensor of shape [batch, seq, hidden_dim] -- the hidden
    states from the previous layer. layer args = (hidden_states, attention_mask,
    position_ids, ...). Code that reads .input for analysis gets the expected
    representation.

ON vLLM (the gap):
    layer.input is an int64 tensor of position IDs, NOT hidden states. vLLM's
    decoder layer signature is forward(positions, hidden_states, residual, ...),
    so the first positional arg is positions. The hidden states are in arg[1].

WHY THIS MATTERS:
    Any code that reads .input to access hidden states entering a layer --
    for input-output comparisons, residual stream analysis, or activation
    patching at layer boundaries -- gets integer position IDs instead of float
    hidden states. Arithmetic on this (e.g., adding a steering vector) produces
    nonsense without any type error because PyTorch silently promotes int to float.

VALIDATION: Check dtype of layer.input -- int64 (vLLM positions) vs float (HF
hidden states).
"""

import argparse
import json

import torch


def run_vllm(model_name, prompt, layer_idx):
    from nnsight.modeling.vllm import VLLM

    model = VLLM(model_name, gpu_memory_utilization=0.05, dispatch=True)

    with model.trace(prompt, temperature=0.0):
        vllm_input = model.model.layers[layer_idx].input.save()
        vllm_inputs = model.model.layers[layer_idx].inputs.save()

    print("vLLM layer.input:")
    if isinstance(vllm_input, torch.Tensor):
        print(f"  shape={vllm_input.shape}, dtype={vllm_input.dtype}")
        print(f"  values (first 5): {vllm_input.flatten()[:5].tolist()}")
        is_positions = vllm_input.dtype in (torch.int32, torch.int64, torch.long)
    else:
        print(f"  type={type(vllm_input)}")
        is_positions = False

    print("\nvLLM layer.inputs (args, kwargs):")
    vllm_args, vllm_kwargs = vllm_inputs
    for i, a in enumerate(vllm_args):
        if isinstance(a, torch.Tensor):
            print(f"  arg[{i}]: shape={a.shape}, dtype={a.dtype}")
        else:
            print(f"  arg[{i}]: type={type(a)}")
    for k, v in vllm_kwargs.items():
        if isinstance(v, torch.Tensor):
            print(f"  kwarg[{k}]: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  kwarg[{k}]: type={type(v)}")

    input_dtype = str(vllm_input.dtype) if isinstance(vllm_input, torch.Tensor) else type(vllm_input).__name__
    status = "CONFIRMED" if is_positions else "NOT_REPRODUCED"
    detail = f"vLLM layer.input dtype={input_dtype}; is_positions={is_positions}"
    return {
        "backend": "vllm",
        "status": status,
        "detail": detail,
        "input_dtype": input_dtype,
        "is_positions": is_positions,
    }


def run_hf(model_name, prompt, layer_idx):
    from nnsight import LanguageModel

    model = LanguageModel(model_name, device_map="cuda", dispatch=True)

    with model.trace(prompt):
        hf_input = model.model.layers[layer_idx].input.save()
        hf_inputs = model.model.layers[layer_idx].inputs.save()

    print("HF layer.input:")
    if isinstance(hf_input, torch.Tensor):
        print(f"  shape={hf_input.shape}, dtype={hf_input.dtype}")
        is_float = hf_input.dtype.is_floating_point
    else:
        print(f"  type={type(hf_input)}")
        is_float = False

    print("\nHF layer.inputs (args, kwargs):")
    hf_args, hf_kwargs = hf_inputs
    for i, a in enumerate(hf_args):
        if isinstance(a, torch.Tensor):
            print(f"  arg[{i}]: shape={a.shape}, dtype={a.dtype}")
        elif a is None:
            print(f"  arg[{i}]: None")
        else:
            print(f"  arg[{i}]: type={type(a)}")
    for k, v in hf_kwargs.items():
        if isinstance(v, torch.Tensor):
            print(f"  kwarg[{k}]: shape={v.shape}, dtype={v.dtype}")
        elif v is None:
            print(f"  kwarg[{k}]: None")
        else:
            print(f"  kwarg[{k}]: type={type(v)}")

    input_dtype = str(hf_input.dtype) if isinstance(hf_input, torch.Tensor) else type(hf_input).__name__
    # HF: layer.input should be float hidden_states
    status = "NO_GAP" if is_float else "UNEXPECTED"
    detail = f"HF layer.input dtype={input_dtype}; is_float={is_float}"
    return {
        "backend": "hf",
        "status": status,
        "detail": detail,
        "input_dtype": input_dtype,
        "is_float": is_float,
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
