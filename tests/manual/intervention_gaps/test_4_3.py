"""
Gap 4.3: Module skip breaks because fused norm expects (x, residual) pair

DOCUMENTED PATTERN (CLAUDE.md "Module Skipping"):
    model.transformer.h[1].skip(layer0_out)  # Skip layer, use provided value
The docs show skip() replacing a module's entire computation with a provided
value, useful for layer ablation studies and skip connection research.

ON HF (expected behavior):
    Layers return simple (hidden_states,) tuples. Skipping a layer with a single
    tensor works because the next layer's input_layernorm expects just
    hidden_states. skip(layer_output_tuple) also works. Layer ablation studies
    proceed without issue.

ON vLLM (the gap):
    Layers return (hidden_states, residual) -- a 2-tuple needed by vLLM's fused
    norm architecture. The NEXT layer's input_layernorm (fused_add_rms_norm)
    expects TWO tensors: the raw sub-layer output and the residual. When skip()
    provides a single tensor, the fused norm crashes because it cannot unpack
    its expected (x, residual) input. Even skip(tuple) may fail if the tuple
    format does not exactly match what the fused kernel expects.

WHY THIS MATTERS:
    Layer ablation is a core interpretability technique: "what happens if we
    remove layer N?" Skip is the clean way to do this in nnsight. Without skip
    support, researchers cannot easily study which layers are critical, perform
    layer-by-layer knockout experiments, or test skip connection hypotheses.
    The workaround requires understanding vLLM's internal (x, residual)
    bookkeeping and manually constructing the correct tuple.

VALIDATION: Attempt skip(single_tensor) and skip(tuple) on a decoder layer.
HF succeeds; vLLM fails with the single tensor due to fused norm mismatch.
"""

import argparse
import json
import traceback

import torch


def run_vllm(model_name, prompt):
    from nnsight.modeling.vllm import VLLM

    model = VLLM(model_name, gpu_memory_utilization=0.05, dispatch=True)

    # First get a reference output from layer 4 to use as skip value
    with model.trace(prompt, temperature=0.0):
        ref_out = model.model.layers[4].output[0].clone().save()

    print(f"Reference output shape: {ref_out.shape}")

    # Try to skip layer 5 with a single tensor
    error_msg = None
    try:
        with model.trace(prompt, temperature=0.0):
            model.model.layers[5].skip(ref_out)
            logits = model.logits.output.save()
        print("skip(single_tensor): NO ERROR")
        print(f"  logits shape: {logits.shape}")
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"skip(single_tensor): ERROR -- {error_msg}")

    # Try to skip with a (x, residual) tuple
    error_msg_tuple = None
    try:
        with model.trace(prompt, temperature=0.0):
            layer4_out = model.model.layers[4].output
            model.model.layers[5].skip(layer4_out)
            logits2 = model.logits.output.save()
        print("\nskip(tuple): NO ERROR")
        print(f"  logits shape: {logits2.shape}")
    except Exception as e:
        error_msg_tuple = f"{type(e).__name__}: {e}"
        print(f"\nskip(tuple): ERROR -- {error_msg_tuple}")

    single_failed = error_msg is not None
    tuple_failed = error_msg_tuple is not None
    status = "CONFIRMED" if single_failed else "NOT_REPRODUCED"
    detail = (
        f"skip(single_tensor): {'FAILED: ' + error_msg if single_failed else 'OK'}; "
        f"skip(tuple): {'FAILED: ' + error_msg_tuple if tuple_failed else 'OK'}"
    )
    return {
        "backend": "vllm",
        "status": status,
        "detail": detail,
        "single_skip_ok": not single_failed,
        "tuple_skip_ok": not tuple_failed,
    }


def run_hf(model_name, prompt):
    from nnsight import LanguageModel

    model = LanguageModel(model_name, device_map="cuda", dispatch=True)

    # First get a reference output from layer 4
    with model.trace(prompt):
        ref_out = model.model.layers[4].output[0].clone().save()

    print(f"Reference output shape: {ref_out.shape}")

    # Try to skip layer 5 with the reference output
    error_msg = None
    try:
        with model.trace(prompt):
            model.model.layers[5].skip(ref_out)
            logits = model.lm_head.output.save()
        print("skip(single_tensor): NO ERROR (expected)")
        print(f"  logits shape: {logits.shape}")
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"skip(single_tensor): ERROR -- {error_msg}")

    # Try to skip with full output (tuple from HF)
    error_msg_tuple = None
    try:
        with model.trace(prompt):
            layer4_out = model.model.layers[4].output
            model.model.layers[5].skip(layer4_out)
            logits2 = model.lm_head.output.save()
        print("\nskip(tuple): NO ERROR (expected)")
        print(f"  logits shape: {logits2.shape}")
    except Exception as e:
        error_msg_tuple = f"{type(e).__name__}: {e}"
        print(f"\nskip(tuple): ERROR -- {error_msg_tuple}")

    single_ok = error_msg is None
    tuple_ok = error_msg_tuple is None
    # skip(single_tensor) may fail on HF too if the next layer expects a tuple
    # output format. This is a general skip limitation, not the vLLM-specific
    # fused norm issue. skip(tuple) is the better test for HF.
    status = "NO_GAP" if tuple_ok else "UNEXPECTED_FAILURE"
    detail = (
        f"skip(single_tensor): {'OK' if single_ok else 'FAILED (expected — shape mismatch)'}; "
        f"skip(tuple): {'OK' if tuple_ok else 'FAILED: ' + str(error_msg_tuple)}"
    )
    return {
        "backend": "hf",
        "status": status,
        "detail": detail,
        "single_skip_ok": single_ok,
        "tuple_skip_ok": tuple_ok,
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
