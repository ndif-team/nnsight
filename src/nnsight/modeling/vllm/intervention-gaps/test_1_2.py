"""
Gap 1.2: layer.output[0] is raw MLP output instead of full residual stream

DOCUMENTED PATTERN (CLAUDE.md "Accessing Outputs" and "Common Patterns > Logit Lens"):
    hs = model.transformer.h[i].output[0]  # Users expect: full hidden state
    logits = model.lm_head(model.transformer.ln_f(hs))  # Logit lens
Users expect output[0] to be the complete hidden state after the layer --
the residual stream that flows between layers (mlp_out + residual).

ON HF (expected behavior):
    layer.output = (hidden_states,) where hidden_states = mlp_out + residual.
    output[0] IS the full residual stream. Logit lens, probing classifiers,
    and steering vectors all operate on the correct representation.

ON vLLM (the gap):
    layer.output = (hidden_states, residual) where hidden_states is the RAW MLP
    output WITHOUT the residual added. output[0] == mlp.output, not the full
    hidden state. The residual is carried separately in output[1] because vLLM's
    fused norm adds it in the next layer. To reconstruct HF-equivalent:
    h = layer.output[0] + layer.output[1].

WHY THIS MATTERS:
    Every interpretability technique that reads layer hidden states gets the wrong
    tensor: logit lens applies the unembedding to raw MLP output instead of the
    residual stream, probing classifiers train on the wrong representation,
    steering vectors are added to the wrong space, and activation patching patches
    an incomplete value. The code runs without error but all results are silently
    incorrect.

VALIDATION: Check whether output[0] matches mlp.output (vLLM gap) or differs
from it (HF expected -- output[0] includes residual).
"""

import argparse
import json

import torch


def run_vllm(model_name, prompt, layer_idx):
    from nnsight.modeling.vllm import VLLM

    model = VLLM(model_name, gpu_memory_utilization=0.05, dispatch=True)

    # With compat layer: output is now (combined,) where combined = mlp_out + residual
    # So output[0] should NOT equal mlp.output (it includes the residual).
    with model.trace(prompt, temperature=0.0):
        mlp_out = model.model.layers[layer_idx].mlp.output.clone().save()
        out0 = model.model.layers[layer_idx].output[0].clone().save()

    print(f"layer.output[0] shape: {out0.shape}, dtype: {out0.dtype}")
    print(f"mlp.output shape: {mlp_out.shape if isinstance(mlp_out, torch.Tensor) else [t.shape for t in mlp_out]}")

    if isinstance(mlp_out, tuple):
        mlp_tensor = mlp_out[0]
    else:
        mlp_tensor = mlp_out

    out0_eq_mlp = torch.equal(out0, mlp_tensor)
    if not out0_eq_mlp:
        out0_mlp_diff = torch.max(torch.abs(out0.float() - mlp_tensor.float())).item()
        out0_mlp_cos = torch.nn.functional.cosine_similarity(
            out0.float().reshape(1, -1), mlp_tensor.float().reshape(1, -1)
        ).item()
    else:
        out0_mlp_diff = 0.0
        out0_mlp_cos = 1.0

    print(f"\noutput[0] == mlp.output: {out0_eq_mlp} (diff={out0_mlp_diff:.6f}, cos={out0_mlp_cos:.6f})")

    # Gap is confirmed if output[0] == mlp.output (raw MLP, no residual added).
    # Gap is NOT reproduced if output[0] != mlp.output (compat layer combined them).
    status = "CONFIRMED" if out0_eq_mlp or out0_mlp_cos > 0.99 else "NOT_REPRODUCED"
    detail = (
        f"output[0]==mlp.output: {out0_eq_mlp} (cos={out0_mlp_cos:.4f}); "
        f"{'compat layer combined dual streams' if not out0_eq_mlp else 'still raw mlp output'}"
    )
    return {
        "backend": "vllm",
        "status": status,
        "detail": detail,
        "out0_eq_mlp": out0_eq_mlp,
    }


def run_hf(model_name, prompt, layer_idx):
    from nnsight import LanguageModel

    model = LanguageModel(model_name, device_map="cuda", dispatch=True)

    # HF forward order: input_layernorm -> self_attn -> post_attention_layernorm -> mlp -> layer returns
    # So mlp.output comes BEFORE layer.output
    with model.trace(prompt):
        mlp_out = model.model.layers[layer_idx].mlp.output.clone().save()
        layer_out = model.model.layers[layer_idx].output.save()

    out0 = layer_out[0]
    is_tuple = isinstance(layer_out, tuple)
    output_len = len(layer_out) if is_tuple else 1

    print(f"layer.output type: {type(layer_out)}")
    print(f"layer.output length: {output_len}")
    print(f"layer.output[0] shape: {out0.shape}, dtype: {out0.dtype}")
    print(f"mlp.output shape: {mlp_out.shape if isinstance(mlp_out, torch.Tensor) else type(mlp_out)}")

    if isinstance(mlp_out, torch.Tensor):
        mlp_tensor = mlp_out
    elif isinstance(mlp_out, tuple):
        mlp_tensor = mlp_out[0]
    else:
        mlp_tensor = mlp_out

    out0_eq_mlp = torch.equal(out0, mlp_tensor)
    if not out0_eq_mlp:
        out0_mlp_diff = torch.max(torch.abs(out0.float() - mlp_tensor.float())).item()
        out0_mlp_cos = torch.nn.functional.cosine_similarity(
            out0.float().reshape(1, -1), mlp_tensor.float().reshape(1, -1)
        ).item()
    else:
        out0_mlp_diff = 0.0
        out0_mlp_cos = 1.0

    print(f"\noutput[0] == mlp.output: {out0_eq_mlp} (diff={out0_mlp_diff:.6f}, cos={out0_mlp_cos:.6f})")

    # HF: output[0] should be mlp_out + residual, NOT equal to raw mlp_out
    status = "NO_GAP" if not out0_eq_mlp else "UNEXPECTED_MATCH"
    detail = (
        f"output is {output_len}-tuple; output[0]==mlp.output: {out0_eq_mlp} (cos={out0_mlp_cos:.4f}); "
        f"HF returns (mlp_out+residual,) — output[0] includes residual"
    )
    return {
        "backend": "hf",
        "status": status,
        "detail": detail,
        "output_len": output_len,
        "out0_eq_mlp": out0_eq_mlp,
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
