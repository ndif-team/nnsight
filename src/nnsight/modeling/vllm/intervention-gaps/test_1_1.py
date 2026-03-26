"""
Gap 1.1: .save() captures corrupted values due to in-place mutation by fused kernels

DOCUMENTED PATTERN (CLAUDE.md "Accessing Outputs"):
    layer_5_out = model.transformer.h[5].output[0].save()
Users expect .save() to capture the tensor value at the moment the hook fires,
preserving the module's actual output for later analysis.

ON HF (expected behavior):
    .save() and .clone().save() return identical values. The saved reference
    faithfully represents the layer's output. Probing classifiers, activation
    patching, and logit lens all work correctly on the saved tensor.

ON vLLM (the gap):
    .save() and .clone().save() DIFFER. vLLM's fused_add_rms_norm mutates
    tensors in-place AFTER the hook fires: when layer N returns (x, residual),
    layer N+1's input_layernorm(x, residual) overwrites x with rms_norm(x +
    residual). Since .save() stores a reference (not a copy), the saved tensor
    silently becomes the post-mutation value -- not what the layer actually output.

WHY THIS MATTERS:
    This is the most insidious gap because it produces wrong results silently.
    Any workflow that saves layer hidden states for offline analysis -- probing
    classifiers, activation patching between clean/corrupted runs, logit lens,
    or simply logging activations -- gets corrupted data without any error.
    The user's code looks correct, runs without errors, but the saved tensors
    contain values from a LATER point in the computation.

VALIDATION: Compare .save() vs .clone().save() for layer outputs, mlp outputs,
and self_attn outputs. If they differ, the reference was silently corrupted.
"""

import argparse
import json

import torch


def clone_value(v):
    """Clone a tensor or each tensor in a tuple."""
    if isinstance(v, torch.Tensor):
        return v.clone()
    elif isinstance(v, tuple):
        return tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in v)
    return v


def compare_values(ref, cloned):
    """Compare ref vs cloned, returning (match, max_diff)."""
    if isinstance(ref, torch.Tensor) and isinstance(cloned, torch.Tensor):
        match = torch.equal(ref, cloned)
        diff = 0.0 if match else torch.max(torch.abs(ref.float() - cloned.float())).item()
        return match, diff
    elif isinstance(ref, tuple) and isinstance(cloned, tuple):
        matches = []
        diffs = []
        for r, c in zip(ref, cloned):
            if isinstance(r, torch.Tensor) and isinstance(c, torch.Tensor):
                m = torch.equal(r, c)
                d = 0.0 if m else torch.max(torch.abs(r.float() - c.float())).item()
                matches.append(m)
                diffs.append(d)
        return all(matches), max(diffs) if diffs else 0.0
    return True, 0.0


def run_vllm(model_name, prompt, layer_idx):
    from nnsight.modeling.vllm import VLLM

    model = VLLM(model_name, gpu_memory_utilization=0.05, dispatch=True)
    findings = {}

    # --- Test layer.output[0] (compat layer combines dual streams) ---
    with model.trace(prompt, temperature=0.0):
        layer_out0_ref = model.model.layers[layer_idx].output[0].save()
        layer_out0_clone = model.model.layers[layer_idx].output[0].clone().save()

    match_0, diff_0 = compare_values(layer_out0_ref, layer_out0_clone)
    findings["layer.output[0]"] = {"match": match_0, "max_diff": diff_0}
    print(f"layer.output[0]: ref==clone? {match_0}  max_diff={diff_0:.6f}")

    # --- Test mlp.output ---
    with model.trace(prompt, temperature=0.0):
        mlp_ref = model.model.layers[layer_idx].mlp.output.save()

    with model.trace(prompt, temperature=0.0):
        mlp_ref2 = model.model.layers[layer_idx].mlp.output.save()

    match_mlp, diff_mlp = compare_values(mlp_ref, mlp_ref2)
    print(f"mlp.output type: {type(mlp_ref)}")
    findings["mlp.output"] = {"match": match_mlp, "max_diff": diff_mlp}
    print(f"mlp.output:      cross-trace match? {match_mlp}  max_diff={diff_mlp:.6f}")

    # --- Test self_attn.output ---
    with model.trace(prompt, temperature=0.0):
        attn_ref = model.model.layers[layer_idx].self_attn.output.save()

    with model.trace(prompt, temperature=0.0):
        attn_ref2 = model.model.layers[layer_idx].self_attn.output.save()

    match_attn, diff_attn = compare_values(attn_ref, attn_ref2)
    findings["self_attn.output"] = {"match": match_attn, "max_diff": diff_attn}
    print(f"self_attn.output: cross-trace match? {match_attn}  max_diff={diff_attn:.6f}")

    corrupted = [k for k, v in findings.items() if not v["match"]]
    safe = [k for k, v in findings.items() if v["match"]]

    if not match_0:
        status = "CONFIRMED"
    else:
        status = "NOT_REPRODUCED"

    detail = (
        f"layer.output[0] ref==clone: {match_0} (diff={diff_0:.6f}); "
        f"Corrupted: {corrupted}; Safe: {safe}"
    )
    return {
        "backend": "vllm",
        "status": status,
        "detail": detail,
        "findings": {k: {"match": v["match"], "max_diff": v["max_diff"]} for k, v in findings.items()},
    }


def run_hf(model_name, prompt, layer_idx):
    from nnsight import LanguageModel

    model = LanguageModel(model_name, device_map="cuda", dispatch=True)
    findings = {}

    # --- Test layer.output[0]: HF returns (hidden_states,) tuple ---
    # HF has no fused in-place mutation, so ref==clone should always be True
    with model.trace(prompt):
        layer_out0_ref = model.model.layers[layer_idx].output[0].save()
        layer_out0_clone = model.model.layers[layer_idx].output[0].clone().save()

    match_0, diff_0 = compare_values(layer_out0_ref, layer_out0_clone)
    findings["layer.output[0]"] = {"match": match_0, "max_diff": diff_0}
    print(f"layer.output[0]: ref==clone? {match_0}  max_diff={diff_0:.6f}")

    # --- Test mlp.output ---
    with model.trace(prompt):
        mlp_ref = model.model.layers[layer_idx].mlp.output.save()

    with model.trace(prompt):
        mlp_ref2 = model.model.layers[layer_idx].mlp.output.save()

    match_mlp, diff_mlp = compare_values(mlp_ref, mlp_ref2)
    findings["mlp.output"] = {"match": match_mlp, "max_diff": diff_mlp}
    print(f"mlp.output:      cross-trace match? {match_mlp}  max_diff={diff_mlp:.6f}")

    # --- Test self_attn.output ---
    with model.trace(prompt):
        attn_ref = model.model.layers[layer_idx].self_attn.output.save()

    with model.trace(prompt):
        attn_ref2 = model.model.layers[layer_idx].self_attn.output.save()

    match_attn, diff_attn = compare_values(attn_ref, attn_ref2)
    findings["self_attn.output"] = {"match": match_attn, "max_diff": diff_attn}
    print(f"self_attn.output: cross-trace match? {match_attn}  max_diff={diff_attn:.6f}")

    corrupted = [k for k, v in findings.items() if not v["match"]]
    safe = [k for k, v in findings.items() if v["match"]]

    all_match = all(v["match"] for v in findings.values())
    status = "NO_GAP" if all_match else "UNEXPECTED_MISMATCH"

    detail = (
        f"layer.output[0] ref==clone: {match_0} (diff={diff_0:.6f}); "
        f"Corrupted: {corrupted}; Safe: {safe}"
    )
    return {
        "backend": "hf",
        "status": status,
        "detail": detail,
        "findings": {k: {"match": v["match"], "max_diff": v["max_diff"]} for k, v in findings.items()},
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
