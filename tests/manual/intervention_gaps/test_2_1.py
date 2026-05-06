"""
Gap 2.1: mlp.gate_proj and mlp.up_proj do not exist -- merged into gate_up_proj

DOCUMENTED PATTERN (CLAUDE.md "Accessing Outputs" implies per-submodule access):
    gate_out = model.model.layers[0].mlp.gate_proj.output.save()
    up_out = model.model.layers[0].mlp.up_proj.output.save()
Users expect to access individual MLP projections to study gating mechanisms,
train SAEs on individual projection outputs, or selectively ablate one
projection while keeping the other.

ON HF (expected behavior):
    mlp has separate gate_proj (Linear), up_proj (Linear), down_proj (Linear),
    and act_fn (SiLU). Each is a distinct module with its own .output hook.
    Researchers can read, modify, or ablate gate and up projections independently.

ON vLLM (the gap):
    gate_proj and up_proj DO NOT EXIST. They are merged into a single
    gate_up_proj (MergedColumnParallelLinear) whose output is
    [tokens, 2 * intermediate_size] -- the gate and up projections concatenated.
    act_fn is SiluAndMul (fused). There is no way to hook into gate_proj or
    up_proj individually. Accessing model.model.layers[0].mlp.gate_proj raises
    AttributeError.

WHY THIS MATTERS:
    Mechanistic interpretability research frequently studies gate vs up
    projections separately -- e.g., analyzing which features the gate allows
    through, training separate SAEs on each projection, or ablating one to
    understand its causal role. The merged module makes all of these workflows
    impossible without manual tensor splitting.

VALIDATION: Check for existence of gate_proj, up_proj, and gate_up_proj
attributes on the MLP module.
"""

import argparse
import json

import torch


def run_vllm(model_name, prompt):
    from nnsight.modeling.vllm import VLLM

    model = VLLM(model_name, gpu_memory_utilization=0.05, dispatch=True)

    print("vLLM MLP structure:")
    vllm_mlp = model.model.layers[0].mlp
    print(f"  {vllm_mlp}")

    vllm_has_gate_proj = hasattr(vllm_mlp, "gate_proj")
    vllm_has_up_proj = hasattr(vllm_mlp, "up_proj")
    vllm_has_gate_up_proj = hasattr(vllm_mlp, "gate_up_proj")
    vllm_has_down_proj = hasattr(vllm_mlp, "down_proj")

    print(f"  gate_proj: {vllm_has_gate_proj}")
    print(f"  up_proj: {vllm_has_up_proj}")
    print(f"  gate_up_proj: {vllm_has_gate_up_proj}")
    print(f"  down_proj: {vllm_has_down_proj}")

    # Forward order: gate_up_proj -> act_fn -> down_proj
    with model.trace(prompt, temperature=0.0):
        if vllm_has_gate_up_proj:
            gu_out = model.model.layers[0].mlp.gate_up_proj.output.save()
        act_out = model.model.layers[0].mlp.act_fn.output.save()
        down_out = model.model.layers[0].mlp.down_proj.output.save()

    if vllm_has_gate_up_proj:
        if isinstance(gu_out, tuple):
            print(f"\n  gate_up_proj.output: tuple, [0].shape={gu_out[0].shape}")
        else:
            print(f"\n  gate_up_proj.output: shape={gu_out.shape}")

    if isinstance(down_out, tuple):
        print(f"  down_proj.output: tuple, [0].shape={down_out[0].shape}")
    else:
        print(f"  down_proj.output: shape={down_out.shape}")

    if isinstance(act_out, torch.Tensor):
        print(f"  act_fn.output: shape={act_out.shape}")
    else:
        print(f"  act_fn.output: type={type(act_out)}")

    gap_confirmed = (not vllm_has_gate_proj) and vllm_has_gate_up_proj
    status = "CONFIRMED" if gap_confirmed else "NOT_REPRODUCED"
    detail = (
        f"vLLM: gate_proj={vllm_has_gate_proj}, up_proj={vllm_has_up_proj}, "
        f"gate_up_proj={vllm_has_gate_up_proj}"
    )
    return {
        "backend": "vllm",
        "status": status,
        "detail": detail,
        "has_gate_proj": vllm_has_gate_proj,
        "has_up_proj": vllm_has_up_proj,
        "has_gate_up_proj": vllm_has_gate_up_proj,
    }


def run_hf(model_name, prompt):
    from nnsight import LanguageModel

    model = LanguageModel(model_name, device_map="cuda", dispatch=True)

    print("HF MLP structure:")
    hf_mlp = model.model.layers[0].mlp
    print(f"  {hf_mlp}")

    hf_has_gate_proj = hasattr(hf_mlp, "gate_proj")
    hf_has_up_proj = hasattr(hf_mlp, "up_proj")
    hf_has_gate_up_proj = hasattr(hf_mlp, "gate_up_proj")
    hf_has_down_proj = hasattr(hf_mlp, "down_proj")

    print(f"  gate_proj: {hf_has_gate_proj}")
    print(f"  up_proj: {hf_has_up_proj}")
    print(f"  gate_up_proj: {hf_has_gate_up_proj}")
    print(f"  down_proj: {hf_has_down_proj}")

    # HF: separate gate_proj and up_proj, no gate_up_proj
    status = "NO_GAP" if hf_has_gate_proj and hf_has_up_proj and not hf_has_gate_up_proj else "UNEXPECTED"
    detail = (
        f"HF: gate_proj={hf_has_gate_proj}, up_proj={hf_has_up_proj}, "
        f"gate_up_proj={hf_has_gate_up_proj}"
    )
    return {
        "backend": "hf",
        "status": status,
        "detail": detail,
        "has_gate_proj": hf_has_gate_proj,
        "has_up_proj": hf_has_up_proj,
        "has_gate_up_proj": hf_has_gate_up_proj,
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
