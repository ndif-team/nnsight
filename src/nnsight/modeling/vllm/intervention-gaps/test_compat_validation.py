"""
Direct validation: HF-style intervention patterns work identically on vLLM via compat layer.

Run: CUDA_VISIBLE_DEVICES=X python test_compat_validation.py
"""
import os
import gc
import torch

PROMPT = "The Eiffel Tower is in the city of"
MODEL = "Qwen/Qwen2-0.5B"
LAYER = 10


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten().unsqueeze(0),
        b.float().flatten().unsqueeze(0),
    ).item()


def maxdiff(a, b):
    return torch.max(torch.abs(a.float() - b.float())).item()


def run_vllm(steer_vec):
    from nnsight.modeling.vllm import VLLM

    lm = VLLM(MODEL, tensor_parallel_size=1, gpu_memory_utilization=0.05, dispatch=True)
    results = {}

    print("=" * 70)
    print("vLLM (with compat layer)")
    print("=" * 70)

    # 1. Save hidden states + check structure
    print("\n--- 1. Save + structure check (Gaps 1.1, 1.2) ---")
    with lm.trace(PROMPT, temperature=0.0, top_p=1):
        vllm_hs = lm.model.layers[LAYER].output[0].save()
        vllm_hs_clone = lm.model.layers[LAYER].output[0].clone().save()
    print(f"  output[0] shape: {vllm_hs.shape}, dtype: {vllm_hs.dtype}")
    diff_1_1 = maxdiff(vllm_hs, vllm_hs_clone)
    print(f"  .save() ref vs clone max_diff: {diff_1_1:.6f}")
    assert vllm_hs.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert diff_1_1 < 0.01, f"Gap 1.1 NOT fixed"
    results["gap_1_1"] = "PASS"
    results["save"] = "PASS"

    # 2. Layer input is float (Gap 1.3)
    print("\n--- 2. Layer input is float (Gap 1.3) ---")
    with lm.trace(PROMPT, temperature=0.0, top_p=1):
        inp = lm.model.layers[4].input.clone().save()
    print(f"  input dtype: {inp.dtype}, shape: {inp.shape}")
    assert inp.dtype in (torch.float16, torch.bfloat16, torch.float32)
    results["gap_1_3"] = "PASS"

    # 3. LayerNorm output is tensor (Gap 1.4)
    print("\n--- 3. LayerNorm output (Gap 1.4) ---")
    with lm.trace(PROMPT, temperature=0.0, top_p=1):
        ln_out = lm.model.layers[0].input_layernorm.output.clone().save()
    print(f"  LayerNorm output type: {type(ln_out).__name__}, shape: {ln_out.shape}")
    assert isinstance(ln_out, torch.Tensor), "Gap 1.4 NOT fixed"
    results["gap_1_4"] = "PASS"

    # 4. down_proj output is tensor (Gap 2.3)
    print("\n--- 4. down_proj output (Gap 2.3) ---")
    with lm.trace(PROMPT, temperature=0.0, top_p=1):
        dp_out = lm.model.layers[0].mlp.down_proj.output.clone().save()
    print(f"  down_proj output type: {type(dp_out).__name__}, shape: {dp_out.shape}")
    assert isinstance(dp_out, torch.Tensor), "Gap 2.3 NOT fixed"
    results["gap_2_3"] = "PASS"

    # 5. Ablation
    print("\n--- 5. Ablation: output[0][:] = 0 ---")
    with lm.trace(PROMPT, temperature=0.0, top_p=1):
        lm.model.layers[LAYER].output[0][:] = 0
        ablate_logits = lm.logits.output[0, -1].clone().save()
    print(f"  ablation logits shape: {ablate_logits.shape}")
    results["ablation"] = "PASS"

    # 6. Steering
    print("\n--- 6. Steering: output[0][-1, :] += vec ---")
    sv = steer_vec.to(vllm_hs.device).to(vllm_hs.dtype)
    with lm.trace(PROMPT, temperature=0.0, top_p=1):
        lm.model.layers[LAYER].output[0][-1, :] += sv
        steer_logits = lm.logits.output[0, -1].clone().save()
    print(f"  steer logits shape: {steer_logits.shape}")
    results["steer"] = "PASS"

    # 7. Patching
    print("\n--- 7. Patching: output[0][:] = target ---")
    target = vllm_hs.clone() * 0.5
    with lm.trace(PROMPT, temperature=0.0, top_p=1):
        lm.model.layers[LAYER].output[0][:] = target.to(lm.model.layers[LAYER].output[0].device)
        patch_logits = lm.logits.output[0, -1].clone().save()
    print(f"  patch logits shape: {patch_logits.shape}")
    results["patch"] = "PASS"

    # 8. Module skip (Gap 4.3)
    print("\n--- 8. Module skip (Gap 4.3) ---")
    try:
        with lm.trace(PROMPT, temperature=0.0, top_p=1):
            layer4_out = lm.model.layers[4].output
            lm.model.layers[5].skip(layer4_out)
            skip_logits = lm.logits.output.save()
        print(f"  skip logits shape: {skip_logits.shape}")
        results["gap_4_3"] = "PASS"
    except Exception as e:
        print(f"  FAILED: {e}")
        results["gap_4_3"] = f"FAIL: {e}"

    # 9. Baseline
    print("\n--- 9. Baseline ---")
    with lm.trace(PROMPT, temperature=0.0, top_p=1):
        baseline = lm.logits.output[0, -1].clone().save()
    print(f"  baseline logits shape: {baseline.shape}")

    tensors = {
        "baseline": baseline.cpu(),
        "ablate": ablate_logits.cpu(),
        "steer": steer_logits.cpu(),
        "patch": patch_logits.cpu(),
    }

    del lm
    torch.cuda.empty_cache()
    gc.collect()

    return results, tensors


def run_hf(steer_vec):
    from nnsight import LanguageModel

    hf = LanguageModel(MODEL, device_map="cuda", dispatch=True)

    print("\n" + "=" * 70)
    print("HF (reference)")
    print("=" * 70)

    # Hidden states
    with hf.trace(PROMPT):
        hf_hs = hf.model.layers[LAYER].output[0].save()

    # Ablation
    print("\n--- Ablation ---")
    with hf.trace(PROMPT):
        hf.model.layers[LAYER].output[0][:] = 0
        ablate_logits = hf.lm_head.output[-1].clone().save()
    print(f"  ablation done")

    # Steering
    print("\n--- Steering ---")
    sv = steer_vec.to(hf_hs.device).to(hf_hs.dtype)
    with hf.trace(PROMPT):
        hf.model.layers[LAYER].output[0][-1, :] += sv  # 2D inside implicit invoke
        steer_logits = hf.lm_head.output[-1].clone().save()
    print(f"  steering done")

    # Patching
    print("\n--- Patching ---")
    target = hf_hs.clone() * 0.5
    with hf.trace(PROMPT):
        hf.model.layers[LAYER].output[0][:] = target.to(hf.model.layers[LAYER].output[0].device)
        patch_logits = hf.lm_head.output[-1].clone().save()
    print(f"  patching done")

    # Baseline
    print("\n--- Baseline ---")
    with hf.trace(PROMPT):
        baseline = hf.lm_head.output[-1].clone().save()
    print(f"  baseline done")

    return {
        "baseline": baseline.cpu(),
        "ablate": ablate_logits.cpu(),
        "steer": steer_logits.cpu(),
        "patch": patch_logits.cpu(),
    }


def main():
    torch.manual_seed(42)
    steer_vec = torch.randn(896) * 0.1

    vllm_gap_results, vllm_tensors = run_vllm(steer_vec)
    hf_tensors = run_hf(steer_vec)

    # Comparison
    print("\n" + "=" * 70)
    print("CROSS-BACKEND COMPARISON: vLLM (compat) vs HF")
    print("=" * 70)

    comparisons = {
        "baseline": (vllm_tensors["baseline"], hf_tensors["baseline"]),
        "ablation": (vllm_tensors["ablate"], hf_tensors["ablate"]),
        "steering": (vllm_tensors["steer"], hf_tensors["steer"]),
        "patching": (vllm_tensors["patch"], hf_tensors["patch"]),
    }

    print(f"\n{'Operation':<15} {'max_diff':>10} {'cosine':>10} {'Status':>10}")
    print(f"{'-'*15} {'-'*10} {'-'*10} {'-'*10}")

    all_ok = True
    for name, (v, h) in comparisons.items():
        md = maxdiff(v, h)
        cos = cosine(v, h)
        status = "OK" if cos > 0.95 else "CONCERN"
        if status == "CONCERN":
            all_ok = False
        print(f"{name:<15} {md:>10.4f} {cos:>10.6f} {status:>10}")

    # Gap summary
    print(f"\n{'='*70}")
    print("COMPAT LAYER GAP FIX SUMMARY")
    print(f"{'='*70}")
    gap_tests = {
        "1.1 (.save() mutation)": vllm_gap_results.get("gap_1_1", "?"),
        "1.3 (float input)": vllm_gap_results.get("gap_1_3", "?"),
        "1.4 (LayerNorm tensor)": vllm_gap_results.get("gap_1_4", "?"),
        "2.3 (down_proj tensor)": vllm_gap_results.get("gap_2_3", "?"),
        "4.3 (module skip)": vllm_gap_results.get("gap_4_3", "?"),
    }
    for gap, status in gap_tests.items():
        print(f"  Gap {gap}: {status}")

    pass_count = sum(1 for s in gap_tests.values() if s == "PASS")
    print(f"\n  {pass_count}/{len(gap_tests)} gaps auto-fixed by compat layer")

    print(f"\n{'='*70}")
    if all_ok:
        print("VERDICT: HF-style interventions produce equivalent results on vLLM")
    else:
        print("VERDICT: SOME INTERVENTIONS DIFFER SIGNIFICANTLY")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
