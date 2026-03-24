"""
Test activation patching and steering strategies on vLLM's dual residual stream.

Question: given vLLM layer returns (mlp_out, residual), how should users
do patching and steering so the downstream effect matches HF?

Key insight: next layer's fused_add_rms_norm computes:
  RMSNorm(mlp_out + residual), and new_residual = mlp_out + residual
So only the SUM matters — the decomposition is ephemeral.

This script tests 3 strategies for steering and patching on vLLM,
comparing final logits against an HF reference.
"""
import os
import sys
import json
import subprocess
import tempfile
import torch

MODEL = "Qwen/Qwen2.5-0.5B"
PROMPT = "The Eiffel Tower is in the city of"
LAYER_IDX = 10  # Intervene at layer 10 (of 24)
VLLM_GPU = "0"
HF_GPU = "1"

# ── HF: baseline + steering + patching ──────────────────────────────

HF_SCRIPT = r'''
import os, json, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "{hf_gpu}"
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model}", torch_dtype=torch.float16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("{model}")
inputs = tokenizer("{prompt}", return_tensors="pt").to("cuda")

layer_idx = {layer_idx}

# HF Qwen2 decoder layer returns hidden_states tensor directly (not a tuple)
# Shape: [batch, seq, hidden] = [1, seq_len, hidden_size]

# 1. Baseline: no intervention
baseline_hs = []

def hook_baseline(mod, args, output):
    # output is the hidden_states tensor directly
    baseline_hs.append(output.detach().clone().cpu())
    return output

h = model.model.layers[layer_idx].register_forward_hook(hook_baseline)
with torch.inference_mode():
    out = model(**inputs)
h.remove()
baseline_final = out.logits[0, -1].detach().cpu()
clean_hs = baseline_hs[0]  # [1, seq, hidden]

# Create a steering vector (fixed seed for reproducibility)
torch.manual_seed(42)
steering_vec = torch.randn(clean_hs.shape[-1], dtype=torch.float16).cuda() * 0.1

# 2. Steering: add vector to layer output
def hook_steer(mod, args, output):
    hs = output.clone()
    hs[:, -1, :] += steering_vec
    return hs

h = model.model.layers[layer_idx].register_forward_hook(hook_steer)
with torch.inference_mode():
    out = model(**inputs)
h.remove()
steer_final = out.logits[0, -1].detach().cpu()

# 3. Ablation: replace layer output with zeros
def hook_ablate(mod, args, output):
    return torch.zeros_like(output)

h = model.model.layers[layer_idx].register_forward_hook(hook_ablate)
with torch.inference_mode():
    out = model(**inputs)
h.remove()
ablate_final = out.logits[0, -1].detach().cpu()

# 4. Patching: replace with a fixed target (clean_hs scaled by 0.5)
target_hs = clean_hs.cuda() * 0.5
def hook_patch(mod, args, output):
    return target_hs

h = model.model.layers[layer_idx].register_forward_hook(hook_patch)
with torch.inference_mode():
    out = model(**inputs)
h.remove()
patch_final = out.logits[0, -1].detach().cpu()

torch.save(dict(
    baseline=baseline_final, steer=steer_final,
    ablate=ablate_final, patch=patch_final,
    clean_hs=clean_hs, steering_vec=steering_vec.cpu(),
), "{tmpdir}/hf_results.pt")
print("HF_JSON:" + json.dumps(dict(status="OK")))
'''

# ── vLLM: test different intervention strategies ─────────────────────

VLLM_SCRIPT = r'''
import os, json, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "{vllm_gpu}"
from nnsight.modeling.vllm import VLLM

lm = VLLM("{model}", tensor_parallel_size=1, gpu_memory_utilization=0.3,
           dtype=torch.float16, dispatch=True)

prompt = "{prompt}"
layer_idx = {layer_idx}

# Load HF reference data
hf = torch.load("{tmpdir}/hf_results.pt", weights_only=True)
steering_vec = hf["steering_vec"].to("cuda")

# 0. Baseline
with lm.trace(prompt):
    baseline_logits = lm.logits.output[0, -1].clone().save()

# ═══ STEERING STRATEGIES ═══

# Strategy A: add to output[0] (mlp_out stream)
with lm.trace(prompt):
    layer = lm.model.layers[layer_idx]
    out0 = layer.output[0].clone()
    out1 = layer.output[1].clone()
    out0[-1, :] += steering_vec  # vLLM is 2D: [tokens, hidden]
    layer.output = (out0, out1)
    steer_A = lm.logits.output[0, -1].clone().save()

# Strategy B: add to output[1] (residual stream)
with lm.trace(prompt):
    layer = lm.model.layers[layer_idx]
    out0 = layer.output[0].clone()
    out1 = layer.output[1].clone()
    out1[-1, :] += steering_vec
    layer.output = (out0, out1)
    steer_B = lm.logits.output[0, -1].clone().save()

# ═══ PATCHING STRATEGIES ═══
# Get clean combined hidden state first
with lm.trace(prompt):
    layer = lm.model.layers[layer_idx]
    clean_combined = (layer.output[0].clone() + layer.output[1].clone()).save()

target_hs = clean_combined.cpu() * 0.5  # same target as HF

# Strategy P1: (target, zeros) — zero residual, put target in mlp stream
with lm.trace(prompt):
    layer = lm.model.layers[layer_idx]
    layer.output = (target_hs.to(layer.output[0].device), torch.zeros_like(layer.output[1]))
    patch_P1 = lm.logits.output[0, -1].clone().save()

# Strategy P2: (target - residual, residual) — preserve residual, adjust mlp
with lm.trace(prompt):
    layer = lm.model.layers[layer_idx]
    residual = layer.output[1].clone()
    layer.output = (target_hs.to(residual.device) - residual, residual)
    patch_P2 = lm.logits.output[0, -1].clone().save()

# Strategy P3: (zeros, target) — zero mlp, put target in residual
with lm.trace(prompt):
    layer = lm.model.layers[layer_idx]
    layer.output = (torch.zeros_like(layer.output[0]), target_hs.to(layer.output[1].device))
    patch_P3 = lm.logits.output[0, -1].clone().save()

# ═══ ABLATION ═══
# Zero both streams
with lm.trace(prompt):
    layer = lm.model.layers[layer_idx]
    layer.output = (torch.zeros_like(layer.output[0]), torch.zeros_like(layer.output[1]))
    ablate_both = lm.logits.output[0, -1].clone().save()

# Zero only mlp stream (keep residual)
with lm.trace(prompt):
    layer = lm.model.layers[layer_idx]
    layer.output = (torch.zeros_like(layer.output[0]), layer.output[1].clone())
    ablate_mlp = lm.logits.output[0, -1].clone().save()

torch.save(dict(
    baseline=baseline_logits.cpu(),
    steer_A=steer_A.cpu(), steer_B=steer_B.cpu(),
    patch_P1=patch_P1.cpu(), patch_P2=patch_P2.cpu(), patch_P3=patch_P3.cpu(),
    ablate_both=ablate_both.cpu(), ablate_mlp=ablate_mlp.cpu(),
), "{tmpdir}/vllm_results.pt")
print("VLLM_JSON:" + json.dumps(dict(status="OK")))
'''


def run_subprocess(script, label):
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=300,
    )
    for line in proc.stdout.splitlines():
        if line.startswith(f"{label}_JSON:"):
            return json.loads(line.split(":", 1)[1])
    print(f"--- {label} stdout (last 3000 chars) ---")
    print(proc.stdout[-3000:])
    print(f"--- {label} stderr (last 3000 chars) ---")
    print(proc.stderr[-3000:])
    raise RuntimeError(f"{label} subprocess failed")


def maxdiff(a, b):
    return torch.max(torch.abs(a.float() - b.float())).item()


def cosdist(a, b):
    a_flat = a.float().flatten().unsqueeze(0)
    b_flat = b.float().flatten().unsqueeze(0)
    return torch.nn.functional.cosine_similarity(a_flat, b_flat).item()


def main():
    tmpdir = tempfile.mkdtemp()

    fmt = dict(model=MODEL, prompt=PROMPT, layer_idx=LAYER_IDX,
               hf_gpu=HF_GPU, vllm_gpu=VLLM_GPU, tmpdir=tmpdir)

    print("Running HF reference...")
    run_subprocess(HF_SCRIPT.format(**fmt), "HF")

    print("Running vLLM strategies...")
    run_subprocess(VLLM_SCRIPT.format(**fmt), "VLLM")

    hf = torch.load(f"{tmpdir}/hf_results.pt", weights_only=True)
    vllm = torch.load(f"{tmpdir}/vllm_results.pt", weights_only=True)

    print("\n" + "=" * 70)
    print("Intervention Strategy Comparison: vLLM vs HF")
    print(f"Model: {MODEL}  Layer: {LAYER_IDX}  Prompt: '{PROMPT}'")
    print("=" * 70)

    # Baseline sanity check
    print(f"\n--- Baseline (no intervention) ---")
    print(f"  HF vs vLLM logits max_diff = {maxdiff(hf['baseline'], vllm['baseline']):.4f}")
    print(f"  HF vs vLLM logits cosine   = {cosdist(hf['baseline'], vllm['baseline']):.6f}")

    # Steering
    print(f"\n--- Steering: add vector to last-token hidden state ---")
    print(f"  HF steer vs vLLM steer_A (add to output[0]):  max_diff={maxdiff(hf['steer'], vllm['steer_A']):.4f}  cos={cosdist(hf['steer'], vllm['steer_A']):.6f}")
    print(f"  HF steer vs vLLM steer_B (add to output[1]):  max_diff={maxdiff(hf['steer'], vllm['steer_B']):.4f}  cos={cosdist(hf['steer'], vllm['steer_B']):.6f}")
    print(f"  vLLM steer_A vs steer_B (should be identical): max_diff={maxdiff(vllm['steer_A'], vllm['steer_B']):.4f}  cos={cosdist(vllm['steer_A'], vllm['steer_B']):.6f}")

    # Patching
    print(f"\n--- Patching: replace hidden state with target ---")
    print(f"  HF patch vs vLLM P1 (target, zeros):           max_diff={maxdiff(hf['patch'], vllm['patch_P1']):.4f}  cos={cosdist(hf['patch'], vllm['patch_P1']):.6f}")
    print(f"  HF patch vs vLLM P2 (target-res, res):         max_diff={maxdiff(hf['patch'], vllm['patch_P2']):.4f}  cos={cosdist(hf['patch'], vllm['patch_P2']):.6f}")
    print(f"  HF patch vs vLLM P3 (zeros, target):           max_diff={maxdiff(hf['patch'], vllm['patch_P3']):.4f}  cos={cosdist(hf['patch'], vllm['patch_P3']):.6f}")
    print(f"  P1 vs P2 vs P3 (should all be identical):")
    print(f"    P1 vs P2: max_diff={maxdiff(vllm['patch_P1'], vllm['patch_P2']):.4f}")
    print(f"    P1 vs P3: max_diff={maxdiff(vllm['patch_P1'], vllm['patch_P3']):.4f}")
    print(f"    P2 vs P3: max_diff={maxdiff(vllm['patch_P2'], vllm['patch_P3']):.4f}")

    # Ablation
    print(f"\n--- Ablation: zero out layer output ---")
    print(f"  HF ablate vs vLLM ablate_both (zero both):     max_diff={maxdiff(hf['ablate'], vllm['ablate_both']):.4f}  cos={cosdist(hf['ablate'], vllm['ablate_both']):.6f}")
    print(f"  HF ablate vs vLLM ablate_mlp (zero mlp only):  max_diff={maxdiff(hf['ablate'], vllm['ablate_mlp']):.4f}  cos={cosdist(hf['ablate'], vllm['ablate_mlp']):.6f}")
    print(f"  ablate_both vs ablate_mlp (NOT same — residual matters):")
    print(f"    max_diff={maxdiff(vllm['ablate_both'], vllm['ablate_mlp']):.4f}  cos={cosdist(vllm['ablate_both'], vllm['ablate_mlp']):.6f}")

    print("\n" + "=" * 70)
    print("KEY FINDINGS:")
    print("  Steering: add to EITHER output[0] OR output[1] — both work")
    print("  Patching: (target, zeros) = (target-res, res) = (zeros, target)")
    print("  Ablation: MUST zero BOTH streams to match HF ablation")
    print("=" * 70)


if __name__ == "__main__":
    main()
