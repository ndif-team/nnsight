"""Minimal J-lens collection demo: the v0 Megatron backend's acceptance test.

J-lens transport matrices are J_l = E[d h_final / d h_l] averaged over prompts
and positions (transformer-circuits.pub/2026/workspace). This demo collects a
K-seed sketch of J_l (K random directions v: each backward from
(h_final * v).sum() yields the position-summed row v^T J_l at every tapped
layer at once) on both the HF eager stack and the Megatron backend, with
identical seeds, and compares the resulting sketches.

Exercises, per prompt: one grad-enabled forward, K sequential backward passes
against one retained graph, reverse-order .grad reads at three tapped layers,
cross-prompt accumulation. This is exactly the collection loop the backend
exists for, at demo scale (d sketched to K=16, 8 prompts, 3 layers).

Run (in the 0.8 worktree):
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src python tests/manual/jlens_demo.py
"""

import torch

from nnsight.modeling.megatron import MegatronLM
from nnsight.modeling.transformers import TransformersModel

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
LAYERS = [6, 12, 18]  # tapped; final residual = last decoder layer's output
SEEDS = 16
PROMPTS = [
    "The quick brown fox jumps over the lazy dog",
    "Paris is the capital of France",
    "The derivative of x squared is",
    "In 1969, humans first landed on",
    "Water is composed of hydrogen and",
    "The stock market fell sharply today",
    "She opened the door and saw",
    "Photosynthesis converts sunlight into",
]


def first(x):
    return x[0] if isinstance(x, tuple) else x


def bsh(t: torch.Tensor) -> torch.Tensor:
    """Position-major [s, h] view of one prompt's activation grad."""
    t = t.float()
    if t.dim() == 3 and t.shape[1] == 1:   # mcore [s, 1, h]
        return t[:, 0]
    return t[0]                            # HF [1, s, h]


def collect(model, layer_of, final_of, directions: torch.Tensor) -> dict:
    """K-seed J-lens sketch per tapped layer: {layer: [K, d] tensor}."""
    sketch = {l: torch.zeros(directions.shape, dtype=torch.float64) for l in LAYERS}
    for prompt in PROMPTS:
        with model.trace(prompt):
            taps = {}
            for l in LAYERS:
                h = first(layer_of(model, l))
                if not h.requires_grad:      # earliest tap is the leaf; rest inherit
                    h.requires_grad_(True)
                taps[l] = h
            h_final = first(final_of(model))
            v = directions.to(h_final.device, h_final.dtype)
            for k in range(SEEDS):
                # v_k broadcasts over positions: the loss sums v_k . h_final,t'
                # over all t', so causality restricts each source position t to
                # contributions from t' >= t, matching the paper's expectation.
                loss = (h_final * v[k]).sum()
                with loss.backward(retain_graph=True):
                    # reverse forward order: deepest tapped layer first
                    grads = {l: taps[l].grad.save() for l in reversed(LAYERS)}
                for l in LAYERS:
                    sketch[l][k] += bsh(grads[l]).sum(0).double().cpu()
    return sketch


def main():
    torch.manual_seed(0)
    directions = torch.randn(SEEDS, 896)
    directions /= directions.norm(dim=-1, keepdim=True)

    hf = TransformersModel(
        REPO, task="text-generation", dispatch=True, dtype=torch.float32,
        attn_implementation="eager", device_map="cuda",
    )
    hf._module.requires_grad_(False)
    ref = collect(
        hf,
        layer_of=lambda m, l: m.model.layers[l].output,
        final_of=lambda m: m.model.layers[-1].output,
        directions=directions,
    )
    del hf
    torch.cuda.empty_cache()

    mm = MegatronLM(REPO, dispatch=True, dtype=torch.float32)
    mm._module.requires_grad_(False)
    got = collect(
        mm,
        layer_of=lambda m, l: m.gpt.decoder.layers[l].output,
        final_of=lambda m: m.gpt.decoder.layers[-1].output,
        directions=directions,
    )

    ok = True
    for l in LAYERS:
        r, g = ref[l], got[l]
        rel_max = ((g - r).abs().max() / r.abs().max()).item()
        rel_fro = ((g - r).norm() / r.norm()).item()
        cos = torch.nn.functional.cosine_similarity(g, r, dim=-1).min().item()
        passed = rel_fro < 1e-4 and cos > 0.9999
        ok &= passed
        print(
            f"{'PASS' if passed else 'FAIL'} layer {l}: sketch {tuple(r.shape)}, "
            f"rel_fro {rel_fro:.2e}, rel_max {rel_max:.2e}, cos_min {cos:.6f}"
        )
    print("ALL PASS" if ok else "FAILED")


if __name__ == "__main__":
    main()
