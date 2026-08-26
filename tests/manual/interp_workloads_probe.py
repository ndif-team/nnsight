"""Interp-workload validation for the Megatron backend's current stage.

Runs the forward+backward-with-hooks workloads from interp-serve-bench
(methods: logit_lens, steering, ablation, activation_patching,
attribution_patching) on the Megatron backend and the HF eager oracle with
identical inputs, and compares results. Prompts come from the bench's
datasets: CounterFact (Meng et al. 2022) and MIB IOI (Mueller et al. 2025),
loaded from a local interp-serve-bench checkout when present, with embedded
samples from those datasets as fallback.

Run (in the 0.8 worktree):
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src python tests/manual/interp_workloads_probe.py
"""

import json
import os

import torch
import torch.nn.functional as F

from nnsight.modeling.megatron import MegatronLM
from nnsight.modeling.transformers import TransformersModel

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
BENCH_DATA = os.path.expanduser("~/interp-serve-bench/data")
LAYER = 12
N_ITEMS = 4

# Samples from the bench datasets, used only when the checkout is absent.
COUNTERFACT_FALLBACK = ["The mother tongue of Danielle Darrieux is"]
IOI_FALLBACK = [
    {
        "clean": "Then, Henry and Phil had a lot of fun at the harbor. Henry gave a basket to",
        "corrupted": "Then, Henry and Phil had a lot of fun at the harbor. Phil gave a basket to",
        "answers": [" Phil", " Henry"],
    }
]


def load_data():
    try:
        with open(f"{BENCH_DATA}/counterfact/counterfact.json") as f:
            cf = [item["prompt"] for item in json.load(f)["items"][:N_ITEMS]]
        with open(f"{BENCH_DATA}/mib/ioi.json") as f:
            ioi = json.load(f)["items"][:N_ITEMS]
    except FileNotFoundError:
        cf, ioi = COUNTERFACT_FALLBACK, IOI_FALLBACK
    return cf, ioi


def first(x):
    return x[0] if isinstance(x, tuple) else x


def bsh(t: torch.Tensor) -> torch.Tensor:
    """[batch, seq, hidden] view regardless of stack layout."""
    t = t.float()
    return t.transpose(0, 1) if t.dim() == 3 and t.shape[1] == 1 and t.shape[0] != 1 else t


class Stack:
    """One backend with the module paths the workloads touch."""

    def __init__(self, model, layers, unembed_weight, logits_of):
        self.model = model
        self.layers = layers          # fn(l) -> layer envoy
        self.unembed = unembed_weight  # [vocab, hidden] tensor
        self.logits_of = logits_of    # fn() -> logits value inside trace


def hf_stack():
    m = TransformersModel(
        REPO, task="text-generation", dispatch=True, dtype=torch.float32,
        attn_implementation="eager", device_map="cuda",
    )
    m._module.requires_grad_(False)
    return Stack(
        m,
        layers=lambda l: m.model.layers[l],
        unembed_weight=m._module.lm_head.weight.detach(),
        logits_of=lambda: m.output.logits,
    )


def mm_stack():
    m = MegatronLM(REPO, dispatch=True, dtype=torch.float32)
    m._module.requires_grad_(False)
    return Stack(
        m,
        layers=lambda l: m.gpt.decoder.layers[l],
        unembed_weight=m._module.gpt.embedding.word_embeddings.weight.detach(),
        logits_of=lambda: m.output,
    )


def logit_lens(stack, prompts):
    """Per-layer last-token logits via the portable unembed (matmul form)."""
    out = []
    final_norm_w = None
    for prompt in prompts:
        with stack.model.trace(prompt):
            rows = []
            for l in range(24):
                h = bsh(first(stack.layers(l).output))[:, -1]
                normed = h / h.norm(dim=-1, keepdim=True) * (h.shape[-1] ** 0.5)
                rows.append(F.linear(normed.float(), stack.unembed.float()))
            lens = torch.stack(rows).save()
        out.append(lens.cpu())
    return torch.stack(out)


def steering(stack, prompts, target=" Rome", alpha=6.0):
    """Add alpha * ||h|| * normalize(W_U[target]) at LAYER, read final logits."""
    tid = stack.model.tokenizer(target, add_special_tokens=False)["input_ids"][0]
    direction = F.normalize(stack.unembed[tid].float(), dim=-1)
    out = []
    for prompt in prompts:
        with stack.model.trace(prompt):
            layer = stack.layers(LAYER)
            o = layer.output
            h = first(o)
            vec = (alpha * h.norm(dim=-1, keepdim=True) * direction.to(h.dtype)).to(h.device)
            steered = h + vec
            layer.output = (steered, *o[1:]) if isinstance(o, tuple) else steered
            logits = stack.logits_of().save()
        out.append(bsh(logits)[:, -1].cpu())
    return torch.stack(out)


def ablation(stack, prompts):
    """Zero the mlp output at LAYER, read final logits."""
    out = []
    for prompt in prompts:
        with stack.model.trace(prompt):
            mlp = stack.layers(LAYER).mlp
            o = mlp.output
            zeroed = first(o) * 0
            mlp.output = (zeroed, *o[1:]) if isinstance(o, tuple) else zeroed
            logits = stack.logits_of().save()
        out.append(bsh(logits)[:, -1].cpu())
    return torch.stack(out)


def activation_patching(stack, items):
    """Two sequential traces: clean residual at LAYER transplanted into corrupted."""
    out = []
    for item in items:
        with stack.model.trace(item["clean"]):
            clean_h = first(stack.layers(LAYER).output).save()
        with stack.model.trace(item["corrupted"]):
            layer = stack.layers(LAYER)
            o = layer.output
            layer.output = (clean_h, *o[1:]) if isinstance(o, tuple) else clean_h
            logits = stack.logits_of().save()
        out.append(bsh(logits)[:, -1].cpu())
    return torch.stack(out)


def attribution_patching(stack, items):
    """Bench-canonical gradient workload: attribution[L] =
    ((a_clean[L] - a_corrupt[L]) * dM/da_corrupt[L]).sum(), M = logit diff."""
    tok = stack.model.tokenizer
    out = []
    for item in items:
        correct = tok(item["answers"][0], add_special_tokens=False)["input_ids"][0]
        incorrect = tok(item["answers"][1], add_special_tokens=False)["input_ids"][0]
        # Mutated inside the traces; a pre-created container propagates by
        # reference regardless of frame-locals push behavior in nested blocks.
        store = {"clean": [], "corrupt": [], "grads": []}
        with stack.model.trace(item["clean"]):
            store["clean"].extend(first(stack.layers(l).output).save() for l in range(24))
        with stack.model.trace(item["corrupted"]):
            taps = []
            for l in range(24):
                h = first(stack.layers(l).output)
                if not h.requires_grad:
                    h.requires_grad_(True)
                taps.append(h)
            store["corrupt"].extend(t.save() for t in taps)
            logits = bsh(stack.logits_of())
            metric = logits[:, -1, correct] - logits[:, -1, incorrect]
            with metric.sum().backward():
                store["grads"].extend(taps[l].grad.save() for l in reversed(range(24)))
        grads = list(reversed(store["grads"]))
        attribution = torch.stack(
            [
                (
                    (store["clean"][l].float() - store["corrupt"][l].float())
                    * grads[l].float()
                ).sum().cpu()
                for l in range(24)
            ]
        )
        out.append(attribution)
    return torch.stack(out)


def compare(name, ref, got, rel_tol):
    rel = ((got - ref).norm() / ref.norm().clamp(min=1e-12)).item()
    argmax_note = ""
    if ref.dim() >= 2 and ref.shape[-1] > 1000:  # logits-like: also check argmax
        agree = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
        argmax_note = f", argmax agree {agree:.3f}"
    status = "PASS" if rel < rel_tol else "FAIL"
    print(f"{status} {name}: rel_fro {rel:.2e}{argmax_note}, shape {tuple(ref.shape)}")
    return rel < rel_tol


def main():
    cf, ioi = load_data()
    print(f"data: {len(cf)} CounterFact prompts, {len(ioi)} MIB IOI items")

    results = {}
    for name, build in [("hf", hf_stack), ("megatron", mm_stack)]:
        stack = build()
        results[name] = dict(
            logit_lens=logit_lens(stack, cf),
            steering=steering(stack, cf),
            ablation=ablation(stack, [i["clean"] for i in ioi]),
            activation_patching=activation_patching(stack, ioi),
            attribution_patching=attribution_patching(stack, ioi),
        )
        del stack
        torch.cuda.empty_cache()

    tol = dict(
        logit_lens=1e-4, steering=1e-4, ablation=1e-4,
        activation_patching=1e-4, attribution_patching=1e-3,
    )
    ok = all(
        compare(name, results["hf"][name], results["megatron"][name], tol[name])
        for name in results["hf"]
    )
    print("ALL PASS" if ok else "FAILED")


if __name__ == "__main__":
    main()
