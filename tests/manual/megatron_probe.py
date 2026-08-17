"""Megatron backend correctness probe vs the HF LanguageModel oracle.

Runs Qwen2.5-0.5B-Instruct through both wrappers on one GPU and compares, in
dependency order (fail fast at the first broken layer):

  1. layer-0 input_layernorm output and attention output   (qkv interleave / RoPE)
  2. layer-12 block output
  3. final logits + argmax agreement
  4. activation-grad from two backward seeds (params frozen)
  5. .source smoke on an mcore attention module
  6. two-invoke batched trace: per-invoke saves equal solo runs
  7. swap: zeroing invoke 0's slice leaves invoke 1's logits unchanged

Run:
  CUDA_VISIBLE_DEVICES=4 python tests/manual/megatron_probe.py [--dtype fp32|bf16]
"""

import argparse

import torch

from nnsight.modeling.transformers import TransformersModel
from nnsight.modeling.megatron import MegatronLM

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "The quick brown fox jumps over the lazy dog"
PROMPT2 = "Hello"
LAYER = 12

# The oracle is ALWAYS the fp32 HF model. Control measurement (this machine,
# this prompt): HF-bf16 itself is cos_min 0.9937 / 88.9% argmax vs HF-fp32,
# while mcore-bf16 is cos_min 0.9953 / 100% argmax vs HF-fp32 - a same-dtype
# bf16 oracle sits inside its own noise floor, so bf16 gates compare against
# fp32 truth. fp32 grad gates use relative error (Qwen grads are large-magnitude).
TOL = {
    "fp32": dict(act=5e-4, logits=2e-3, grad=1e-4, argmax=1.0, cos=None),
    # bf16 grad gate calibrated to the measured noise floor: HF-bf16 grads are
    # themselves rel 1.65e-1 from HF-fp32 grads (mcore-bf16 measured 1.25e-1).
    "bf16": dict(act=5e-2, logits=None, grad=2e-1, argmax=0.95, cos=0.99),
}


def sbh_to_bsh(t: torch.Tensor) -> torch.Tensor:
    return t.transpose(0, 1) if t.dim() == 3 else t


def first(x):
    """Decoder layers return tuples in both stacks (HF tf5: (hidden, ...);
    mcore TransformerLayer: (hidden, context)); unpack to the hidden tensor."""
    return x[0] if isinstance(x, tuple) else x


def report(name, got, ref, tol, transpose=False, relative=False, gate=True):
    if transpose:
        got = sbh_to_bsh(got)
    assert got.shape == ref.shape, f"{name}: shape {tuple(got.shape)} vs {tuple(ref.shape)}"
    diff = (got.float() - ref.float()).abs().max().item()
    rel = diff / ref.float().abs().max().clamp(min=1e-12).item()
    metric = rel if relative else diff
    ok = metric < tol
    status = ("PASS" if ok else "FAIL") if gate else "INFO"
    print(f"{status} {name}: max|d| {diff:.2e} (rel {rel:.2e}), shape {tuple(ref.shape)}")
    if gate:
        assert ok, f"{name}: {'rel' if relative else 'abs'} diff {metric} > {tol}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    args = parser.parse_args()
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    tol = TOL[args.dtype]
    rel = args.dtype == "bf16"  # bf16 magnitudes (Qwen massive activations) make absolute gates meaningless

    hf = TransformersModel(
        REPO, task="text-generation", dispatch=True, dtype=torch.float32,
        attn_implementation="eager", device_map="cuda",
    )
    hf._module.requires_grad_(False)
    mm = MegatronLM(REPO, dispatch=True, dtype=dtype)
    mm._module.requires_grad_(False)

    # oracle traces
    with hf.trace(PROMPT):
        r_ln0 = hf.model.layers[0].input_layernorm.output.save()
        r_attn0 = hf.model.layers[0].self_attn.output.save()
        r_mid = hf.model.layers[LAYER].output.save()
        r_logits = hf.output.logits.save()
    with hf.trace(PROMPT):
        h = first(hf.model.layers[LAYER].output)
        h.requires_grad_(True)
        logits = hf.output.logits
        with logits[:, -1].sum().backward():
            r_g1 = h.grad.save()
    with hf.trace(PROMPT):
        h = first(hf.model.layers[LAYER].output)
        h.requires_grad_(True)
        logits = hf.output.logits
        with (logits[:, 0] * 2).sum().backward():
            r_g2 = h.grad.save()

    # subject traces
    with mm.trace(PROMPT):
        m_ln0 = mm.gpt.decoder.layers[0].input_layernorm.output.save()
        m_attn0 = mm.gpt.decoder.layers[0].self_attention.output.save()
        m_mid = mm.gpt.decoder.layers[LAYER].output.save()
        m_logits = mm.output.save()
    with mm.trace(PROMPT):
        h = first(mm.gpt.decoder.layers[LAYER].output)
        h.requires_grad_(True)
        logits = mm.output
        with logits[:, -1].sum().backward():
            m_g1 = h.grad.save()
    with mm.trace(PROMPT):
        h = first(mm.gpt.decoder.layers[LAYER].output)
        h.requires_grad_(True)
        logits = mm.output
        with (logits[:, 0] * 2).sum().backward():
            m_g2 = h.grad.save()

    hf_attn0 = r_attn0[0] if isinstance(r_attn0, tuple) else r_attn0
    mc_attn0 = m_attn0[0] if isinstance(m_attn0, tuple) else m_attn0
    hf_mid = r_mid[0] if isinstance(r_mid, tuple) else r_mid
    mc_mid = m_mid[0] if isinstance(m_mid, tuple) else m_mid

    report("layer-0 input_layernorm", m_ln0, r_ln0, tol["act"], transpose=True, relative=rel)
    report("layer-0 attention output", mc_attn0, hf_attn0, tol["act"], transpose=True, relative=rel)
    report(f"layer-{LAYER} output", mc_mid, hf_mid, tol["act"], transpose=True, relative=rel)
    # bf16 logits are gated by cosine + argmax below; magnitudes are reporting only
    report("logits", m_logits, r_logits, tol["logits"] or 0, relative=rel, gate=not rel)
    if tol["cos"] is not None:
        cos = torch.nn.functional.cosine_similarity(
            m_logits.float().flatten(0, 1), r_logits.float().flatten(0, 1), dim=-1
        ).min().item()
        status = "PASS" if cos > tol["cos"] else "FAIL"
        print(f"{status} logits cosine similarity vs fp32 (min over positions): {cos:.6f}")
        assert cos > tol["cos"]

    agree = (m_logits.float().argmax(-1) == r_logits.float().argmax(-1)).float().mean().item()
    status = "PASS" if agree >= tol["argmax"] else "FAIL"
    print(f"{status} argmax agreement: {agree:.4f}")
    assert agree >= tol["argmax"]

    report("grad seed 1", sbh_to_bsh(m_g1), r_g1, tol["grad"], relative=True)
    report("grad seed 2", sbh_to_bsh(m_g2), r_g2, tol["grad"], relative=True)

    src = mm.gpt.decoder.layers[0].self_attention.source
    assert len(str(src)) > 0
    print(f"PASS .source resolves ({len(str(src).splitlines())} op lines)")

    # batched two-invoke trace vs solo runs
    with mm.trace(PROMPT2):
        solo2_mid = first(mm.gpt.decoder.layers[LAYER].output).save()
        solo2_logits = mm.output.save()
    with mm.trace() as tracer:
        with tracer.invoke(PROMPT):
            b_mid1 = first(mm.gpt.decoder.layers[LAYER].output).save()
        with tracer.invoke(PROMPT2):
            b_mid2 = first(mm.gpt.decoder.layers[LAYER].output).save()
    assert b_mid1.shape[1] == 1 and b_mid2.shape[1] == 1, (
        f"per-invoke batch sizes: {b_mid1.shape}, {b_mid2.shape}"
    )
    report("batched invoke 0 vs solo", b_mid1, mc_mid, tol["act"], relative=rel)
    n2 = solo2_mid.shape[0]
    report("batched invoke 1 vs solo", b_mid2[:n2], solo2_mid, tol["act"], relative=rel)

    # swap: zero invoke 0's slice; invoke 1's logits must match its solo run
    with mm.trace() as tracer:
        with tracer.invoke(PROMPT):
            out = mm.gpt.decoder.layers[LAYER].output
            mm.gpt.decoder.layers[LAYER].output = (first(out) * 0, *out[1:]) if isinstance(out, tuple) else out * 0
        with tracer.invoke(PROMPT2):
            s_logits2 = mm.output.save()
    # same-dtype mcore-vs-mcore comparison: act tolerance applies on the bf16 path
    report("swap isolation (invoke 1 logits)", s_logits2[:, :solo2_logits.shape[1]], solo2_logits, tol["logits"] or tol["act"], relative=rel)

    print("ALL PASS")


if __name__ == "__main__":
    main()
