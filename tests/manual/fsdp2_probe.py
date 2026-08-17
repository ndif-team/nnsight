"""FSDP2 + nnsight compatibility probe.

Runs a HF model under torch.distributed.fsdp.fully_shard on 2 GPUs and checks,
against an unsharded single-GPU reference on the same rank:

  1. hooked activations are whole tensors with correct values
  2. .source on a sharded module still resolves (AST rewrite of forward)
  3. activation-grad backward (requires_grad_ on activation, params frozen)
  4. a second backward seed via a second forward
  5. (MoE) router logits and one expert's routed output match the reference

Run:
  CUDA_VISIBLE_DEVICES=4,7 torchrun --nproc_per_node=2 tests/manual/fsdp2_probe.py [--model qwen-moe]
"""

import argparse

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM, AutoTokenizer

from nnsight import LanguageModel

CONFIGS = {
    "gpt2": dict(
        model_id="openai-community/gpt2",
        dtype=torch.float32,
        atol=1e-4,
        blocks=lambda m: m.transformer.h,
        layer=lambda lm: lm.transformer.h[6],
        source_module=lambda lm: lm.transformer.h[0].attn,
        moe=None,
    ),
    "qwen-moe": dict(
        model_id="Qwen/Qwen1.5-MoE-A2.7B",
        dtype=torch.bfloat16,
        atol=1e-2,  # bf16 + MoE combine (index_add atomics) is not run-to-run bitwise
        blocks=lambda m: m.model.layers,
        layer=lambda lm: lm.model.layers[12],
        # tf 5.x: experts is ONE module (stacked weights, Python loop); no
        # per-expert submodules, so per-expert visibility is via .source on it.
        source_module=lambda lm: lm.model.layers[0].mlp.experts,
        moe=dict(
            gate=lambda lm: lm.model.layers[12].mlp.gate,
            experts=lambda lm: lm.model.layers[12].mlp.experts,
        ),
    ),
}

PROMPT = "The quick brown fox jumps over the lazy dog"


def build(cfg, sharded: bool):
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=cfg["dtype"])
    model.requires_grad_(False)  # mimic NDIF: frozen weights, autograd live
    if sharded:
        for block in cfg["blocks"](model):
            fully_shard(block)
        fully_shard(model)
    else:
        model.cuda()
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return LanguageModel(model, tokenizer=tok)


def run_traces(cfg, lm):
    out = {}
    # One backward per forward: FSDP2 frees gathered params after its
    # post-backward, so a retain_graph second backward hits freed storage.
    with lm.trace(PROMPT):
        if cfg["moe"]:
            # forward-pass order: mlp internals fire before the layer's output
            out["routing weights"] = cfg["moe"]["gate"](lm).output[1].save()
            out["experts combined output"] = cfg["moe"]["experts"](lm).output.save()
        h = cfg["layer"](lm).output
        h.requires_grad_(True)
        out["activation"] = h.save()
        logits = lm.lm_head.output
        with logits[:, -1].sum().backward():
            out["grad seed 1"] = h.grad.save()
    with lm.trace(PROMPT):
        h = cfg["layer"](lm).output
        h.requires_grad_(True)
        logits = lm.lm_head.output
        with (logits[:, 0] * 2).sum().backward():
            out["grad seed 2 (second forward)"] = h.grad.save()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", choices=CONFIGS)
    cfg = CONFIGS[parser.parse_args().model]

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    ref = run_traces(cfg, build(cfg, sharded=False))
    got = run_traces(cfg, build(cfg, sharded=True))

    for name, r in ref.items():
        g = got[name]
        assert not isinstance(g, torch.distributed.tensor.DTensor), f"{name}: DTensor leaked to hook"
        assert g.shape == r.shape, f"{name}: shape {g.shape} vs {r.shape}"
        diff = (g.float() - r.float()).abs().max().item()
        assert diff < cfg["atol"], f"{name}: max diff {diff}"
        if rank == 0:
            print(f"PASS {name}: shape {tuple(g.shape)}, max diff {diff:.2e}")

    lm = build(cfg, sharded=True)
    src = cfg["source_module"](lm).source
    assert len(str(src)) > 0
    if rank == 0:
        print(f"PASS .source resolves on sharded module ({len(str(src).splitlines())} op lines)")

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("ALL PASS")


if __name__ == "__main__":
    main()
