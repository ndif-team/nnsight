"""Stage 2, tier 1: the bench workloads on a tensor-parallel HF model.

Runs the five forward+backward-with-hooks workloads of
interp_workloads_probe.py on a TransformersModel sharded with transformers'
native tensor parallelism (reassembled at hook points by nnsight's
TPFragments), and compares against the unsharded single-GPU run. Every rank
runs the identical block; rank 0 writes results.

Reference:  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=<tf515 overlay>:src \\
              python tests/manual/tp_workloads_probe.py --tp 1 --out ref.pt
TP=2:       CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=<tf515 overlay>:src \\
              torchrun --nproc_per_node=2 tests/manual/tp_workloads_probe.py --tp 2 --out tp2.pt
Compare:    python tests/manual/tp_workloads_probe.py --compare ref.pt tp2.pt
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interp_workloads_probe import (  # noqa: E402
    REPO, Stack, ablation, activation_patching, attribution_patching,
    compare, load_data, logit_lens, steering,
)


def full(weight: torch.Tensor, rows: int) -> torch.Tensor:
    """A sharded parameter as one whole [rows, hidden] tensor on every rank.

    transformers' TP keeps the local shard as a plain tensor (not a DTensor),
    so a rank-local read of ``lm_head.weight`` is half the vocabulary at tp=2.
    A workload that derives anything from a parameter (the steering direction,
    the portable unembed) must reassemble it first, or the ranks silently
    diverge on their intervention and the collectives mix the results.
    """
    weight = weight.detach()
    if isinstance(weight, torch.distributed.tensor.DTensor):
        return weight.full_tensor()
    if weight.shape[0] == rows or not torch.distributed.is_initialized():
        return weight
    world = torch.distributed.get_world_size()
    shards = [torch.empty_like(weight) for _ in range(world)]
    torch.distributed.all_gather(shards, weight.contiguous())
    gathered = torch.cat(shards, dim=0)
    assert gathered.shape[0] == rows, (gathered.shape, rows)
    return gathered


def build(tp: int, repo: str) -> Stack:
    from nnsight.modeling.transformers import TransformersModel

    if tp > 1:
        from transformers.distributed import DistributedConfig

        m = TransformersModel(
            repo, task="text-generation", dispatch=True, dtype=torch.float32,
            attn_implementation="eager",
            distributed_config=DistributedConfig(tp_size=tp),
        )
        assert m.interleaver.fragments.enabled, "TP model did not enable TPFragments"
    else:
        m = TransformersModel(
            repo, task="text-generation", dispatch=True, dtype=torch.float32,
            attn_implementation="eager", device_map={"": 0},
        )
    m._module.requires_grad_(False)
    return Stack(
        m,
        layers=lambda l: m.model.layers[l],
        unembed_weight=full(m._module.lm_head.weight, m._module.config.vocab_size),
        logits_of=lambda: m.output.logits,
    )


def moe_internals(stack, prompts):
    """MoE-internal hook points under expert sharding: router output and the
    experts module's combined output (the values the bench workloads never
    touch, and the ones HF's ep_router / moe_tp_experts styles shard)."""
    routers, experts = [], []
    for prompt in prompts:
        with stack.model.trace(prompt):
            mlp = stack.layers(12).mlp
            r = mlp.gate.output
            routers.append((r[1] if isinstance(r, tuple) else r).save())
            experts.append(mlp.experts.output.save())
    return dict(
        router=torch.cat([r.float().cpu() for r in routers]),
        experts=torch.cat([e.float().cpu() for e in experts]),
    )


def run(tp: int, out: str, repo: str) -> None:
    cf, ioi = load_data()
    stack = build(tp, repo)
    results = dict(
        logit_lens=logit_lens(stack, cf),
        steering=steering(stack, cf),
        ablation=ablation(stack, [i["clean"] for i in ioi]),
        activation_patching=activation_patching(stack, ioi),
        attribution_patching=attribution_patching(stack, ioi),
    )
    if hasattr(stack.layers(12).mlp._module, "experts"):
        internals = moe_internals(stack, cf[:2])
        results["moe_router"] = internals["router"]
        results["moe_experts_output"] = internals["experts"]
    if int(os.environ.get("RANK", 0)) == 0:
        torch.save(results, out)
        print(f"wrote {out}")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def compare_files(ref_path: str, got_path: str) -> None:
    ref, got = torch.load(ref_path), torch.load(got_path)
    tol = dict(
        logit_lens=1e-4, steering=1e-4, ablation=1e-4,
        activation_patching=1e-4, attribution_patching=1e-3,
        moe_router=1e-4, moe_experts_output=1e-4,
    )
    ok = all(compare(name, ref[name], got[name], tol[name]) for name in ref)
    print("ALL PASS" if ok else "FAILED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--out")
    parser.add_argument("--compare", nargs=2, metavar=("REF", "GOT"))
    parser.add_argument("--repo", default=REPO)
    args = parser.parse_args()
    if args.compare:
        compare_files(*args.compare)
    else:
        run(args.tp, args.out, args.repo)


if __name__ == "__main__":
    main()
