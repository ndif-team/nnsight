"""One rank of the expert-parallel test, run under ``torch.distributed.run``.

Not a test module (no ``test_`` prefix, so pytest does not collect it):
`test_cpu_expert_parallel.py` launches this once in a single process for the
reference and once across N ranks, then compares.

Expert parallelism is a different axis from tensor parallelism. It is asked for
with ``enable_expert_parallel=True``, and transformers answers by applying the
model's *expert* plan — ``base_model_ep_plan``, which distributes whole experts
across ranks — instead of its tensor-parallel one. ``model.tp_plan`` resolves to
whichever is in force, which is the only thing nnsight has to read.

Every rank runs the identical block, so nothing here may branch on rank.
"""

from __future__ import annotations

import argparse
import os

import torch

import nnsight

# Tiny, MoE, and its plan uses the three expert-parallel styles: `ep_router` on
# the router, `grouped_gemm` on the expert parameters, `moe_tp_experts` on the
# experts module.
REPO = "hf-internal-testing/tiny-random-GptOssForCausalLM"

# Below the tiny model's vocabulary, so no tokenizer is needed and the ids are
# identical on every rank by construction.
INPUT_IDS = [[1, 2, 3, 4, 5, 6]]
LAYER = 0


def build(ep: int):
    from nnsight.modeling.tp import TPFragments
    from nnsight.modeling.transformers import TransformersModel

    kwargs = {}
    if ep > 1:
        from transformers.distributed import DistributedConfig

        kwargs["distributed_config"] = DistributedConfig(
            tp_size=ep, enable_expert_parallel=True
        )
    else:
        kwargs["device_map"] = {"": "cpu"}

    model = TransformersModel(
        REPO, task="text-generation", dispatch=True, dtype=torch.float32, **kwargs
    )

    fragments = model.interleaver.fragments
    assert isinstance(fragments, TPFragments), type(fragments)
    if ep > 1:
        assert fragments.enabled, "expert-parallel model did not enable the TP path"
        assert fragments.tp_rules, "expert-parallel model recorded no rules"
    else:
        assert not fragments.enabled, "unsharded model enabled the TP path"

    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    model = build(args.ep)
    layer = model.model.layers[LAYER]
    results: dict[str, torch.Tensor] = {}

    def record(name: str, tensor: torch.Tensor) -> None:
        results[name] = tensor.detach().float().cpu()

    ids = torch.tensor(INPUT_IDS)
    with model.trace({"input_ids": ids}):
        # The router is replicated: every rank computes the same thing, and the
        # masking that makes it rank-specific happens in its own post-transform,
        # after the handoff. So this must match the single-process run exactly.
        record("router_logits", layer.mlp.router.output[0].save())
        # The experts module holds only this rank's experts and produces its term
        # of the sum; a worker must be shown the sum.
        record("experts_out", layer.mlp.experts.output.save())
        record("mlp_out", layer.mlp.output[0].save())
        record("logits", model.lm_head.output.save())

    # An edit on the summed expert output has to survive the way back.
    with model.trace({"input_ids": ids}):
        layer.mlp.experts.output = layer.mlp.experts.output * 0.5
        record("edited_logits", model.lm_head.output.save())

    torch.save(results, os.path.join(args.out, f"rank{rank}.pt"))


if __name__ == "__main__":
    main()
