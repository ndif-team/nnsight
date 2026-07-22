"""Worker for the MoE deferred-reduce batching tests (run in a subprocess).

Qwen-MoE-family models build their fused-experts module with
``reduce_results=False``: the module returns per-rank PARTIAL sums (the
shared-expert and routed-experts contributions of this rank only) and the
outer block all-reduces three lines later. Intervention code reading or
swapping ``mlp.experts.output`` must therefore see/produce the FULL value,
exactly like the existing ``RowParallelLinear`` handling.

Checks (result JSON written to --out):

  read_cos / read_max_delta:
      In one trace, save ``experts.output`` (tuple: shared, routed) and the
      block's own ``mlp.output``. The block output IS the all-reduce of the
      per-rank sums, so ``sum(experts.output) == mlp.output`` iff the read
      was gathered. Ungathered tp0 partials give cos ~0.6-0.93.

  write_max_delta:
      Swap ``experts.output`` with two constant tensors (a, b) and save the
      block output; the block computes ``all_reduce(a' + b')`` where a', b'
      are what the write-back left on each rank. Correct write-back gives
      ``a + b``; a missing divide double-counts to ``2(a + b)``.

Usage: moe_worker.py --ep {0,1} --out RESULT.json
"""

import argparse
import json

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--ep", type=int, required=True)
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()

from nnsight.modeling.vllm import VLLM

MODEL = "Qwen/Qwen1.5-MoE-A2.7B"
PROMPT = "The Eiffel Tower is located in the city of"


def main():
    model = VLLM(
        MODEL,
        tensor_parallel_size=2,
        enable_expert_parallel=bool(args.ep),
        gpu_memory_utilization=0.40,
        dispatch=True,
    )

    result = {"ep": args.ep}

    # --- read: gathered experts.output must equal the block's own output ---
    # The block output is cloned at access: vLLM's next layer mutates the
    # returned hidden-states tensor in place (fused add_rms_norm), so a raw
    # save reads back corrupted values. That save-time hazard is a separate,
    # independently regression-tested bug (clone-on-save, #661); cloning here
    # keeps this oracle measuring the batcher only. The experts tuple needs no
    # clone: the gather's all-reduce already allocates fresh tensors.
    with model.trace(PROMPT, temperature=0.0, top_p=1, max_tokens=1):
        experts_out = model.model.layers[11].mlp.experts.output.save()
        block_out = model.model.layers[11].mlp.output.clone().save()

    ex_sum = (experts_out[0] + experts_out[1]).float().cpu()
    block = block_out.float().cpu()
    result["read_cos"] = torch.nn.functional.cosine_similarity(
        ex_sum.flatten(), block.flatten(), dim=0
    ).item()
    result["read_max_delta"] = (ex_sum - block).abs().max().item()

    # --- write: a swapped-in value must reach the block exactly once ---
    n_tokens = block.shape[0]
    hidden = block.shape[-1]
    with model.trace(PROMPT, temperature=0.0, top_p=1, max_tokens=1):
        mlp = model.model.layers[11].mlp
        shape = mlp.experts.output[0].shape
        dev = mlp.experts.output[0].device
        dtype = mlp.experts.output[0].dtype
        a = torch.full(tuple(shape), 0.01, device=dev, dtype=dtype)
        b = torch.full(tuple(shape), 0.02, device=dev, dtype=dtype)
        mlp.experts.output = (a, b)
        swapped_block = mlp.output.clone().save()

    sb = swapped_block.float().cpu()
    expected = torch.full((n_tokens, hidden), 0.03)
    result["write_max_delta"] = (sb - expected).abs().max().item()

    with open(args.out, "w") as f:
        json.dump(result, f)
    print("RESULT", json.dumps(result))


if __name__ == "__main__":
    main()
