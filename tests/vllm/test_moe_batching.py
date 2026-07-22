"""MoE (FusedMoE) deferred-reduce values must be gathered on access and
un-scaled on write-back, in both parallel layouts of the same ranks:
tensor-sliced experts (EP off) and whole-expert placement (EP on).

Qwen-MoE-family models build their fused-experts module with
``reduce_results=False``: it returns per-rank partial sums the outer block
all-reduces afterwards. Without batcher handling, a read of
``mlp.experts.output`` ships one rank's partial (silently wrong values) and
a swap gets double-counted by the downstream all-reduce. This is the same
exposure ``RowParallelLinear`` has always had handling for; these tests pin
the FusedMoE case.

Runs a subprocess per engine (see moe_worker.py for the checks and oracle).
Needs 2 GPUs with ~33 GiB free each and the Qwen1.5-MoE-A2.7B weights.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest


def _find_free_gpus(min_free_mib):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        free = []
        for line in result.stdout.strip().split("\n"):
            idx, free_mib = line.split(",")
            if int(free_mib.strip()) >= min_free_mib:
                free.append(int(idx.strip()))
        return free
    except Exception:
        return []


# vLLM claims gpu_memory_utilization=0.40 of an 80 GiB card (~33 GiB).
FREE_GPUS = _find_free_gpus(min_free_mib=36000)

if len(FREE_GPUS) < 2:
    pytest.skip(
        f"MoE batching tests need 2 free GPUs, found {len(FREE_GPUS)}: {FREE_GPUS}",
        allow_module_level=True,
    )

GPUS = f"{FREE_GPUS[0]},{FREE_GPUS[1]}"
WORKER = os.path.join(os.path.dirname(__file__), "moe_worker.py")
TIMEOUT_S = 600

# Kernel-noise scale for a bf16 activation compared against its all-reduce.
READ_MAX_DELTA = 0.05
READ_MIN_COS = 0.9999
# The write check compares two constants (0.01 + 0.02); only bf16 rounding
# and the all-reduce contribute.
WRITE_MAX_DELTA = 0.005


def _run_worker(ep: int):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS)
    proc = subprocess.run(
        [sys.executable, WORKER, "--ep", str(ep), "--out", out_path],
        capture_output=True, text=True, timeout=TIMEOUT_S, env=env,
    )
    assert proc.returncode == 0, (
        f"worker failed (ep={ep}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    )
    with open(out_path) as f:
        return json.load(f)


@pytest.mark.parametrize("ep", [0, 1], ids=["tensor_sliced_experts", "whole_expert_placement"])
def test_experts_output_read_is_full_value_and_write_reaches_block_once(ep):
    result = _run_worker(ep)
    assert result["read_cos"] >= READ_MIN_COS, (
        f"experts.output read is a per-rank partial, not the full value: "
        f"cos={result['read_cos']:.4f} vs its own block output"
    )
    assert result["read_max_delta"] <= READ_MAX_DELTA, (
        f"experts.output read diverges from its own block output: "
        f"max|delta|={result['read_max_delta']:.4f}"
    )
    assert result["write_max_delta"] <= WRITE_MAX_DELTA, (
        f"swapped experts.output does not reach the block exactly once "
        f"(missing write-back scaling double-counts it): "
        f"max|delta|={result['write_max_delta']:.4f}"
    )
