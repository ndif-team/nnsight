"""Tensor parallelism: sharded activations read and edit as whole tensors.

A model loaded with ``distributed_config=DistributedConfig(tp_size=N)`` has its
linears split across ranks, so the value at a column-parallel *output* or a
row-parallel *input* is only that rank's slice.
[`TPFragments`][nnsight.modeling.tp.fragments.TPFragments] gathers those
before a worker sees them and re-splits whatever the worker leaves, so a trace is
written exactly as it would be against one GPU.

These tests pin that down by running the identical block twice — once on a single
GPU for the reference, once across N ranks — and comparing. Each run is a
subprocess, because transformers tensor parallelism requires the *calling* process
to be a rank; the sibling `tests/vllm/test_tensor_parallel.py` can stay in-process
only because vLLM spawns its own workers.

This whole directory needs hardware, so CI ignores it (`--ignore=tests/tp`,
alongside `tests/vllm`). Run it by hand on a multi-GPU box:

    python -m pytest tests/tp

Two independent things are checked, because they fail differently:

* **the ranks agree** — bit-for-bit. A failure here means the ranks diverged,
  which is the deadlock/corruption class of bug.
* **rank 0 matches the single-GPU run** — within a relative tolerance, since an
  all-reduce sums in a different order than one big matmul and float arithmetic
  does not associate. A mismatch of order 1 means the gather or the re-split is
  wrong; 1e-3 is drift.

Skipped unless the machine has >=2 GPUs.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="transformers tensor parallelism needs >=2 GPUs",
)

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")

# Tiny, and shards cleanly: its head counts and intermediate size all divide by 2.
REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TP_SIZE = 2

# Relative error below this is float non-associativity between a 1-rank and an
# N-rank run, not a layout error (which shows up at order 1).
DRIFT = 1e-3


def _visible_devices(tp: int) -> str:
    """Which cards the ranks use: whatever was set, else the first ``tp``."""
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited:
        return ",".join(inherited.split(",")[:tp])
    return ",".join(str(index) for index in range(tp))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _run(tp: int, out: str) -> None:
    """Run the worker at ``tp`` ranks, writing one file per rank into ``out``."""
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [SRC, os.environ.get("PYTHONPATH")])),
        # Rank i must land on the i-th visible device; see
        # transformers.integrations.tensor_parallel.initialize_tensor_parallelism,
        # which uses LOCAL_RANK directly as a CUDA index. An inherited setting
        # wins, so a shared machine can be pointed at the cards that are free.
        "CUDA_VISIBLE_DEVICES": _visible_devices(tp),
    }
    command = [sys.executable]
    if tp > 1:
        command += [
            "-m", "torch.distributed.run",
            f"--nproc_per_node={tp}",
            f"--master_port={_free_port()}",
        ]
    command += [WORKER, "--tp", str(tp), "--repo", REPO, "--out", out]

    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        pytest.fail(
            f"tp={tp} worker failed ({completed.returncode})\n"
            f"--- stdout ---\n{completed.stdout[-4000:]}\n"
            f"--- stderr ---\n{completed.stderr[-4000:]}"
        )


def _rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """max|a-b| / max|b| — scale-free, so it reads the same at any layer."""
    if actual.shape != expected.shape:
        return float("inf")
    scale = expected.abs().max().item()
    return (actual - expected).abs().max().item() / (scale if scale else 1.0)


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> tuple[dict, list[dict]]:
    """``(reference, [per-rank sharded results])`` from two subprocess runs."""
    reference_dir = tmp_path_factory.mktemp("tp1")
    sharded_dir = tmp_path_factory.mktemp(f"tp{TP_SIZE}")

    _run(1, str(reference_dir))
    _run(TP_SIZE, str(sharded_dir))

    reference = torch.load(reference_dir / "rank0.pt", weights_only=False)
    sharded = [
        torch.load(sharded_dir / f"rank{rank}.pt", weights_only=False)
        for rank in range(TP_SIZE)
    ]
    return reference, sharded


# Every value the worker records: a sharded read, a boundary-straddling edit, a
# cached read, and a generation.
VALUES = [
    "gate_proj_out",
    "down_proj_in",
    "layer_out",
    "baseline_logits",
    "partial_edit_logits",
    "cached_gate_out",
    "partial_backward_grad",   # backward through a row-parallel output
    "source_colwise_gathered",   # tp.gather on an op inside a sharded module
    "manual_gathered_heads",   # tp.gather on a value between two shards
    "manual_ablated_logits",   # ... edited and tp.shard-ed back
    "generated",
    "batched_first",
    "batched_edited_logits",
    "generated_steps",
    "adhoc_colwise",
    "adhoc_rowwise",
    "adhoc_hooked",   # hook=True must not skip the bracket
    "adhoc_nested",   # an ad-hoc call inside its own open handoff
    "skip_read_back",   # a .skip replacement is not gathered
    "skip_logits",
    "adhoc_lens",
]


@pytest.mark.parametrize("name", VALUES)
def test_ranks_agree(runs, name):
    """Every rank produced the same value, bit for bit."""
    _, sharded = runs
    first = sharded[0][name]
    for rank, results in enumerate(sharded[1:], start=1):
        assert results[name].shape == first.shape, f"rank {rank} shape differs"
        assert torch.equal(results[name], first), f"rank {rank} diverged on {name}"


@pytest.mark.parametrize("name", VALUES)
def test_matches_single_gpu(runs, name):
    """The sharded run reproduces the single-GPU run."""
    reference, sharded = runs
    expected, actual = reference[name], sharded[0][name]

    # Full width, not this rank's slice — the gather's whole point.
    assert actual.shape == expected.shape
    assert _rel(actual, expected) < DRIFT


def test_generation_is_identical(runs):
    """Token ids are exact, so any drift in the logits never changed a choice."""
    reference, sharded = runs
    assert torch.equal(sharded[0]["generated"], reference["generated"])


def test_edit_actually_changed_the_output(runs):
    """The edit case isn't passing vacuously by leaving the model untouched.

    Without this, an intervention that silently did nothing would still agree
    across ranks and still match the reference — every other test would pass.
    """
    reference, sharded = runs
    for results, label in ((reference, "tp=1"), (sharded[0], "sharded")):
        moved = _rel(results["partial_edit_logits"], results["baseline_logits"])
        assert moved > DRIFT, f"{label}: zeroing half of gate_proj changed nothing"
