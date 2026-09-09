"""Expert parallelism: whole tensors from a model whose *experts* are split.

`test_cpu_gloo.py` covers tensor parallelism, where a module's activation is one
rank's slice along a tensor axis. Expert parallelism splits a different thing —
whole experts across ranks — and transformers expresses it with a separate plan
(``base_model_ep_plan``) applied when ``enable_expert_parallel=True``. The styles
in it are not variations on colwise/rowwise:

* ``ep_router`` leaves the router replicated and masks non-local experts in its
  own post-transform, so at the handoff its value is already whole.
* ``grouped_gemm`` shards expert *parameters* and installs an identity wrapper,
  so it too has nothing to gather.
* ``moe_tp_experts`` produces this rank's term of a sum, like a row-parallel
  output.

All three used to be refused outright. Two of them turn out to need no gather at
all, and the third was already described — but nothing had ever run the path, so
"refused" and "correct" were indistinguishable. This is what tells them apart.

Runs on CPU over gloo, like its tensor-parallel sibling, so CI covers it.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys

import pytest
import torch

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ep_worker.py")
EP_SIZE = 2

# float32 on two CPU ranks; the only arithmetic difference from the reference is
# the order an all-reduce sums in.
DRIFT = 1e-4


def _nnsight_path() -> str:
    """The directory that makes the worker import the nnsight this session did."""
    import nnsight

    return os.path.dirname(os.path.dirname(os.path.abspath(nnsight.__file__)))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _run(ep: int, out: str) -> None:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(None, [_nnsight_path(), os.environ.get("PYTHONPATH")])
        ),
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
    }
    command = [sys.executable]
    if ep > 1:
        command += [
            "-m", "torch.distributed.run",
            f"--nproc_per_node={ep}",
            f"--master_port={_free_port()}",
        ]
    command += [WORKER, "--ep", str(ep), "--out", out]

    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        pytest.fail(
            f"ep={ep} worker failed ({completed.returncode})\n"
            f"--- stdout ---\n{completed.stdout[-4000:]}\n"
            f"--- stderr ---\n{completed.stderr[-4000:]}"
        )


def _rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if actual.shape != expected.shape:
        return float("inf")
    scale = expected.abs().max().item()
    return (actual - expected).abs().max().item() / (scale if scale else 1.0)


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> tuple[dict, list[dict]]:
    reference_dir = tmp_path_factory.mktemp("ep1")
    sharded_dir = tmp_path_factory.mktemp(f"ep{EP_SIZE}")

    _run(1, str(reference_dir))
    _run(EP_SIZE, str(sharded_dir))

    reference = torch.load(os.path.join(reference_dir, "rank0.pt"), weights_only=False)
    sharded = [
        torch.load(os.path.join(sharded_dir, f"rank{rank}.pt"), weights_only=False)
        for rank in range(EP_SIZE)
    ]
    return reference, sharded


def test_the_ranks_agree(runs) -> None:
    """Every rank saw the same values — the collectives lined up."""
    _, sharded = runs
    first, *rest = sharded
    for rank, result in enumerate(rest, start=1):
        for name, value in first.items():
            assert torch.equal(value, result[name]), f"rank {rank} disagrees on {name}"


@pytest.mark.parametrize(
    "name",
    [
        "router_logits",   # replicated: masked only after the handoff
        "experts_out",     # this rank's term of the expert sum
        "mlp_out",
        "logits",
        "edited_logits",   # an edit on the summed expert output, carried back
    ],
)
def test_rank0_matches_the_single_process_run(runs, name) -> None:
    reference, sharded = runs
    drift = _rel(sharded[0][name], reference[name])
    assert drift < DRIFT, (
        f"{name}: relative error {drift:.2e} against the 1-process run "
        f"(shapes {tuple(sharded[0][name].shape)} vs {tuple(reference[name].shape)}). "
        "Order 1 means the value was not made whole; this is not drift."
    )
