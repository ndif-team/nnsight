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

The suite also covers the other way to shard a MoE model — plain tensor
parallelism, no EP — because the two configurations resolve the *same* experts
module to different styles, and getting that resolution wrong is invisible to
both the dense TP suite and the EP run (see `MOE_TP_REPO` below).

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


def _run(world: int, out: str, mode: str = "ep", repo: str | None = None) -> None:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(None, [_nnsight_path(), os.environ.get("PYTHONPATH")])
        ),
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
    }
    command = [sys.executable]
    if world > 1:
        command += [
            "-m", "torch.distributed.run",
            f"--nproc_per_node={world}",
            f"--master_port={_free_port()}",
        ]
    command += [WORKER, "--ep", str(world), "--out", out, "--mode", mode]
    if repo is not None:
        command += ["--repo", repo]

    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        pytest.fail(
            f"{mode}={world} worker failed ({completed.returncode})\n"
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


# A MoE model sharded with plain *tensor* parallelism — no expert parallelism.
# This is its own case because the raw plans collide: Qwen3-MoE names
# ``layers.*.mlp.experts`` as ``moe_tp_experts`` in ``tp_plan`` and as
# ``ep_dispatch_experts`` in ``ep_plan``, and transformers applies the EP entry
# only when ``ep_size > 1``. nnsight has to resolve styles the way they were
# *applied*: merging ``ep_plan`` in unconditionally made this run resolve the
# experts to a style transformers never installed, strip the real wrapper, and
# die on ``aten._grouped_mm got mixed torch.Tensor and DTensor``.
#
# The model is generated here rather than pulled from the hub:
# ``hf-internal-testing/tiny-random-Qwen3MoeForCausalLM`` predates 5.19's
# ``mlp.router`` -> ``mlp.gate`` rename, so its router is *randomly
# re-initialized on every load* and no two processes compute the same thing —
# plain transformers already drifts 7e-2 from itself on it. A checkpoint saved
# by the running transformers has no such skew, and needs no network.


@pytest.fixture(scope="module")
def moe_tp_repo(tmp_path_factory) -> str:
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        moe_intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, num_experts=4,
        num_experts_per_tok=2, decoder_sparse_step=1, mlp_only_layers=[],
    )
    path = tmp_path_factory.mktemp("qwen3moe_tiny")
    Qwen3MoeForCausalLM(config).float().save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def tp_runs(tmp_path_factory, moe_tp_repo) -> tuple[dict, list[dict]]:
    reference_dir = tmp_path_factory.mktemp("moe_tp1")
    sharded_dir = tmp_path_factory.mktemp(f"moe_tp{EP_SIZE}")

    _run(1, str(reference_dir), mode="tp", repo=moe_tp_repo)
    _run(EP_SIZE, str(sharded_dir), mode="tp", repo=moe_tp_repo)

    reference = torch.load(os.path.join(reference_dir, "rank0.pt"), weights_only=False)
    sharded = [
        torch.load(os.path.join(sharded_dir, f"rank{rank}.pt"), weights_only=False)
        for rank in range(EP_SIZE)
    ]
    return reference, sharded


def test_tp_only_moe_ranks_agree(tp_runs) -> None:
    _, sharded = tp_runs
    first, *rest = sharded
    for rank, result in enumerate(rest, start=1):
        for name, value in first.items():
            assert torch.equal(value, result[name]), f"rank {rank} disagrees on {name}"


@pytest.mark.parametrize("name", ["experts_out", "mlp_out", "logits", "edited_logits"])
def test_tp_only_moe_matches_the_single_process_run(tp_runs, name) -> None:
    reference, sharded = tp_runs
    drift = _rel(sharded[0][name], reference[name])
    assert drift < DRIFT, (
        f"{name}: relative error {drift:.2e} against the 1-process run "
        f"(shapes {tuple(sharded[0][name].shape)} vs {tuple(reference[name].shape)}). "
        "Order 1 means the value was not made whole; this is not drift."
    )
