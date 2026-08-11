"""Tensor parallelism against a real checkpoint, not a random tiny one.

`test_sharded_tracing.py` proves the mechanics on
``tiny-random-LlamaForCausalLM`` — 16 hidden units, untied embeddings, float32.
That is enough for the row math and fast enough to run often, and it is *not*
enough to trust a deployment, because the things production models do differently
are exactly the things that have broken:

* **Tied embeddings.** A checkpoint with ``tie_word_embeddings=True`` shares the
  LM head's weight with the embedding. On a transformers whose plan shards the
  head but not the embedding it is tied to, the head keeps its full weight while
  its ``colwise_gather_output`` hook still fires, and logits come back
  ``tp_size`` times too wide — inside transformers, before nnsight sees anything.
  The tiny model has this flag off, so only a real one catches it. Asserting the
  logits width here is the regression test for that.
* **Real head counts.** ``num_key_value_heads`` is what actually limits the
  degree a model shards into (Llama-3.2-3B has 8, so it divides by 4; a model
  with 2 would not). The tiny model's heads all equal each other.
* **bfloat16.** What deployments serve, and where the drift between a one-rank
  and an N-rank run shows up.

Gated on 4 GPUs and skipped by CI along with the rest of this directory. Reaching
for a ~6.4GB checkpoint means this is a deliberate, occasional run:

    python -m pytest tests/tp/test_real_model.py
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 4,
    reason="a real tensor-parallel model needs >=4 GPUs",
)

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")

# Divides cleanly by 4 on every axis that matters: 24 attention heads, 8
# key/value heads, 8192 intermediate. And it ties its embeddings.
REPO = "meta-llama/Llama-3.2-3B"
TP_SIZE = 4
VOCAB = 128256
HIDDEN = 3072
INTERMEDIATE = 8192

# bfloat16 across a 28-layer model drifts further than the tiny float32 one, and
# an all-reduce sums in a different order than a single matmul. Order 1 would
# mean a layout error; this is arithmetic.
DRIFT = 5e-2


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
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [SRC, os.environ.get("PYTHONPATH")])),
        "CUDA_VISIBLE_DEVICES": _visible_devices(tp),
    }
    command = [sys.executable]
    if tp > 1:
        command += [
            "-m", "torch.distributed.run",
            f"--nproc_per_node={tp}",
            f"--master_port={_free_port()}",
        ]
    command += [
        WORKER, "--tp", str(tp), "--repo", REPO, "--out", out, "--dtype", "bfloat16",
    ]

    completed = subprocess.run(command, env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        pytest.fail(
            f"tp={tp} worker failed ({completed.returncode})\n"
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
    reference_dir = tmp_path_factory.mktemp("real_tp1")
    sharded_dir = tmp_path_factory.mktemp(f"real_tp{TP_SIZE}")

    _run(1, str(reference_dir))
    _run(TP_SIZE, str(sharded_dir))

    reference = torch.load(reference_dir / "rank0.pt", weights_only=False)
    sharded = [
        torch.load(sharded_dir / f"rank{rank}.pt", weights_only=False)
        for rank in range(TP_SIZE)
    ]
    return reference, sharded


class TestWidths:
    """Every value arrives at the width the architecture says, not a fraction of
    it — and not a multiple, which is the tied-embedding failure."""

    def test_logits_are_not_a_multiple_of_the_vocabulary(self, runs):
        # The regression test for the tied-embedding gather. A head whose weight
        # was never sharded yields tp_size * VOCAB, and nothing downstream looks
        # wrong: the argmax still lands inside the first copy.
        _, sharded = runs
        assert sharded[0]["baseline_logits"].shape[-1] == VOCAB

    def test_a_column_parallel_output_is_whole(self, runs):
        _, sharded = runs
        assert sharded[0]["gate_proj_out"].shape[-1] == INTERMEDIATE

    def test_a_row_parallel_input_is_whole(self, runs):
        _, sharded = runs
        assert sharded[0]["down_proj_in"].shape[-1] == INTERMEDIATE

    def test_a_layer_output_is_whole(self, runs):
        _, sharded = runs
        assert sharded[0]["layer_out"].shape[-1] == HIDDEN


VALUES = [
    "gate_proj_out",
    "down_proj_in",
    "layer_out",
    "baseline_logits",
    "partial_edit_logits",
    "cached_gate_out",
    "generated",
    "batched_first",
    "batched_edited_logits",
    "generated_steps",
]


@pytest.mark.parametrize("name", VALUES)
def test_ranks_agree(runs, name):
    _, sharded = runs
    first = sharded[0][name]
    for rank, results in enumerate(sharded[1:], start=1):
        assert results[name].shape == first.shape, f"rank {rank} shape differs"
        assert torch.equal(results[name], first), f"rank {rank} diverged on {name}"


@pytest.mark.parametrize("name", VALUES)
def test_matches_single_gpu(runs, name):
    reference, sharded = runs
    expected, actual = reference[name], sharded[0][name]
    assert actual.shape == expected.shape
    assert _rel(actual, expected) < DRIFT


def test_generation_is_identical(runs):
    """Exact token ids: bf16 drift never changed a choice."""
    reference, sharded = runs
    assert torch.equal(sharded[0]["generated"], reference["generated"])
