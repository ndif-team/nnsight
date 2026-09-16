"""Topology coverage: three pipeline stages, and tensor parallelism under PP.

Same structure as the parity suite: each engine runs in its own subprocess
via _parity_worker.py and the parent compares JSON. Two topologies:

* PP=3 (3 GPUs): reads land on all three stages, and the cross-stage write
  takes a value from a non-adjacent stage (rank 2 reads rank 0's layer, past
  rank 1).
* TP=2 x PP=2 (4 GPUs): the reference is TP=2 with a single pipeline stage,
  so the comparison isolates what PP adds under sharding; values cross stages
  between column peers.
"""

import json
import os
import sys

import pytest
import torch

pytest.importorskip("vllm")

from _support import PROMPT, cosine, free_gpus, run_worker

pytestmark = pytest.mark.gpu


FREE_GPUS = free_gpus()

if len(FREE_GPUS) < 3:
    pytest.skip(
        f"topology tests need at least 3 free GPUs, found {len(FREE_GPUS)}",
        allow_module_level=True,
    )

needs_four_gpus = pytest.mark.skipif(
    len(FREE_GPUS) < 4,
    reason=f"TP=2 x PP=2 needs 4 free GPUs, found {len(FREE_GPUS)}",
)



def run(tp, pp, scenario, *extra):
    """One scenario on a tp x pp engine, in its own subprocess."""
    return run_worker(scenario, pp=pp, tp=tp, gpus=",".join(FREE_GPUS[: tp * pp]), extra=tuple(extra))


def test_three_stage_reads_match_reference():
    reference = run(1, 1, "hidden_three_stages")
    pipelined = run(1, 3, "hidden_three_stages")
    assert reference["argmax"] == pipelined["argmax"]
    for site in ("early", "middle", "late"):
        assert reference[f"{site}_shape"] == pipelined[f"{site}_shape"]
        similarity = cosine(reference[site], pipelined[site])
        assert similarity > 0.99, f"{site}-layer cosine {similarity:.6f}"


def test_three_stage_nonadjacent_write_matches_reference():
    reference = run(1, 1, "write_cross")
    pipelined = run(1, 3, "write_cross")
    similarity = cosine(reference["logits"], pipelined["logits"])
    assert similarity > 0.99, f"grafted-run logits cosine {similarity:.6f}"
    assert reference["argmax"] == pipelined["argmax"]
    clean = run(1, 1, "logits")
    moved = cosine(reference["logits"], clean["logits"])
    assert moved < 0.999, f"graft had no effect (cosine to clean {moved:.6f})"


@needs_four_gpus
def test_tp_pp_reads_match_tp_reference():
    reference = run(2, 1, "hidden")
    sharded_pipelined = run(2, 2, "hidden")
    assert reference["argmax"] == sharded_pipelined["argmax"]
    for site in ("early", "late"):
        assert reference[f"{site}_shape"] == sharded_pipelined[f"{site}_shape"]
        similarity = cosine(reference[site], sharded_pipelined[site])
        assert similarity > 0.99, f"{site}-layer cosine {similarity:.6f}"


@needs_four_gpus
def test_tp_pp_write_matches_tp_reference():
    reference = run(2, 1, "write_cross")
    sharded_pipelined = run(2, 2, "write_cross")
    similarity = cosine(reference["logits"], sharded_pipelined["logits"])
    assert similarity > 0.99, f"grafted-run logits cosine {similarity:.6f}"
    assert reference["argmax"] == sharded_pipelined["argmax"]
    clean = run(2, 1, "logits")
    moved = cosine(reference["logits"], clean["logits"])
    assert moved < 0.999, f"graft had no effect (cosine to clean {moved:.6f})"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-x", "-p", "no:cacheprovider"]))


@pytest.mark.skipif(len(FREE_GPUS) < 4, reason=f"TP=2 x PP=2 needs 4 free GPUs, found {len(FREE_GPUS)}")
def test_tp_pp_every_rank_releases_workers():
    counts = run(2, 2, "release")["counts"]
    assert len(counts) == 4 and all(count == 0 for count in counts), counts


@needs_four_gpus
def test_tp_pp_param_reads_match_tp_reference():
    """A parameter fetched from the other stage and gathered across this
    stage's TP ranks equals the TP=2 single-stage read of the same
    parameter, on both sharding axes."""
    reference = run(2, 1, "param_reads")
    sharded_pipelined = run(2, 2, "param_reads")
    for name in ("norm", "down", "qkv"):
        assert reference[f"{name}_shape"] == sharded_pipelined[f"{name}_shape"], name
        assert reference[f"{name}_sum"] == pytest.approx(sharded_pipelined[f"{name}_sum"], rel=1e-4), name
    for name in ("down_row", "qkv_row"):
        assert reference[name] == pytest.approx(sharded_pipelined[name], rel=1e-4), name


@needs_four_gpus
def test_tp_pp_remote_calls_match_tp_reference():
    """A module called from the stage that does not hold it runs on the
    owner's shards with this stage's collectives and gives the TP=2
    single-stage value."""
    reference = run(2, 1, "remote_calls")
    sharded_pipelined = run(2, 2, "remote_calls")
    for name in ("normed", "mlp"):
        assert reference[f"{name}_shape"] == sharded_pipelined[f"{name}_shape"], name
        similarity = cosine(reference[name], sharded_pipelined[name])
        assert similarity > 0.999, f"{name} cosine {similarity:.6f}"
