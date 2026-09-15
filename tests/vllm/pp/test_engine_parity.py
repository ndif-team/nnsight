"""PP=2 vs PP=1 parity on a real engine.

Every scenario runs identical intervention code at PP=1 (the reference) and
PP=2, each in its own subprocess (one engine per process; separate processes
avoid distributed-env and GPU-memory conflicts). The parent compares the JSON.

Bitwise equality is not expected: the PP boundary transfer re-orders bf16
reductions. The bar is identical argmax at every compared site and cosine
similarity above 0.99 on hidden states and logits.

The concurrent scenario is the one place multiple mediators live in the same
engine step under PP: two invokes with different-length prompts, each doing
cross-stage reads. It pins per-request publish narrowing, request-id-keyed
pulls, and the save merge keeping both invokes' values distinct.
"""

import os
import sys

import pytest
import torch

pytest.importorskip("vllm")

import _support
from _support import PROMPT, cosine, free_gpus, run_worker

pytestmark = pytest.mark.gpu


FREE_GPUS = free_gpus()

if len(FREE_GPUS) < 2:
    pytest.skip(
        f"parity tests need 2 free GPUs, found {len(FREE_GPUS)}: {FREE_GPUS}",
        allow_module_level=True,
    )

PROMPT_B = _support.PROMPT_B
GPUS_PP1 = FREE_GPUS[0]
GPUS_PP2 = f"{FREE_GPUS[0]},{FREE_GPUS[1]}"


def run(pp, scenario, *extra):
    """One scenario on a PP=1 reference or PP=2 engine, in its own subprocess."""
    return run_worker(scenario, pp=pp, gpus=GPUS_PP1 if pp == 1 else GPUS_PP2, extra=tuple(extra))


def test_reference_predicts_paris():
    reference = run(1, "logits")
    assert reference["top_token"].strip() == "Paris", reference["top_token"]


def test_logits_argmax_and_cosine_match():
    reference = run(1, "logits")
    pipelined = run(2, "logits")
    assert reference["argmax"] == pipelined["argmax"], (
        f"PP=1 {reference['top_token']!r} vs PP=2 {pipelined['top_token']!r}"
    )
    similarity = cosine(reference["logits"], pipelined["logits"])
    assert similarity > 0.99, f"logits cosine {similarity:.6f}"


def test_early_layer_hidden_matches():
    reference = run(1, "hidden")
    pipelined = run(2, "hidden")
    assert reference["early_shape"] == pipelined["early_shape"]
    similarity = cosine(reference["early"], pipelined["early"])
    assert similarity > 0.99, f"early-layer cosine {similarity:.6f}"


def test_late_layer_hidden_matches():
    reference = run(1, "hidden")
    pipelined = run(2, "hidden")
    assert reference["late_shape"] == pipelined["late_shape"]
    similarity = cosine(reference["late"], pipelined["late"])
    assert similarity > 0.99, f"late-layer cosine {similarity:.6f}"
    assert reference["argmax"] == pipelined["argmax"]


def test_stage_local_write_parity():
    reference = run(1, "write_local")
    pipelined = run(2, "write_local")
    similarity = cosine(reference["logits"], pipelined["logits"])
    assert similarity > 0.99, f"written-run logits cosine {similarity:.6f}"
    assert reference["argmax"] == pipelined["argmax"]
    # A silently-dropped write leaves both runs at the unperturbed logits,
    # making the parity above vacuous; zeroing a layer must move them.
    clean = run(1, "logits")
    moved = cosine(reference["logits"], clean["logits"])
    assert moved < 0.999, f"write had no effect (cosine to clean {moved:.6f})"


def test_cross_stage_read_modify_write():
    # The graft forces an upstream value while parked inside the late layer's
    # hook on the owning rank; the intercept serves upstream pulls in place,
    # so the swap that follows still lands before the forward moves on.
    reference = run(1, "write_cross")
    pipelined = run(2, "write_cross")
    similarity = cosine(reference["logits"], pipelined["logits"])
    assert similarity > 0.99, f"grafted-run logits cosine {similarity:.6f}"
    assert reference["argmax"] == pipelined["argmax"]
    clean = run(1, "logits")
    moved = cosine(reference["logits"], clean["logits"])
    assert moved < 0.999, f"graft had no effect (cosine to clean {moved:.6f})"


def test_multi_token_ids_and_late_hidden_match():
    reference = run(1, "multigen", "--max-tokens", "3")
    pipelined = run(2, "multigen", "--max-tokens", "3")
    assert reference["ids"] == pipelined["ids"], (
        f"sampled ids diverge: PP=1 {reference['ids']} vs PP=2 {pipelined['ids']}"
    )
    assert reference["late_shapes"] == pipelined["late_shapes"]
    for step, (ref_step, pp_step) in enumerate(
        zip(reference["late"], pipelined["late"])
    ):
        similarity = cosine(ref_step, pp_step)
        assert similarity > 0.99, f"step {step} late-layer cosine {similarity:.6f}"


def test_multi_token_forced_cross_stage_reads_match():
    # Forces the late layer's value on every generation step, so the
    # non-owning rank issues one pinned pull per round and re-parks on the
    # next round's pull from inside the step-start serve. Deadlocked until
    # the serve completed only already-produced rounds there.
    reference = run(1, "multigen_forced", "--max-tokens", "3")
    pipelined = run(2, "multigen_forced", "--max-tokens", "3")
    assert reference["ids"] == pipelined["ids"], (
        f"sampled ids diverge: {reference['ids']} vs {pipelined['ids']}"
    )
    for step, (ref, pp) in enumerate(zip(reference["norms"], pipelined["norms"])):
        assert abs(ref - pp) / max(abs(ref), 1e-6) < 0.01, (step, ref, pp)


def test_concurrent_requests_match_reference_per_invoke():
    reference = run(1, "concurrent", "--prompt-b", PROMPT_B)
    pipelined = run(2, "concurrent", "--prompt-b", PROMPT_B)
    for invoke in ("first", "second"):
        ref, pp = reference[invoke], pipelined[invoke]
        # Each invoke's saves must be its own rows of the shared batch: the
        # prompt's token count, not the whole slab, and not the peer's rows.
        assert pp["early_shape"][0] == pp["prompt_tokens"], (invoke, pp["early_shape"])
        assert ref["early_shape"] == pp["early_shape"]
        assert ref["late_shape"] == pp["late_shape"]
        assert ref["argmax"] == pp["argmax"], invoke
        for site in ("early", "late"):
            similarity = cosine(ref[site], pp[site])
            assert similarity > 0.99, (
                f"{invoke} invoke {site}-layer cosine {similarity:.6f}"
            )


def test_concurrent_requests_see_different_activations():
    # Guard against cross-request clobber at PP=2: different prompts must
    # produce different values, and both invokes' saves must have come home
    # (a dropped second-invoke save was a real bug in this branch's history).
    pipelined = run(2, "concurrent", "--prompt-b", PROMPT_B)
    first, second = pipelined["first"], pipelined["second"]
    assert first["early"] and second["early"]
    assert first["argmax"] != second["argmax"] or first["early"] != second["early"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-x", "-p", "no:cacheprovider"]))


def test_every_rank_releases_workers_after_a_trace():
    counts = run(2, "release")["counts"]
    assert len(counts) == 2 and all(count == 0 for count in counts), counts
