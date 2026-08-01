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

import functools
import json
import os
import subprocess
import sys
import tempfile

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("vllm")


def _free_gpus(min_free_mib=12000):
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            index, free = line.split(",")
            if int(free.strip()) >= min_free_mib:
                gpus.append(index.strip())
        return gpus
    except Exception:
        return []


FREE_GPUS = _free_gpus()

if len(FREE_GPUS) < 2:
    pytest.skip(
        f"parity tests need 2 free GPUs, found {len(FREE_GPUS)}: {FREE_GPUS}",
        allow_module_level=True,
    )

GPUS_PP1 = FREE_GPUS[0]
GPUS_PP2 = f"{FREE_GPUS[0]},{FREE_GPUS[1]}"

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_parity_worker.py")
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

PROMPT = "The Eiffel Tower is located in the city of"
PROMPT_B = "Madison Square Garden is located in the city of"


@functools.lru_cache(maxsize=None)
def run(pp, scenario, *extra):
    """Run one scenario in a subprocess; cached so references are booted once."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_path = f.name
    log_path = output_path + ".log"
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = GPUS_PP1 if pp == 1 else GPUS_PP2
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env["PYTHONPATH"] = os.path.join(REPO_ROOT, "src")
        cmd = [
            sys.executable,
            WORKER,
            scenario,
            "--pp",
            str(pp),
            "--prompt",
            PROMPT,
            "--output",
            output_path,
            *extra,
        ]
        # Engine logs go to a file, not pipes: a pipe would make this call
        # wait for EOF, which a leaked engine subprocess could hold open long
        # after the worker itself exited.
        with open(log_path, "w") as log:
            result = subprocess.run(
                cmd, stdout=log, stderr=log, timeout=600, env=env, cwd=REPO_ROOT
            )
        try:
            with open(output_path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = None
        if data is None or data.get("status") != "ok":
            with open(log_path) as log:
                tail = log.read()[-4000:]
            detail = (
                f"{data.get('error')}\n{data.get('traceback')}"
                if data
                else f"no output written\nWORKER LOG TAIL:\n{tail}"
            )
            raise RuntimeError(
                f"parity worker failed (scenario={scenario}, pp={pp}, "
                f"rc={result.returncode}):\n{detail}"
            )
        return data
    finally:
        for path in (output_path, log_path):
            try:
                os.unlink(path)
            except OSError:
                pass


def cosine(a, b):
    a = torch.tensor(a, dtype=torch.float32).flatten()
    b = torch.tensor(b, dtype=torch.float32).flatten()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "a write that follows a forced cross-stage read loses its swap window "
        "on the owning rank: the worker parks on the pull inside the target "
        "layer's hook, pulls are only served at serve points outside the "
        "forward, and by then the forward has passed the write site — the swap "
        "raises OutOfOrderError. The 0.7 thread-based PP supported this "
        "pattern (the hook blocked until the worker's pull completed)."
    ),
)
def test_cross_stage_read_modify_write():
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
