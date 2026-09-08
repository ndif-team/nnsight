"""Topology coverage: three pipeline stages, and tensor parallelism under PP.

Same structure as the parity suite: each engine runs in its own subprocess
via _parity_worker.py and the parent compares JSON. Two topologies:

* PP=3 (3 GPUs): reads land on all three stages, the payload merge crosses
  three ranks instead of two, and the cross-stage write pulls from a
  non-adjacent stage (rank 2 reads rank 0's layer, past rank 1).
* TP=2 x PP=2 (4 GPUs): the reference is TP=2 with a single pipeline stage,
  so the comparison isolates what PP adds under sharding; saves must ship
  from each stage's TP-rank-0 and pull traffic rides per-TP-column groups.
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

if len(FREE_GPUS) < 3:
    pytest.skip(
        f"topology tests need at least 3 free GPUs, found {len(FREE_GPUS)}",
        allow_module_level=True,
    )

needs_four_gpus = pytest.mark.skipif(
    len(FREE_GPUS) < 4,
    reason=f"TP=2 x PP=2 needs 4 free GPUs, found {len(FREE_GPUS)}",
)

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_parity_worker.py")
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

PROMPT = "The Eiffel Tower is located in the city of"


@functools.lru_cache(maxsize=None)
def run(tp, pp, scenario, *extra):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_path = f.name
    log_path = output_path + ".log"
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(FREE_GPUS[: tp * pp])
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        env["PYTHONPATH"] = os.path.join(REPO_ROOT, "src")
        cmd = [
            sys.executable,
            WORKER,
            scenario,
            "--tp",
            str(tp),
            "--pp",
            str(pp),
            "--prompt",
            PROMPT,
            "--output",
            output_path,
            *extra,
        ]
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
                f"topology worker failed (scenario={scenario}, tp={tp}, pp={pp}, "
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
