"""Which scheduler a PP=2 engine builds, however vLLM settled async scheduling.

With async scheduling the later stages' values go back by broadcast and the
engine keeps vLLM's AsyncScheduler; without it they go back through
NNsightScheduler. vLLM settles async scheduling after nnsight's arguments are
read, including by picking Ray itself inside a Ray placement group, so each
case is built for real and its engine core's scheduler looked at. A block
that writes a first-stage layer from the last stage's logits of the step
before must give the same values on every route.

Each case runs in a fresh process (``_scheduler_worker.py``).
"""

import functools
import json
import os
import subprocess
import sys

import pytest
import torch

from _support import free_gpus

pytest.importorskip("vllm")

pytestmark = pytest.mark.gpu

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "..", "src")


@functools.lru_cache(maxsize=None)
def run(scenario: str, scale: str = "1e-4") -> dict:
    if torch.cuda.device_count() < 2:
        pytest.skip("PP=2 needs 2 GPUs")
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES") or ",".join(free_gpus()[:2])
    if len(gpus.split(",")) < 2:
        pytest.skip(f"PP=2 needs 2 free GPUs, found {gpus!r}")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpus, "PYTHONPATH": SRC, "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
    done = subprocess.run(
        [sys.executable, os.path.join(HERE, "_scheduler_worker.py"), scenario, scale],
        env=env, capture_output=True, text=True, timeout=1200,
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
    assert lines, f"{scenario} produced no result (rc={done.returncode}):\n{done.stderr[-4000:]}"
    return json.loads(lines[-1][len("RESULT "):])


def test_the_first_stage_write_changes_what_the_model_computes():
    """Without this, equal values across routes could mean the write never landed."""
    assert run("default")["sums"][1:] != run("default", "0")["sums"][1:]


def test_with_async_scheduling_the_engine_keeps_vllms_async_scheduler():
    result = run("default")
    assert result["async_scheduling"] is True
    assert result["scheduler"] == "AsyncScheduler"


def test_without_async_scheduling_the_values_go_back_through_nnsights_scheduler():
    result = run("async_off")
    assert result["async_scheduling"] is False
    assert result["scheduler"] == "NNsightScheduler"
    assert result["sums"] == run("default")["sums"]


def test_ray_picked_by_vllm_inside_a_placement_group_gets_nnsights_scheduler():
    pytest.importorskip("ray")
    result = run("ray_auto")
    assert result["backend"] == "ray" and result["async_scheduling"] is False
    assert result["scheduler"] == "NNsightScheduler"
    assert result["sums"] == run("default")["sums"]
