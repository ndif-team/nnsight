"""Pipeline parallelism on vLLM's Ray executor.

Under Ray the forward runs on one thread of the worker actor (Ray's compiled
graph) and ``collect_nnsight`` arrives on another. A block's greenlet and the
saved-id set belong to the forward thread, so collect must only read what the
forward thread recorded. These tests run the shapes that end a request with
a value from the last stage, which is where a collect-time resume would be
tempting, against a PP=2 engine driven over Ray.

Skipped unless Ray is installed and two GPUs are free.
"""

import os

import nnsight
import pytest
import torch

from _support import EARLY, LATE, PROMPT, free_gpus

pytest.importorskip("vllm")
pytest.importorskip("ray")

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def pp2_ray_engine():
    if torch.cuda.device_count() < 2:
        pytest.skip("PP=2 needs 2 GPUs")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        gpus = free_gpus()
        if len(gpus) < 2:
            pytest.skip(f"PP=2 needs 2 free GPUs, found {gpus}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus[:2])
    os.environ.setdefault("NNSIGHT_VLLM_CLONE_READS", "1")
    from nnsight.modeling.vllm import VLLM

    return VLLM(
        "Qwen/Qwen2.5-0.5B",
        pipeline_parallel_size=2,
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.12,
        dispatch=True,
    )


def test_saved_values_of_both_stages_come_home_over_ray(pp2_ray_engine):
    model = pp2_ray_engine
    with model.trace(PROMPT, temperature=0.0, max_tokens=1):
        early = model.model.layers[EARLY].output[0].save()
        late = model.model.layers[LATE].output[0].save()
        logits = model.logits.save()
    for value in (early, late, logits):
        assert isinstance(value, torch.Tensor), type(value)
    assert model.tokenizer.decode(logits[-1].argmax(dim=-1)).strip() == "Paris"


def test_a_used_last_stage_value_in_the_last_round_comes_home_over_ray(pp2_ray_engine):
    """The first stage's copy of the block ends parked on this value; the last
    stage's copy computes the line, and collect only reads what it recorded."""
    model = pp2_ray_engine
    with model.trace(PROMPT, temperature=0.0, max_tokens=1):
        total = float(model.model.layers[LATE].output[0].float().sum()).save()
    assert isinstance(total, float) and total == total


def test_per_step_used_reads_of_the_last_stage_come_home_over_ray(pp2_ray_engine):
    model = pp2_ray_engine
    with model.trace(PROMPT, temperature=0.0, max_tokens=3, ignore_eos=True) as tracer:
        sums = nnsight.save([])
        for _ in tracer.iter[:3]:
            sums.append(float(model.model.layers[LATE].output[0].float().sum()))
    assert len(sums) == 3 and all(isinstance(s, float) for s in sums)
