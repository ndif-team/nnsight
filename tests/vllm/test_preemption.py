"""A request vLLM preempts mid-generation comes back whole.

The KV cache is sized well below what the batch needs, so vLLM has to evict
requests and recompute them from their prompts. A preempted request's worker
continues (vLLM replays the tokens it already saw inside one recompute step, and
``Requests.scope`` keeps the visits it sat out off its count), so what it saves
is what an uninterrupted run saves: one value per generated step, no error.

That accounting is what keeps a preempted request from looking like a loop that
outran it. A bounded ``tracer.iter`` whose count the request cannot reach is an
error, and the steps a request sits out of the batch are not steps it missed — so
each block below appends a sentinel after its loop, and the sentinel arriving is
the proof that the loop ended on its own count rather than being unwound.
"""

import pytest
import torch

pytest.importorskip("vllm")

GPU_COUNT = torch.cuda.device_count()
NEW = 48


@pytest.fixture(scope="module")
def starved_gpt2():
    from nnsight.modeling.vllm import VLLM

    if GPU_COUNT < 1:
        pytest.skip("vLLM tests need a GPU")
    # 40 blocks of 16 tokens: room for roughly one 400-token request at a time,
    # so four of them in flight guarantees preemption.
    return VLLM(
        "gpt2",
        gpu_memory_utilization=0.1,
        num_gpu_blocks_override=40,
        max_num_seqs=4,
        max_model_len=512,
        disable_log_stats=False,  # exposes vllm:num_preemptions, which the test asserts on
        dispatch=True,
    )


def _preemptions(model) -> int:
    """vLLM's preemption counter."""
    for metric in model.vllm_entrypoint.get_metrics():
        if metric.name == "vllm:num_preemptions":
            return int(metric.value)
    raise AssertionError("vLLM exposed no vllm:num_preemptions metric")


@torch.no_grad()
def test_preempted_requests_finish_with_every_step(starved_gpt2):
    model = starved_gpt2
    before = _preemptions(model)
    prompt = "The history of the city, told at length, begins with " + "and then more " * 60
    with model.trace(temperature=0.0, top_p=1.0, max_tokens=NEW, ignore_eos=True) as tracer:
        with tracer.invoke(prompt + "one."):
            a = list().save()
            for _ in tracer.iter[:NEW]:
                a.append(model.logits.argmax(-1).clone())
            a.append("after the loop")
        with tracer.invoke(prompt + "two."):
            b = list().save()
            for _ in tracer.iter[:NEW]:
                b.append(model.logits.argmax(-1).clone())
            b.append("after the loop")
        with tracer.invoke(prompt + "three."):
            c = list().save()
            for _ in tracer.iter[:NEW]:
                c.append(model.logits.argmax(-1).clone())
            c.append("after the loop")
        with tracer.invoke(prompt + "four."):
            d = list().save()
            for _ in tracer.iter[:NEW]:
                d.append(model.logits.argmax(-1).clone())
            d.append("after the loop")

    for saved in (a, b, c, d):
        assert len(saved) == NEW + 1
        assert saved[-1] == "after the loop", "a preempted request was read as a loop that outran it"
    assert _preemptions(model) > before, "the KV budget did not force a preemption; the test proves nothing"
