"""PP=2 end-to-end smoke: cross-stage reads, writes, and saves on a real engine.

Needs 2 GPUs. The block reads an early layer (stage 0), a late layer (stage 1),
and the logits (last stage): on each rank one subset is local and the rest are
remote-owned, so every save crosses the merge with sentinels on one side.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest
import torch

GPU_COUNT = torch.cuda.device_count()

pytestmark = pytest.mark.skipif(GPU_COUNT < 2, reason="PP=2 needs 2 GPUs")

PROMPT = "The Eiffel Tower is located in the city of"


@pytest.fixture(scope="module")
def vllm_pp2():
    from nnsight.modeling.vllm import VLLM

    return VLLM(
        "Qwen/Qwen2.5-0.5B",
        pipeline_parallel_size=2,
        gpu_memory_utilization=0.1,
        dispatch=True,
    )


def test_cross_stage_reads_and_logits(vllm_pp2):
    model = vllm_pp2
    with model.trace(PROMPT, temperature=0.0, max_tokens=1):
        early = model.model.layers[2].output.save()
        late = model.model.layers[20].output.save()
        logits = model.logits.save()

    # Layer outputs are (hidden, residual) tuples on Qwen2; every slot must be
    # a real tensor after the merge — a sentinel here means a stage's
    # contribution was dropped.
    for name, value in (("early", early), ("late", late)):
        hidden = value[0] if isinstance(value, tuple) else value
        assert isinstance(hidden, torch.Tensor), (name, type(value))
        assert torch.isfinite(hidden.float()).all(), name
    assert isinstance(logits, torch.Tensor) and logits.shape[-1] > 100_000 // 2
    city = model.tokenizer.decode(logits[-1].argmax(dim=-1)).strip()
    assert city == "Paris", city


def test_cross_stage_write_changes_logits(vllm_pp2):
    model = vllm_pp2
    with model.trace(PROMPT, temperature=0.0, max_tokens=1):
        clean = model.logits.save()
    with model.trace(PROMPT, temperature=0.0, max_tokens=1):
        # A stage-0-owned write, replicated on both ranks: applied by the
        # owner, absorbed by the non-owner.
        hidden = model.model.layers[2].output[0]
        model.model.layers[2].output = (hidden * 0,) + tuple(
            model.model.layers[2].output[1:]
        )
        zeroed = model.logits.save()
    assert not torch.equal(clean, zeroed)


def test_generation_with_per_step_saves(vllm_pp2):
    model = vllm_pp2
    with model.trace(PROMPT, temperature=0.0, max_tokens=4) as tracer:
        import nnsight

        steps = nnsight.save([])
        for _ in tracer.iter[:4]:
            steps.append(model.model.layers[20].output[0])
    assert len(steps) == 4
    for step_value in steps:
        assert isinstance(step_value, torch.Tensor)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-x", "-p", "no:cacheprovider"]))
