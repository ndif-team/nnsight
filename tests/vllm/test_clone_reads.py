"""``NNSIGHT_VLLM_CLONE_READS`` on a real engine: one copy per visit, edits copied back.

Without it a read is a view of vLLM's live buffer, and Qwen's next decoder layer
runs ``fused_add_rms_norm`` in place over this layer's (hidden, residual) pair, so
an uncloned read of a layer output comes back holding the next layer's values.

Each engine is its own, built with the variable set, because the worker reads it
once, when it loads the model. At two ranks ``gate_up_proj`` is sharded: it is
gathered into a fresh tensor rather than copied, and is read and edited too.
"""

import pytest
import torch

pytest.importorskip("vllm")

GPU_COUNT = torch.cuda.device_count()
LAYER = 5
LOCATIONS = {
    "layer": lambda model: model.model.layers[LAYER],
    "gate_up_proj": lambda model: model.model.layers[LAYER].mlp.gate_up_proj,
}
ENGINES = ["vllm_qwen_cloned", "vllm_qwen_cloned_tp"]


def _build(tensor_parallel_size):
    from nnsight.modeling.vllm import VLLM
    from nnsight.modeling.vllm.interleaver import CLONE_READS

    with pytest.MonkeyPatch.context() as env:
        env.setenv(CLONE_READS, "1")
        return VLLM(
            "Qwen/Qwen2.5-0.5B",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.2,
            dispatch=True,
        )


@pytest.fixture(scope="module")
def vllm_qwen_cloned():
    if GPU_COUNT < 1:
        pytest.skip("vLLM tests need a GPU")
    return _build(1)


@pytest.fixture(scope="module")
def vllm_qwen_cloned_tp():
    if GPU_COUNT < 2:
        pytest.skip("tensor-parallel tests need >=2 GPUs")
    return _build(2)


@torch.no_grad()
@pytest.mark.parametrize("location", LOCATIONS)
@pytest.mark.parametrize("engine", ENGINES)
def test_a_kept_read_equals_a_cloned_one(request, engine, location, ET_prompt):
    model = request.getfixturevalue(engine)
    module = LOCATIONS[location](model)
    with model.trace(ET_prompt, temperature=0.0, top_p=1):
        kept = module.output[0].save()
        cloned = module.output[0].clone().save()
    assert torch.equal(kept, cloned)


@torch.no_grad()
@pytest.mark.parametrize("location", LOCATIONS)
@pytest.mark.parametrize("engine", ENGINES)
def test_an_in_place_edit_lands(request, engine, location, ET_prompt):
    model = request.getfixturevalue(engine)
    module = LOCATIONS[location](model)
    with model.trace(ET_prompt, temperature=0.0, top_p=1):
        clean = model.logits.save()
    with model.trace(ET_prompt, temperature=0.0, top_p=1):
        module.output[0][:] = 0
        in_place = model.logits.save()
    with model.trace(ET_prompt, temperature=0.0, top_p=1):
        out = module.output
        module.output = (torch.zeros_like(out[0]), *out[1:])
        swapped = model.logits.save()
    assert not torch.equal(in_place, clean)
    assert torch.equal(in_place, swapped)
