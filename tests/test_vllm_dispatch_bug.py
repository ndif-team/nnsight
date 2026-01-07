"""Test for VLLM dispatch=False tracing bug."""
import pytest
import torch

try:
    from nnsight.modeling.vllm import VLLM
except Exception as e:
    pytest.skip(f"Skipping VLLM tests: \n{e}", allow_module_level=True)


@pytest.fixture(scope="module")
def vllm_gpt2_no_dispatch():
    """VLLM model initialized without dispatch=True."""
    return VLLM("gpt2", tensor_parallel_size=1, gpu_memory_utilization=0.1)


@torch.no_grad()
def test_trace_without_dispatch(vllm_gpt2_no_dispatch):
    """Tracing should work even when dispatch=False at init time."""
    model = vllm_gpt2_no_dispatch

    assert not model.dispatched, "Model should not be dispatched initially"
    assert model.vllm_entrypoint is None, "vllm_entrypoint should be None initially"

    with model.trace("The Eiffel Tower is located in the city of", temperature=0.0, top_p=1):
        logits = model.logits.output.save()

    assert model.dispatched, "Model should be dispatched after trace"
    assert model.vllm_entrypoint is not None, "vllm_entrypoint should exist after trace"

    next_token = model.tokenizer.decode(logits.argmax(dim=-1))
    assert next_token == " Paris"
