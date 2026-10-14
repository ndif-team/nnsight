"""A traced prompt is prefilled whole.

Chunked prefill splits a prompt across steps when it does not fit the step's
token budget; a block would then see a slice of its prompt on one step and the
rest on the next. nnsight engines turn it off (a prompt that does not fit waits a
step instead), and a caller who turns it back on is told, per request, when a
prompt was chunked rather than handed a slice.
"""

import pytest
import torch

pytest.importorskip("vllm")

GPU_COUNT = torch.cuda.device_count()


def _prompt(model, n: int) -> list[int]:
    ids = model.tokenizer("word " * (n + 8))["input_ids"]
    return ids[:n]


@torch.no_grad()
def test_prompts_that_do_not_fit_one_step_wait_rather_than_split(vllm_qwen):
    # Six prompts of a fifth of the step's token budget cannot all prefill in
    # one step. Every invoke still reads its whole prompt at layer 0: the ones
    # that did not fit ran a step later.
    model = vllm_qwen
    budget = model.vllm_entrypoint.llm_engine.vllm_config.scheduler_config.max_num_batched_tokens
    n = budget // 5
    prompts = [_prompt(model, n) for _ in range(6)]
    with model.trace(temperature=0.0, top_p=1.0, max_tokens=1) as tracer:
        with tracer.invoke(prompts[0]):
            a = model.model.layers[0].output[0].clone().save()
        with tracer.invoke(prompts[1]):
            b = model.model.layers[0].output[0].clone().save()
        with tracer.invoke(prompts[2]):
            c = model.model.layers[0].output[0].clone().save()
        with tracer.invoke(prompts[3]):
            d = model.model.layers[0].output[0].clone().save()
        with tracer.invoke(prompts[4]):
            e = model.model.layers[0].output[0].clone().save()
        with tracer.invoke(prompts[5]):
            f = model.model.layers[0].output[0].clone().save()
    for h in (a, b, c, d, e, f):
        assert h.shape[0] == n, h.shape


@pytest.fixture(scope="module")
def vllm_qwen_chunking():
    from nnsight.modeling.vllm import VLLM

    if GPU_COUNT < 1:
        pytest.skip("vLLM tests need a GPU")
    # Chunking asked for explicitly, with a budget any real prompt exceeds.
    return VLLM(
        "Qwen/Qwen2.5-0.5B",
        gpu_memory_utilization=0.1,
        dispatch=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=64,
        max_num_seqs=8,
    )


@torch.no_grad()
def test_a_chunked_prompt_is_refused_when_chunking_is_on(vllm_qwen_chunking):
    model = vllm_qwen_chunking
    with pytest.raises(RuntimeError, match="split across steps by chunked prefill"):
        with model.trace(_prompt(model, 200), temperature=0.0, top_p=1.0, max_tokens=1):
            model.model.layers[0].output[0].clone().save()


@torch.no_grad()
def test_a_prompt_within_the_budget_still_traces(vllm_qwen_chunking):
    model = vllm_qwen_chunking
    with model.trace(_prompt(model, 32), temperature=0.0, top_p=1.0, max_tokens=1):
        h = model.model.layers[0].output[0].clone().save()
    assert h.shape[0] == 32
