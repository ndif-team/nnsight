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
def test_prompts_that_do_not_fit_one_step_wait_rather_than_split(vllm_gpt2):
    # gpt2's context is 1024, so the budget is at least that; six 400-token
    # prompts cannot all prefill in one step. Every invoke still reads its
    # whole prompt at layer 0: the ones that did not fit ran a step later.
    model = vllm_gpt2
    prompts = [_prompt(model, 400) for _ in range(6)]
    with model.trace(temperature=0.0, top_p=1.0, max_tokens=1) as tracer:
        with tracer.invoke(prompts[0]):
            a = model.transformer.h[0].output.clone().save()
        with tracer.invoke(prompts[1]):
            b = model.transformer.h[0].output.clone().save()
        with tracer.invoke(prompts[2]):
            c = model.transformer.h[0].output.clone().save()
        with tracer.invoke(prompts[3]):
            d = model.transformer.h[0].output.clone().save()
        with tracer.invoke(prompts[4]):
            e = model.transformer.h[0].output.clone().save()
        with tracer.invoke(prompts[5]):
            f = model.transformer.h[0].output.clone().save()
    for h in (a, b, c, d, e, f):
        assert h.shape[0] == 400, h.shape


@pytest.fixture(scope="module")
def vllm_gpt2_chunking():
    from nnsight.modeling.vllm import VLLM

    if GPU_COUNT < 1:
        pytest.skip("vLLM tests need a GPU")
    # Chunking asked for explicitly, with a budget any real prompt exceeds.
    return VLLM(
        "gpt2",
        gpu_memory_utilization=0.1,
        dispatch=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=64,
        max_num_seqs=8,
    )


@torch.no_grad()
def test_a_chunked_prompt_is_refused_when_chunking_is_on(vllm_gpt2_chunking):
    model = vllm_gpt2_chunking
    with pytest.raises(RuntimeError, match="split across steps by chunked prefill"):
        with model.trace(_prompt(model, 200), temperature=0.0, top_p=1.0, max_tokens=1):
            model.transformer.h[0].output.clone().save()


@torch.no_grad()
def test_a_prompt_within_the_budget_still_traces(vllm_gpt2_chunking):
    model = vllm_gpt2_chunking
    with model.trace(_prompt(model, 32), temperature=0.0, top_p=1.0, max_tokens=1):
        h = model.transformer.h[0].output.clone().save()
    assert h.shape[0] == 32
