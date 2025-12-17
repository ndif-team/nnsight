import pytest
import nnsight
import torch
from typing import TYPE_CHECKING

try:
    from nnsight.modeling.vllm import VLLM
except Exception as e:
    pytest.skip(f"Skipping VLLM tests: \n{e}", allow_module_level=True)


@pytest.fixture(scope="module")
def tp(request):
    tp = request.config.getoption("--tp")
    if tp > torch.cuda.device_count() or tp < 1:
        pytest.exit("--tp can't be higher than the number of availale GPUs.")
    return tp


@pytest.fixture(scope="module")
def vllm_gpt2(tp: int):
    return VLLM(
        "gpt2", tensor_parallel_size=tp, gpu_memory_utilization=0.1, dispatch=True
    )


@pytest.fixture
def ET_prompt():
    return "The Eiffel Tower is located in the city of"


@pytest.fixture
def MSG_prompt():
    return "Madison Square Garden is located in the city of"


@torch.no_grad()
def test_single_logit(vllm_gpt2, ET_prompt: str):
    with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1):
        logits = vllm_gpt2.logits.output.save()

    next_token = vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1))
    assert next_token == " Paris"


@torch.no_grad()
def test_request_cleanup(vllm_gpt2, ET_prompt: str, MSG_prompt: str):
    with vllm_gpt2.trace() as tracer:
        with tracer.invoke(ET_prompt):
            pass

        with tracer.invoke(MSG_prompt):
            pass

    with vllm_gpt2.trace(MSG_prompt, temperature=0.0, top_p=1, max_tokens=3) as tracer:
        logits = list().save()
        with tracer.iter[0:3]:
            logits.append(vllm_gpt2.logits.output)

    assert vllm_gpt2.tokenizer.batch_decode(
        [logit.argmax(dim=-1) for logit in logits]
    ) == [" New", " York", " City"]


@torch.no_grad()
def test_multi_token_generation(vllm_gpt2, MSG_prompt: str):
    with vllm_gpt2.trace(
        MSG_prompt, temperature=0.0, top_p=1.0, max_tokens=3
    ) as tracer:
        logits = list().save()
        with tracer.iter[0:3]:
            logits.append(vllm_gpt2.logits.output)

    assert vllm_gpt2.tokenizer.batch_decode(
        [logit.argmax(dim=-1) for logit in logits]
    ) == [" New", " York", " City"]


@torch.no_grad()
def test_sampling(vllm_gpt2, MSG_prompt: str):
    with vllm_gpt2.trace(max_tokens=3) as tracer:
        with tracer.invoke(MSG_prompt, temperature=0.8, top_p=0.95):
            samples_2 = list().save()
            with tracer.iter[0:3]:
                samples_2.append(vllm_gpt2.samples.output.item())

        with tracer.invoke(MSG_prompt, temperature=0.0, top_p=1.0):
            samples_1 = list().save()
            with tracer.iter[0:3]:
                samples_1.append(vllm_gpt2.samples.output.item())

    assert vllm_gpt2.tokenizer.batch_decode(
        samples_1
    ) != vllm_gpt2.tokenizer.batch_decode(samples_2)


@torch.no_grad()
def test_max_token_generation(vllm_gpt2, ET_prompt: str):
    with vllm_gpt2.trace(ET_prompt, max_tokens=10) as tracer:
        logits = list().save()
        with tracer.all():
            logits.append(vllm_gpt2.logits.output)

    assert len(logits) == 10


@torch.no_grad()
def test_intervention(vllm_gpt2, ET_prompt: str):
    with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1) as tracer:
        out = vllm_gpt2.transformer.h[-2].mlp.output.clone()
        out[:] = 0

        vllm_gpt2.transformer.h[-2].mlp.output = out
        hs = vllm_gpt2.transformer.h[-2].mlp.output.save()
        logits = vllm_gpt2.logits.output.save()

    next_token = vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1))
    assert torch.all(hs == 0)
    assert next_token == " London"


@torch.no_grad()
def test_swap_intervention(vllm_gpt2, ET_prompt: str):
    with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1) as tracer:
        vllm_gpt2.transformer.h[-2].mlp.output = torch.zeros_like(
            vllm_gpt2.transformer.h[-2].mlp.output
        )
        hs = vllm_gpt2.transformer.h[-2].mlp.output.save()
        logits = vllm_gpt2.logits.output.save()

    next_token = vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1))
    assert next_token == " London"
    assert torch.all(hs == 0)


@torch.no_grad()
def test_batched_intervention(
    vllm_gpt2,
    ET_prompt: str,
):
    with vllm_gpt2.trace(temperature=0.0, top_p=1) as tracer:

        with tracer.invoke(ET_prompt):
            clean_hs = vllm_gpt2.transformer.h[-2].mlp.output.save()
            clean_logits = vllm_gpt2.logits.output.save()
        with tracer.invoke(ET_prompt):
            out = vllm_gpt2.transformer.h[-2].mlp.output[:].clone()
            out[:] = 0
            vllm_gpt2.transformer.h[-2].mlp.output = out
            corrupted_hs = vllm_gpt2.transformer.h[-2].mlp.output.save()
            corrupted_logits = vllm_gpt2.logits.output.save()

    clean_token = vllm_gpt2.tokenizer.decode(clean_logits.argmax(dim=-1))
    corrupted_token = vllm_gpt2.tokenizer.decode(corrupted_logits.argmax(dim=-1))

    assert not torch.all(clean_hs == 0)
    assert torch.all(corrupted_hs == 0)
    assert clean_token == " Paris"
    assert corrupted_token == " London"


@torch.no_grad()
def test_batched_multi_token_generation(vllm_gpt2, ET_prompt: str, MSG_prompt: str):
    max_token_1: int = 3
    max_token_2: int = 5

    num_prompts_1: int = 2
    num_prompts_2: int = 1

    with vllm_gpt2.trace() as tracer:
        with tracer.invoke([MSG_prompt, ET_prompt], max_tokens=max_token_1):
            MSG_ET_hs = list().save()
            MSG_ET_logits = list().save()
            MSG_ET_samples = list().save()
            with tracer.iter[:]:
                MSG_ET_hs.append(vllm_gpt2.transformer.h[5].output)
                MSG_ET_logits.append(vllm_gpt2.logits.output)
                MSG_ET_samples.append(vllm_gpt2.samples.output)
        with tracer.invoke(MSG_prompt, max_tokens=max_token_2):
            MSG_hs = list().save()
            MSG_logits = list().save()
            MSG_samples = list().save()
            with tracer.iter[:]:
                MSG_hs.append(vllm_gpt2.transformer.h[5].output)
                MSG_logits.append(vllm_gpt2.logits.output)
                MSG_samples.append(vllm_gpt2.samples.output)

    assert len(MSG_ET_hs) == max_token_1
    assert all(hs.shape[0] == num_prompts_1 for hs in MSG_ET_hs[1:])

    assert len(MSG_ET_logits) == max_token_1
    assert all(logit.shape[0] == num_prompts_1 for logit in MSG_ET_logits)

    assert len(MSG_ET_samples) == max_token_1
    assert all(sample.shape[0] == num_prompts_1 for sample in MSG_ET_samples)

    assert len(MSG_hs) == max_token_2
    assert all(hs.shape[0] == num_prompts_2 for hs in MSG_hs[1:])

    assert len(MSG_logits) == max_token_2
    assert all(logit.shape[0] == num_prompts_2 for logit in MSG_logits)

    assert len(MSG_samples) == max_token_2
    assert all(sample.shape[0] == num_prompts_2 for sample in MSG_samples)


@torch.no_grad()
def test_batched_multi_token_generation_with_iter(
    vllm_gpt2, ET_prompt: str, MSG_prompt: str
):
    with vllm_gpt2.trace(max_tokens=10) as tracer:
        with tracer.invoke(ET_prompt):
            ET_logits = list().save()
            with tracer.iter[1:7]:
                ET_logits.append(vllm_gpt2.logits.output)
        with tracer.invoke(MSG_prompt, max_tokens=5):
            MSG_logits = list().save()
            with tracer.iter[:5]:
                MSG_logits.append(vllm_gpt2.logits.output)

    assert len(ET_logits) == 6
    assert len(MSG_logits) == 5


@torch.no_grad()
def test_mutli_token_generation_with_intervention(tp, vllm_gpt2, MSG_prompt: str):
    with vllm_gpt2.trace(
        MSG_prompt, temperature=0.0, top_p=1.0, max_tokens=5
    ) as tracer:
        logits = list().save()
        hs_list = list().save()
        with tracer.iter[:] as it:
            if it == 2:
                out = vllm_gpt2.transformer.h[-2].output.clone()
                out[0][:] = 0
                vllm_gpt2.transformer.h[-2].output = out

            hs_list.append(vllm_gpt2.transformer.h[-2].output[0])
            logits.append(vllm_gpt2.logits.output)

    assert [torch.all(hs == 0) for hs in hs_list] == [False, False, True, False, False]

    if tp == 1:
        assert vllm_gpt2.tokenizer.decode(logits[2].argmax(dim=-1)) != " City"


@torch.no_grad()
def test_invoker_group_batching(vllm_gpt2, ET_prompt: str, MSG_prompt: str):

    max_tokens_1 = 1
    max_tokens_2 = 2
    max_tokens_3 = 3

    MSG_logits = list()
    ET_logits = list()
    two_prompts_logits = list()
    all_logits = list()

    with vllm_gpt2.trace() as tracer:

        with tracer.invoke(MSG_prompt, max_tokens=max_tokens_1):

            with tracer.iter[:]:
                MSG_logits.append(vllm_gpt2.logits.output)

        with tracer.invoke():

            with tracer.all():
                all_logits.append(vllm_gpt2.logits.output)

        with tracer.invoke([ET_prompt, MSG_prompt], max_tokens=max_tokens_3):

            with tracer.all():
                two_prompts_logits.append(vllm_gpt2.logits.output)

        with tracer.invoke(ET_prompt, max_tokens=max_tokens_2):

            with tracer.iter[:]:
                ET_logits.append(vllm_gpt2.logits.output)

    # each invoker has the correct number of logits
    assert len(MSG_logits) == max_tokens_1
    assert len(ET_logits) == max_tokens_2
    assert len(two_prompts_logits) == max_tokens_3
    assert len(all_logits) == max_tokens_3

    # check correctness of prompt-less invoker
    assert (
        all_logits[0].shape[0] == 4
        and all_logits[1].shape[0] == 3
        and all_logits[2].shape[0] == 2
    )

    # iter 0
    assert torch.equal(all_logits[0][0], MSG_logits[0][0])
    assert torch.equal(all_logits[0][1:3], two_prompts_logits[0][:2])
    assert torch.equal(all_logits[0][3], ET_logits[0][0])

    # iter 1
    assert torch.equal(all_logits[1][0:2], two_prompts_logits[1])
    assert torch.equal(all_logits[1][2], ET_logits[1][0])

    # iter 2
    assert torch.equal(all_logits[2], two_prompts_logits[2])


@torch.no_grad()
def test_tensor_parallelism(tp, vllm_gpt2, ET_prompt: str):
    if tp < 2:
        pytest.skip("Skipping test for tp>1!")

    with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1.0):
        value_tuple = vllm_gpt2.transformer.h[5].mlp.c_fc.output
        value = value_tuple[0].clone()
        value[:, 2000:] = 0
        value_tuple = (value, *value_tuple[1:])
        vllm_gpt2.transformer.h[5].mlp.c_fc.output = value_tuple
        hs = vllm_gpt2.transformer.h[5].mlp.c_fc.output[0].save()
        logit = vllm_gpt2.logits.output.save()
        next_token = vllm_gpt2.tokenizer.decode(logit.argmax(dim=-1)).save()

    assert next_token != " Paris"
    assert hs.shape == torch.Size([11, 3072])
    assert torch.all(hs[:, 2000:] == 0)
