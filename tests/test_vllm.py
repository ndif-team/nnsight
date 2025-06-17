import pytest
import nnsight
import torch
from typing import TYPE_CHECKING

try:
    from nnsight.modeling.vllm import VLLM
except:
    pytest.skip("Skipping VLLM tests", allow_module_level=True)

pytest.skip("Skipping VLLM tests", allow_module_level=True)

if TYPE_CHECKING:
    from nnsight.tracing.graph import Graph

from nnsight.tracing.backends import Backend
from nnsight.tracing.protocols import StopProtocol

class AssertSavedLenBackend(Backend):
    
    def __init__(self, len:int) -> None:
        self.len = len
    
    def __call__(self, graph: "Graph") -> None:
        
        try:
         
            graph.nodes[-1].execute()
            
        except StopProtocol.StopException:
            
            pass
        
        finally:
            
            assert self.len == len([node for node in graph.nodes if node.done])
                            
            graph.nodes.clear()
            graph.stack.clear()


@pytest.fixture(scope="module")
def tp(request):
    tp = request.config.getoption("--tp")
    if tp > torch.cuda.device_count() or tp < 1:
        pytest.exit("--tp can't be higher than the number of availale GPUs.")
    return tp

@pytest.fixture(scope="module")
def vllm_gpt2(tp: int):
    return VLLM("gpt2", tensor_parallel_size=tp, dispatch=True)

@pytest.fixture
def ET_prompt():
    return "The Eiffel Tower is located in the city of"

@pytest.fixture
def MSG_prompt():
    return "Madison Square Garden is located in the city of"


def test_single_logit(vllm_gpt2, ET_prompt: str):
    with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1, backend=AssertSavedLenBackend(1)):
        logits = vllm_gpt2.logits.output.save()

    next_token = vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1))
    assert next_token == " Paris"


def test_multi_token_generation(vllm_gpt2, MSG_prompt: str):
    with vllm_gpt2.trace(MSG_prompt, temperature=0.0, top_p=1.0, max_tokens=3):
        logits = nnsight.list().save()
        for ii in range(3):
            logits.append(vllm_gpt2.logits.output)
            vllm_gpt2.logits.next()

    assert vllm_gpt2.tokenizer.batch_decode([logit.argmax(dim=-1) for logit in logits]) == [" New", " York", " City"]


def test_sampling(vllm_gpt2, MSG_prompt: str):
    with vllm_gpt2.trace(max_tokens=3) as tracer:
        with tracer.invoke(MSG_prompt, temperature=0.0, top_p=1.0, max_tokens=3):
            samples_1 = nnsight.list().save()
            for ii in range(3):
                samples_1.append(vllm_gpt2.samples.output)
                vllm_gpt2.samples.next()
        with tracer.invoke(MSG_prompt, temperature=0.8, top_p=0.95):
            samples_2 = nnsight.list().save()
            for ii in range(3):
                samples_2.append(vllm_gpt2.samples.output)
                vllm_gpt2.samples.next()

    assert vllm_gpt2.tokenizer.batch_decode(samples_1) == [" New", " York", " City"]
    assert vllm_gpt2.tokenizer.batch_decode(samples_2) == [" Richmond", " on", " the"]


""" def test_max_token_generation(vllm_gpt2, ET_prompt: str):
    with vllm_gpt2.trace(ET_prompt, max_tokens=10):
        logits = nnsight.list().save()
        with vllm_gpt2.logits.all():
            logits.append(vllm_gpt2.logits.output)

    assert len(logits) == 10 """


""" def test_sampling(vllm_gpt2, ET_prompt:str):
    with vllm_gpt2.trace(ET_prompt, temperature=0.8, top_p=0.95, max_tokens=3):
        samples = nnsight.list().save()
        with vllm_gpt2.sample.all():
            li.append(vllm_gpt2.sample.output)

    samples = vllm_gpt2.batch_decode([sample.argmax(dim=-1) for sample in samples])
    assert samples == [' Canary', ' Wh', 'arf'] """


def test_intervention(vllm_gpt2, ET_prompt: str):
    with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1, backend=AssertSavedLenBackend(2)) as tracer:
        vllm_gpt2.transformer.h[-2].mlp.output[:] = 0
        hs = vllm_gpt2.transformer.h[-2].mlp.output.save()
        logits = vllm_gpt2.logits.output.save()

    next_token = vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1))
    assert next_token == " London"
    assert torch.all(hs == 0)


def test_swap_intervention(vllm_gpt2, ET_prompt: str):
    with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1, backend=AssertSavedLenBackend(2)) as tracer:
        vllm_gpt2.transformer.h[-2].mlp.output = torch.zeros_like(vllm_gpt2.transformer.h[-2].mlp.output)
        hs = vllm_gpt2.transformer.h[-2].mlp.output.save()
        logits = vllm_gpt2.logits.output.save()

    next_token = vllm_gpt2.tokenizer.decode(logits.argmax(dim=-1))
    assert next_token == " London"
    assert torch.all(hs == 0)


def test_batched_intervention(vllm_gpt2, ET_prompt: str,):
    with vllm_gpt2.trace(temperature=0.0, top_p=1, backend=AssertSavedLenBackend(4)) as tracer:

        with tracer.invoke(ET_prompt):
            clean_hs = vllm_gpt2.transformer.h[-2].mlp.output.save()
            clean_logits = vllm_gpt2.logits.output.save()
        with tracer.invoke(ET_prompt):
            vllm_gpt2.transformer.h[-2].mlp.output[:] = 0
            corrupted_hs = vllm_gpt2.transformer.h[-2].mlp.output.save()
            corrupted_logits = vllm_gpt2.logits.output.save()

    clean_token = vllm_gpt2.tokenizer.decode(clean_logits.argmax(dim=-1))
    corrupted_token = vllm_gpt2.tokenizer.decode(corrupted_logits.argmax(dim=-1))

    assert clean_token == " Paris"
    assert corrupted_token == " London"
    assert not torch.all(clean_hs == 0)
    assert torch.all(corrupted_hs == 0)


def test_batched_multi_token_generation(vllm_gpt2, ET_prompt: str, MSG_prompt: str):
    max_token_1: int = 3
    max_token_2: int = 5

    num_prompts_1: int = 2
    num_prompts_2: int = 1

    with vllm_gpt2.trace() as tracer:
        with tracer.invoke([MSG_prompt, ET_prompt], max_tokens=max_token_1):
            MSG_ET_hs = nnsight.list().save()
            MSG_ET_logits = nnsight.list().save()
            MSG_ET_samples = nnsight.list().save()
            for ii in range(max_token_1):
                MSG_ET_hs.append(vllm_gpt2.transformer.h[5].output)
                vllm_gpt2.transformer.h[5].next()
                MSG_ET_logits.append(vllm_gpt2.logits.output)
                vllm_gpt2.logits.next()
                MSG_ET_samples.append(vllm_gpt2.samples.output)
                vllm_gpt2.samples.next()
        with tracer.invoke(MSG_prompt, max_tokens=max_token_2):
            MSG_hs = nnsight.list().save()
            MSG_logits = nnsight.list().save()
            MSG_samples = nnsight.list().save()
            for ii in range(max_token_2):
                MSG_hs.append(vllm_gpt2.transformer.h[5].output)
                vllm_gpt2.transformer.h[5].next()
                MSG_logits.append(vllm_gpt2.logits.output)
                vllm_gpt2.logits.next()
                MSG_samples.append(vllm_gpt2.samples.output)
                vllm_gpt2.samples.next()

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


""" def test_batched_multi_token_generation_with_iter(vllm_gpt2, ET_prompt: str, MSG_prompt: str):
    with vllm_gpt2.trace(max_tokens=10) as tracer:
        with tracer.invoke(ET_prompt):
            ET_logits = nnsight.list().save()
            with vllm_gpt2.logits.iter[:3]:
                ET_logits.append(vllm_gpt2.logits.output)
            #vllm_gpt2.output.save()
        with tracer.invoke(MSG_prompt, max_tokens=5):
            MSG_logits = nnsight.list().save()
            with vllm_gpt2.logits.iter[:5]:
                MSG_logits.append(vllm_gpt2.logits.output)

    assert len(ET_logits.value) == 3
    assert len(MSG_logits.value) == 5 """


def test_mutli_token_generation_with_intervention(tp, vllm_gpt2, MSG_prompt: str):
    with vllm_gpt2.trace(MSG_prompt, temperature=0.0, top_p=1.0, max_tokens=5) as tracer:
        logits = nnsight.list().save()
        hs_list = nnsight.list().save()
        for ii in range(5):
            if ii == 2:
                vllm_gpt2.transformer.h[-2].output[0][:] = 0
            hs_list.append(vllm_gpt2.transformer.h[-2].output[0])
            vllm_gpt2.transformer.h[-2].next()
            logits.append(vllm_gpt2.logits.output)
            vllm_gpt2.logits.next()

    assert [torch.all(hs == 0) for hs in hs_list] == [False, False, True, False, False]

    if tp == 1:
        assert vllm_gpt2.tokenizer.batch_decode([logit.argmax(dim=-1) for logit in logits]) == [' New', ' York', '\n', '\n', 'The']


""" def test_multi_referenced_module(vllm_gpt2, ET_prompt: str):
    with vllm_gpt2.trace(ET_prompt):
        act_in = vllm_gpt2.transformer.h[0].mlp.act.input.save()
        vllm_gpt2.transformer.h[0].mlp.act.next()
        act_in_other = vllm_gpt2.transformer.h[1].mlp.act.input.save()

    assert not torch.equal(act_in, act_in_other) """


def test_tensor_parallelism(tp, vllm_gpt2, ET_prompt: str):
    if tp < 2:
        pytest.skip("Skipping test for tp>1!")

    with vllm_gpt2.trace(ET_prompt, temperature=0.0, top_p=1.0):
        vllm_gpt2.transformer.h[5].mlp.c_fc.output[0][:, 2000:] = 0
        hs = vllm_gpt2.transformer.h[5].mlp.c_fc.output[0].save()
        logit = vllm_gpt2.logits.output.save()

    next_token = vllm_gpt2.tokenizer.decode(logit.argmax(dim=-1))

    #assert next_token != " Paris"
    assert hs.shape == torch.Size([11, 3072])
    assert torch.all(hs[:, 2000:] == 0)
