import pytest
import torch

import nnsight
from nnsight.contexts.Runner import Runner
from nnsight.pydantics import RequestModel
from nnsight.tracing.Graph import Graph


@pytest.fixture(scope="module")
def gpt2(device: str):
    return nnsight.LanguageModel("gpt2", device_map=device, dispatch=True)


@pytest.fixture
def MSG_prompt():
    return "Madison Square Garden is located in the city of"


def _test_serialize(runner: Runner):
    request = RequestModel(
        args=runner.args,
        kwargs=runner.kwargs,
        repo_id=runner.model.repoid_path_clsname,
        generation=runner.generation,
        intervention_graph=runner.graph.nodes,
        batched_input=runner.batched_input,
    )

    request_json = request.model_dump(
        mode="json", exclude=["session_id", "received", "id"]
    )

    request2 = RequestModel(**request_json)
    request2.compile()

    assert isinstance(request2.intervention_graph, Graph)


def test_generation(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.generate(max_new_tokens=3) as generator:
        with generator.invoke(MSG_prompt) as invoker:
            pass

    output = gpt2.tokenizer.decode(generator.output[0])

    assert output == "Madison Square Garden is located in the city of New York City"

    _test_serialize(generator)


def test_save(gpt2: nnsight.LanguageModel):
    with gpt2.generate(max_new_tokens=1) as generator:
        with generator.invoke("Hello world") as invoker:
            hs = gpt2.transformer.h[-1].output[0].save()
            hs_input = gpt2.transformer.h[-1].input[0][0].save()

    assert hs.value is not None
    assert isinstance(hs.value, torch.Tensor)
    assert hs.value.ndim == 3

    assert hs_input.value is not None
    assert isinstance(hs_input.value, torch.Tensor)
    assert hs_input.value.ndim == 3

    _test_serialize(generator)


def test_set1(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.generate(max_new_tokens=1) as generator:
        with generator.invoke(MSG_prompt) as invoker:
            pre = gpt2.transformer.h[-1].output[0].clone().save()

            gpt2.transformer.h[-1].output[0][:] = 0

            post = gpt2.transformer.h[-1].output[0].save()

    output = gpt2.tokenizer.decode(generator.output[0])

    assert not (pre.value == 0).all().item()
    assert (post.value == 0).all().item()
    assert output != "Madison Square Garden is located in the city of New"

    _test_serialize(generator)


def test_set2(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.generate(max_new_tokens=1) as generator:
        with generator.invoke(MSG_prompt) as invoker:
            pre = gpt2.transformer.wte.output.clone().save()

            gpt2.transformer.wte.output = gpt2.transformer.wte.output * 0

            post = gpt2.transformer.wte.output.save()

    output = gpt2.tokenizer.decode(generator.output[0])

    assert not (pre.value == 0).all().item()
    assert (post.value == 0).all().item()
    assert output != "Madison Square Garden is located in the city of New"

    _test_serialize(generator)


def test_adhoc_module(gpt2: nnsight.LanguageModel):
    with gpt2.generate() as generator:
        with generator.invoke("The Eiffel Tower is in the city of") as invoker:
            hidden_states = gpt2.transformer.h[-1].output[0]
            hidden_states = gpt2.lm_head(gpt2.transformer.ln_f(hidden_states))
            tokens = torch.softmax(hidden_states, dim=2).argmax(dim=2).save()

    output = gpt2.tokenizer.decode(tokens.value[0])

    assert output == "\n-el Tower is a the middle centre Paris"

    _test_serialize(generator)


def test_embeddings_set1(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.generate(max_new_tokens=3) as generator:
        with generator.invoke(MSG_prompt) as invoker:
            embeddings = gpt2.transformer.wte.output

        with generator.invoke("_ _ _ _ _ _ _ _ _") as invoker:
            gpt2.transformer.wte.output = embeddings

    output1 = gpt2.tokenizer.decode(generator.output[0])
    output2 = gpt2.tokenizer.decode(generator.output[1])

    assert output1 == "Madison Square Garden is located in the city of New York City"
    assert output2 == "_ _ _ _ _ _ _ _ _ New York City"

    _test_serialize(generator)


def test_embeddings_set2(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.generate(max_new_tokens=3) as generator:
        with generator.invoke(MSG_prompt) as invoker:
            embeddings = gpt2.transformer.wte.output.save()

    output1 = gpt2.tokenizer.decode(generator.output[0])

    with gpt2.generate(max_new_tokens=3) as generator:
        with generator.invoke("_ _ _ _ _ _ _ _ _") as invoker:
            gpt2.transformer.wte.output = embeddings.value

    output2 = gpt2.tokenizer.decode(generator.output[0])

    assert output1 == "Madison Square Garden is located in the city of New York City"
    assert output2 == "_ _ _ _ _ _ _ _ _ New York City"

    _test_serialize(generator)


def test_retain_grad(gpt2: nnsight.LanguageModel):
    with gpt2.forward(inference=False) as runner:
        with runner.invoke("Hello World") as invoker:
            hidden_states = gpt2.transformer.h[-1].output[0].save()
            hidden_states.retain_grad()

            logits = gpt2.lm_head.output

            logits.sum().backward()

    _test_serialize(runner)

    assert hidden_states.value.grad is not None


def test_grad(gpt2: nnsight.LanguageModel):
    with gpt2.forward(inference=False) as runner:
        with runner.invoke("Hello World") as invoker:
            hidden_states = gpt2.transformer.h[-1].output[0].save()
            hidden_states_grad = hidden_states.grad.save()
            hidden_states_grad[:] = 0

            logits = gpt2.lm_head.output

            logits.sum().backward()

    hidden_states.value

    assert (hidden_states_grad.value == 0).all().item()

    _test_serialize(runner)

    with gpt2.forward(inference=False) as runner:
        with runner.invoke("Hello World") as invoker:
            hidden_states = gpt2.transformer.h[-1].output[0].save()
            grad = hidden_states.grad.clone()
            grad[:] = 0
            hidden_states.grad = grad

            logits = gpt2.lm_head.output

            logits.sum().backward()

    hidden_states.value
    assert (hidden_states_grad.value == 0).all().item()

    _test_serialize(runner)
