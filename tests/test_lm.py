import pytest
import torch

import nnsight
from nnsight.contexts.GraphBasedContext import GlobalTracingContext
from nnsight.contexts.Tracer import Tracer
from nnsight.schema.Request import RequestModel
from nnsight.tracing.Graph import Graph


@pytest.fixture(scope="module")
def gpt2(device: str):
    return nnsight.LanguageModel(
        "openai-community/gpt2", device_map=device, dispatch=True
    )


@pytest.fixture
def MSG_prompt():
    return "Madison Square Garden is located in the city of"


def _test_serialize(tracer: Tracer):
    with GlobalTracingContext.exit_global_tracing_context():
        request = RequestModel(
            object=tracer, model_key=tracer.remote_backend_get_model_key()
        )
        request_json = request.model_dump(
            mode="json", exclude=["session_id", "received", "id"]
        )

        request2 = RequestModel(**request_json)
        tracer = request2.deserialize(tracer.model)
    assert isinstance(tracer.graph, Graph)


@torch.no_grad()
def test_generation(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.generate(max_new_tokens=3, validate=True) as generator:
        with generator.invoke(MSG_prompt, scan=True) as invoker:
            output = gpt2.generator.output.save()

        _test_serialize(generator)

    output = gpt2.tokenizer.decode(output.value[0])

    assert (
        output
        == "Madison Square Garden is located in the city of New York City"
    )


@torch.no_grad()
def test_save(gpt2: nnsight.LanguageModel):
    with gpt2.generate("Hello world", validate=True, scan=True) as tracer:

        hs = gpt2.transformer.h[-1].output[0].save()
        hs_input = gpt2.transformer.h[-1].input.save()

        _test_serialize(tracer)

    assert hs.value is not None
    assert isinstance(hs.value, torch.Tensor)
    assert hs.value.ndim == 3

    assert hs_input.value is not None
    assert isinstance(hs_input.value, torch.Tensor)
    assert hs_input.value.ndim == 3


@torch.no_grad()
def test_set1(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.generate(validate=True) as tracer:
        with tracer.invoke(MSG_prompt, scan=True) as invoker:
            pre = gpt2.transformer.h[-1].output[0].clone().save()

            gpt2.transformer.h[-1].output[0][:] = 0

            post = gpt2.transformer.h[-1].output[0].save()

            output = gpt2.generator.output.save()

        _test_serialize(tracer)

    output = gpt2.tokenizer.decode(output.value[0])

    assert not (pre.value == 0).all().item()
    assert (post.value == 0).all().item()
    assert output != "Madison Square Garden is located in the city of New"


@torch.no_grad()
def test_set2(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.generate(validate=True) as generator:
        with generator.invoke(MSG_prompt, scan=True) as invoker:
            pre = gpt2.transformer.wte.output.clone().save()

            gpt2.transformer.wte.output = gpt2.transformer.wte.output * 0

            post = gpt2.transformer.wte.output.save()

            output = gpt2.generator.output.save()

        _test_serialize(generator)

    output = gpt2.tokenizer.decode(output.value[0])

    assert not (pre.value == 0).all().item()
    assert (post.value == 0).all().item()
    assert output != "Madison Square Garden is located in the city of New"


@torch.no_grad()
def test_adhoc_module(gpt2: nnsight.LanguageModel):
    with gpt2.generate(validate=True) as generator:
        with generator.invoke(
            "The Eiffel Tower is in the city of", scan=True
        ) as invoker:
            hidden_states = gpt2.transformer.h[-1].output[0]
            hidden_states = gpt2.lm_head(gpt2.transformer.ln_f(hidden_states))
            tokens = torch.softmax(hidden_states, dim=2).argmax(dim=2).save()

        _test_serialize(generator)

    output = gpt2.tokenizer.decode(tokens.value[0])

    assert output == "\n-el Tower is a the middle centre Paris"


@torch.no_grad()
def test_embeddings_set1(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.generate(max_new_tokens=3, validate=True) as generator:
        with generator.invoke(MSG_prompt, scan=True) as invoker:
            embeddings = gpt2.transformer.wte.output

            output1 = gpt2.generator.output.save()

        with generator.invoke("_ _ _ _ _ _ _ _ _", scan=True) as invoker:
            gpt2.transformer.wte.output = embeddings

            output2 = gpt2.generator.output.save()

        _test_serialize(generator)

    output1 = gpt2.tokenizer.decode(output1.value[0])
    output2 = gpt2.tokenizer.decode(output2.value[0])

    assert (
        output1
        == "Madison Square Garden is located in the city of New York City"
    )
    assert output2 == "_ _ _ _ _ _ _ _ _ New York City"


@torch.no_grad()
def test_embeddings_set2(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.generate(max_new_tokens=3, validate=True) as generator:
        with generator.invoke(MSG_prompt, scan=True) as invoker:
            embeddings = gpt2.transformer.wte.output.save()

            output = gpt2.generator.output.save()

    output1 = gpt2.tokenizer.decode(output.value[0])

    with gpt2.generate(max_new_tokens=3, validate=True) as generator:
        with generator.invoke("_ _ _ _ _ _ _ _ _", scan=True) as invoker:
            gpt2.transformer.wte.output = embeddings.value

            output = gpt2.generator.output.save()

        _test_serialize(generator)

    output2 = gpt2.tokenizer.decode(output.value[0])

    assert (
        output1
        == "Madison Square Garden is located in the city of New York City"
    )
    assert output2 == "_ _ _ _ _ _ _ _ _ New York City"


def test_retain_grad(gpt2: nnsight.LanguageModel):
    with gpt2.trace(validate=True) as tracer:
        with tracer.invoke("Hello World", scan=True) as invoker:
            hidden_states = gpt2.transformer.h[-1].output[0].save()
            hidden_states.retain_grad()

            logits = gpt2.lm_head.output

            logits.sum().backward()

        _test_serialize(tracer)

    assert hidden_states.value.grad is not None


def test_grad(gpt2: nnsight.LanguageModel):
    with gpt2.trace(validate=True) as tracer:
        with tracer.invoke("Hello World", scan=True) as invoker:
            hidden_states = gpt2.transformer.h[-1].output[0].save()
            hidden_states_grad = hidden_states.grad.save()
            hidden_states_grad[:] = 0

            logits = gpt2.lm_head.output

            logits.sum().backward()

        _test_serialize(tracer)

    hidden_states.value

    assert (hidden_states_grad.value == 0).all().item()

    with gpt2.trace(validate=True) as tracer:
        with tracer.invoke("Hello World", scan=True) as invoker:
            hidden_states = gpt2.transformer.h[-1].output[0].save()
            grad = hidden_states.grad.clone()
            grad[:] = 0
            hidden_states.grad = grad

            logits = gpt2.lm_head.output

            logits.sum().backward()

        _test_serialize(tracer)

    hidden_states.value
    assert (hidden_states_grad.value == 0).all().item()


def test_other_device_tensors(gpt2: nnsight.LanguageModel):

    device = next(gpt2.parameters())

    lin = torch.nn.Linear(768, 768).to(device)
    bias = torch.randn(768).to(device)

    def fun(x):
        return torch.nn.ReLU()(lin(x) - bias)

    with gpt2.trace("fish", validate=True, scan=True) as tracer:
        x = gpt2.transformer.h[0].mlp.output
        y = fun(x)
        z = y.save()

        # TODO
        # _test_serialize(tracer)

    z.value


def test_multi_grad(gpt2: nnsight.LanguageModel):
    with gpt2.trace(validate=True) as tracer:
        with tracer.invoke("Hello World", scan=True) as invoker:
            hidden_states = gpt2.transformer.h[-1].output[0].save()

            hidden_states_grad1 = hidden_states.grad.save()

            logits = gpt2.lm_head.output

            logits.sum().backward(retain_graph=True)

            hidden_states_grad2 = hidden_states.grad.save()

            logits = logits * 2

            logits.sum().backward()

        _test_serialize(tracer)

    assert not torch.all(hidden_states_grad1.eq(hidden_states_grad2))


def test_editing(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    from nnsight.util import WrapperModule

    class ComplexModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.one = WrapperModule()

        def forward(self, x):
            return self.one(x)

    l0 = gpt2.transformer.h[0]
    l0.attachment = ComplexModule()

    with gpt2.edit() as gpt2_edited:
        acts = l0.output[0]
        l0.output[0][:] = l0.attachment(acts, hook=True)

    # Get values pre editing
    with gpt2.trace(MSG_prompt):
        original = l0.output[0].clone().save()
        l0.output[0][:] *= 0.0
        original_output = gpt2.output.logits.save()

    with gpt2_edited.trace(MSG_prompt):
        one = l0.attachment.one.output.clone().save()
        l0.attachment.output *= 0.0
        edited_output = gpt2.output.logits.save()

    # Check that submodule in attached model
    # is equal to original output.
    assert torch.equal(original, one)
    # Check that edits propagate from attached module
    assert torch.equal(original_output, edited_output)


def test_non_inplace_editing(gpt2: nnsight.LanguageModel, MSG_prompt: str):

    with gpt2.edit(inplace=True):
        gpt2.transformer.h[1].output[0][:, 0] = 0

    with gpt2.edit() as gpt2_edited:
        gpt2.transformer.h[1].output[0][:, 1] = 0

    with gpt2.trace(MSG_prompt):
        l1_out = gpt2.transformer.h[1].output[0].save()

    with gpt2_edited.trace(MSG_prompt):
        l1_out_edited = gpt2_edited.transformer.h[1].output[0].save()

    assert torch.all(l1_out[:, 0] == 0) and torch.all(l1_out[:, 1] != 0)
    assert torch.all(l1_out_edited[:, 0] == 0) and torch.all(l1_out_edited[: , 1] == 0)


def test_clear_edits(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.edit(inplace=True):
        gpt2.transformer.h[1].output[0][:] = 0

    with gpt2.trace(MSG_prompt):
        l1_out = gpt2.transformer.h[1].output[0].save()

    gpt2.clear_edits()

    with gpt2.trace(MSG_prompt):
        l1_out_unedited = gpt2.transformer.h[1].output[0].save()

    assert torch.all(l1_out == 0)
    assert torch.all(l1_out_unedited != 0)


def test_batched_editing(gpt2: nnsight.LanguageModel):
    from nnsight.util import WrapperModule

    class ComplexModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.one = WrapperModule()

        def forward(self, x):
            return self.one(x)

    l0 = gpt2.transformer.h[0]
    l0.attachment = ComplexModule()

    batch = ["a", "b"]
    single = "a"

    with gpt2.edit(single) as gpt2_edited:
        acts = l0.output[0]
        l0.output[0][:] = l0.attachment(acts, hook=True)

    with gpt2_edited.trace(batch):
        edited = l0.attachment.output.save()

    # Check that the batch size does not narrow
    assert edited.shape[0] == 2


def test_conditional_interventions(gpt2: nnsight.LanguageModel):
    with gpt2.session() as session:
        with gpt2.trace("Hello World", validate=True, scan=True) as tracer:
            with tracer.cond(
                torch.all(gpt2.transformer.h[5].output[0] < 100000)
            ):
                gpt2.transformer.h[-1].output[0][:] = 0

            output = gpt2.transformer.h[-1].output[0].save()

        _test_serialize(session)

    assert torch.all(output.value == 0)


def test_input_setting(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.session():
        with gpt2.trace(MSG_prompt):
            hs = gpt2.transformer.h[6].inputs
            tokens_out_1 = gpt2.lm_head.output.argmax(dim=-1).save()

        with gpt2.trace(MSG_prompt):
            gpt2.transformer.h[6].input = hs[0][0]
            tokens_out_2 = gpt2.lm_head.output.argmax(dim=-1).save()

    prediction_1 = gpt2.tokenizer.decode(tokens_out_1[0][-1])
    prediction_2 = gpt2.tokenizer.decode(tokens_out_2[0][-1])

    assert prediction_1 == prediction_2
