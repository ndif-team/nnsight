import pytest
import torch
import nnsight


def _test_serialize(tracer):
    pass

@pytest.fixture(scope="module")
def gpt2(device: str):
    return nnsight.LanguageModel(
        "openai-community/gpt2", device_map=device, dispatch=True
    )


@pytest.fixture
def MSG_prompt():
    return "Madison Square Garden is located in the city of"



@torch.no_grad()
def test_generation(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.generate(
        max_new_tokens=3,
        
        ) as generator:
        
        
        with generator.invoke(MSG_prompt) as invoker:
            
            
            output = gpt2.generator.output

    output = gpt2.tokenizer.decode(output[0])

    assert output == "Madison Square Garden is located in the city of New York City"


@torch.no_grad()
def test_save(gpt2: nnsight.LanguageModel):
    with gpt2.generate(
        "Hello world"
    ) as tracer:
        hs_input = gpt2.transformer.h[-1].input
        hs = gpt2.transformer.h[-1].output[0]
        

    assert hs is not None
    assert isinstance(hs, torch.Tensor)
    assert hs.ndim == 3

    assert hs_input is not None
    assert isinstance(hs_input, torch.Tensor)
    assert hs_input.ndim == 3


@torch.no_grad()
def test_set1(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.generate() as tracer:
        with tracer.invoke(MSG_prompt) as invoker:
            pre = gpt2.transformer.h[-1].output[0].clone()

            gpt2.transformer.h[-1].output[0][:] = 0

            post = gpt2.transformer.h[-1].output[0]

            output = gpt2.generator.output

    output = gpt2.tokenizer.decode(output[0])

    assert not (pre == 0).all().item()
    assert (post == 0).all().item()
    assert output != "Madison Square Garden is located in the city of New"


@torch.no_grad()
def test_set2(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.generate() as generator:
        with generator.invoke(MSG_prompt) as invoker:
            pre = gpt2.transformer.wte.output.clone()

            gpt2.transformer.wte.output = gpt2.transformer.wte.output * 0

            post = gpt2.transformer.wte.output

            output = gpt2.generator.output

    output = gpt2.tokenizer.decode(output[0])

    assert not (pre == 0).all().item()
    assert (post == 0).all().item()
    assert output != "Madison Square Garden is located in the city of New"


@torch.no_grad()
def test_adhoc_module(gpt2: nnsight.LanguageModel):
    with gpt2.generate() as generator:
        with generator.invoke(
            "The Eiffel Tower is in the city of"
        ) as invoker:
            hidden_states = gpt2.transformer.h[-1].output[0]
            hidden_states = gpt2.lm_head(gpt2.transformer.ln_f(hidden_states))
            tokens = torch.softmax(hidden_states, dim=2).argmax(dim=2)

    output = gpt2.tokenizer.decode(tokens[0])

    assert output == "\n-el Tower is a the middle centre Paris"


@torch.no_grad()
def test_embeddings_set1(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.generate(
        max_new_tokens=3
    ) as generator:
        with generator.invoke(MSG_prompt) as invoker:
            embeddings = gpt2.transformer.wte.output


            output1 = gpt2.generator.output

        with generator.invoke("_ _ _ _ _ _ _ _ _") as invoker:
            gpt2.transformer.wte.output
            gpt2.transformer.wte.output = embeddings

            output2 = gpt2.generator.output


    output1 = gpt2.tokenizer.decode(output1[0])
    output2 = gpt2.tokenizer.decode(output2[0])

    assert output1 == "Madison Square Garden is located in the city of New York City"

    assert output2 == "_ _ _ _ _ _ _ _ _ New York City"


@torch.no_grad()
def test_embeddings_set2(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.generate(
        max_new_tokens=3
    ) as generator:
        with generator.invoke(MSG_prompt) as invoker:
            embeddings = gpt2.transformer.wte.output

            output = gpt2.generator.output

    output1 = gpt2.tokenizer.decode(output[0])

    with gpt2.generate(
        max_new_tokens=3
    ) as generator:
        with generator.invoke("_ _ _ _ _ _ _ _ _") as invoker:
            gpt2.transformer.wte.output = embeddings

            output = gpt2.generator.output

        _test_serialize(generator)

    output2 = gpt2.tokenizer.decode(output[0])

    assert output1 == "Madison Square Garden is located in the city of New York City"
    assert output2 == "_ _ _ _ _ _ _ _ _ New York City"


def test_retain_grad(gpt2: nnsight.LanguageModel):
    with gpt2.trace() as tracer:
        with tracer.invoke("Hello World") as invoker:
            hidden_states = gpt2.transformer.h[-1].output[0]
            hidden_states.retain_grad()

            logits = gpt2.lm_head.output

            logits.sum().backward()

        _test_serialize(tracer)

    assert hidden_states.grad is not None


def test_grad(gpt2: nnsight.LanguageModel):
    with gpt2.trace() as tracer:
        with tracer.invoke("Hello World") as invoker:
            hidden_states = gpt2.transformer.h[-1].output[0]
            
            logits = gpt2.lm_head.output
            
            

            with logits.sum().backward():
                hidden_states_grad = hidden_states.grad
                hidden_states_grad[:] = 0


    assert (hidden_states_grad == 0).all().item()

    with gpt2.trace() as tracer:
        with tracer.invoke("Hello World") as invoker:
            hidden_states = gpt2.transformer.h[-1].output[0]
            
            logits = gpt2.lm_head.output
            

            

            with logits.sum().backward():
                grad = hidden_states.grad.clone()
                grad[:] = 0
                hidden_states.grad = grad
                hidden_states_2grad = hidden_states.grad
                
                

        _test_serialize(tracer)

    assert (hidden_states_2grad == 0).all().item()


def test_other_device_tensors(gpt2: nnsight.LanguageModel):

    device = next(gpt2.parameters())

    lin = torch.nn.Linear(768, 768).to(device)
    bias = torch.randn(768).to(device)

    def fun(x):
        return torch.nn.ReLU()(lin(x) - bias)

    with gpt2.trace(
        "fish"
    ) as tracer:
        x = gpt2.transformer.h[0].mlp.output
        y = fun(x)
        z = y

        # TODO
        # _test_serialize(tracer)

    z


def test_multi_grad(gpt2: nnsight.LanguageModel):
    with gpt2.trace() as tracer:
        with tracer.invoke("Hello World") as invoker:
            hidden_states = gpt2.transformer.h[-1].output[0]
            
            logits = gpt2.lm_head.output

            
            with logits.sum().backward(retain_graph=True):
                hidden_states_grad1 = hidden_states.grad


            logits = logits * 2
            with logits.sum().backward():
                
                hidden_states_grad2 = hidden_states.grad
                
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
        original = l0.output[0].clone()
        l0.output[0][:] *= 0.0
        original_output = gpt2.output.logits

    with gpt2_edited.trace(MSG_prompt):
        one = l0.attachment.one.output.clone()
        l0.attachment.output *= 0.0
        edited_output = gpt2_edited.output.logits

    # Check that submodule in attached model
    # is equal to original output.
    assert torch.equal(original, one)
    # Check that edits propagate from attached module
    assert torch.equal(original_output, edited_output)


def test_non_inplace_editing(gpt2: nnsight.LanguageModel, MSG_prompt: str):

    with gpt2.edit(inplace=True):
        gpt2.transformer.h[1].output[0][:, 0] = 0

    with gpt2.edit() as gpt2_edited:
        gpt2_edited.transformer.h[1].output[0][:, 1] = 0

    with gpt2.trace(MSG_prompt):
        l1_out = gpt2.transformer.h[1].output[0]

    with gpt2_edited.trace(MSG_prompt):
        l1_out_edited = gpt2_edited.transformer.h[1].output[0]

    assert torch.all(l1_out[:, 0] == 0) and torch.all(l1_out[:, 1] != 0)
    assert torch.all(l1_out_edited[:, 0] == 0) and torch.all(l1_out_edited[:, 1] == 0)


def test_clear_edits(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.edit(inplace=True):
        gpt2.transformer.h[1].output[0][:] = 0

    with gpt2.trace(MSG_prompt):
        l1_out = gpt2.transformer.h[1].output[0]

    gpt2.clear_edits()

    with gpt2.trace(MSG_prompt):
        l1_out_unedited = gpt2.transformer.h[1].output[0]

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

    with gpt2.edit() as gpt2_edited:
        acts = l0.output[0]
        l0.output[0][:] = l0.attachment(acts, hook=True)

    with gpt2_edited.trace(batch):
        edited = l0.attachment.output

    # Check that the batch size does not narrow
    assert edited.shape[0] == 2


def test_conditional_interventions(gpt2: nnsight.LanguageModel):
    with gpt2.session() as session:
        with gpt2.trace("Hello World") as tracer:
            if torch.all(gpt2.transformer.h[5].output[0] < 100000):
                gpt2.transformer.h[-1].output[0][:] = 0

            output = gpt2.transformer.h[-1].output[0]

    assert torch.all(output == 0)


def test_input_setting(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.session():
        with gpt2.trace(MSG_prompt):
            hs = gpt2.transformer.h[6].inputs
            tokens_out_1 = gpt2.lm_head.output.argmax(dim=-1)

        with gpt2.trace(MSG_prompt):
            gpt2.transformer.h[6].input = hs[0][0]
            tokens_out_2 = gpt2.lm_head.output.argmax(dim=-1)

    prediction_1 = gpt2.tokenizer.decode(tokens_out_1[0][-1])
    prediction_2 = gpt2.tokenizer.decode(tokens_out_2[0][-1])

    assert prediction_1 == prediction_2


# def test_parameter_protocol(gpt2: nnsight.LanguageModel, MSG_prompt: str):
#     og_weights = gpt2.transformer.h[0].mlp.c_proj.weight
#     og_bias = gpt2.transformer.h[0].mlp.c_proj.bias

#     with gpt2.trace(MSG_prompt):
#         cl_weights = gpt2.transformer.h[0].mlp.c_proj.weight
#         cl_bias = gpt2.transformer.h[0].mlp.c_proj.bias

#     assert not (cl_weights is og_weights)
#     assert not (cl_bias is og_bias)

#     assert torch.equal(og_weights, cl_weights)
#     assert torch.equal(og_bias, cl_bias)


# def test_parameter_protocol_meta(MSG_prompt: str):
#     model = nnsight.LanguageModel("openai-community/gpt2")

#     with model.trace(MSG_prompt):
        
#         weights = model.transformer.h[0].mlp.c_proj.weight
#         bias = model.transformer.h[0].mlp.c_proj.bias

#     assert weights.device.type != "meta"
#     assert bias.device.type != "bias"


# def test_parameter_protocol_result(gpt2: nnsight.LanguageModel, MSG_prompt: str):
#     with gpt2.trace(MSG_prompt):
#         weights = gpt2.transformer.h[0].mlp.c_proj.weight
#         bias = gpt2.transformer.h[0].mlp.c_proj.bias

#         c_proj_in = gpt2.transformer.h[0].mlp.c_proj.input

#         c_proj_out_2 = torch.addmm(
#             bias, c_proj_in.view(-1, c_proj_in.size(-1)), weights
#         )

#         c_proj_out = gpt2.transformer.h[0].mlp.c_proj.output[0]

#     assert torch.equal(c_proj_out_2, c_proj_out)


# def test_parameter_protocol_safety(gpt2: nnsight.LanguageModel, MSG_prompt: str):
#     with gpt2.trace(MSG_prompt):
#         hs_1 = gpt2.transformer.h[0].mlp.c_proj.output[0]

#     with gpt2.trace(MSG_prompt):
#         gpt2.transformer.h[0].mlp.c_proj.weight[:] = 0

#     with gpt2.trace(MSG_prompt):
#         hs_2 = gpt2.transformer.h[0].mlp.c_proj.output[0]

#     assert torch.equal(hs_1, hs_2)


def test_undispatched_extra_module(device: str, MSG_prompt: str):

    model = nnsight.LanguageModel(
        "openai-community/gpt2", device_map=device, dispatch=False
    )

    with model.generate(MSG_prompt, max_new_tokens=1):

        output = model.generator.output

    output
