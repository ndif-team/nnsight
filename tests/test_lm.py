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
def ET_prompt():
    return "The Eiffel Tower is located in the city of"


@pytest.fixture
def MSG_prompt():
    return "Madison Square Garden is located in the city of"


@torch.no_grad()
def test_generation(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.generate(
        max_new_tokens=3,
        
        ) as generator:
        
        
        with generator.invoke(MSG_prompt) as invoker:
            
            
            output = gpt2.generator.output.save()

    output = gpt2.tokenizer.decode(output[0])

    assert output == "Madison Square Garden is located in the city of New York City"


@torch.no_grad()
def test_save(gpt2: nnsight.LanguageModel):
    with gpt2.generate(
        "Hello world"
    ) as tracer:
        hs_input = gpt2.transformer.h[-1].input.save()
        hs = gpt2.transformer.h[-1].output[0].save()
        

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
            pre = gpt2.transformer.h[-1].output[0].clone().save()

            gpt2.transformer.h[-1].output[0][:] = 0

            post = gpt2.transformer.h[-1].output[0].save()

            output = gpt2.generator.output.save()

    output = gpt2.tokenizer.decode(output[0])

    assert not (pre == 0).all().item()
    assert (post == 0).all().item()
    assert output != "Madison Square Garden is located in the city of New"


@torch.no_grad()
def test_set2(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.generate() as generator:
        with generator.invoke(MSG_prompt) as invoker:
            pre = gpt2.transformer.wte.output.clone().save()

            gpt2.transformer.wte.output = gpt2.transformer.wte.output * 0

            post = gpt2.transformer.wte.output.save()

            output = gpt2.generator.output.save()

    output = gpt2.tokenizer.decode(output[0])

    assert not (pre == 0).all().item()
    assert (post == 0).all().item()
    assert output != "Madison Square Garden is located in the city of New"

@torch.no_grad()
def test_set3(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.generate() as generator:
        with generator.invoke(MSG_prompt) as invoker:
            pre = gpt2.transformer.h[-1].output.save()

            gpt2.transformer.h[-1].output = (torch.zeros_like(gpt2.transformer.h[-1].output[0]),) + gpt2.transformer.h[-1].output[1:]

            post = gpt2.transformer.h[-1].output.save()

            output = gpt2.generator.output.save()

        _test_serialize(generator)

    output = gpt2.tokenizer.decode(output[0])

    assert not (pre[0] == 0).all().item()
    assert (post[0] == 0).all().item()
    assert output != "Madison Square Garden is located in the city of New"

@torch.no_grad()
def test_set4(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.trace() as tracer:
        with tracer.invoke(MSG_prompt):
            gpt2.transformer.h[-1].output = (torch.zeros_like(gpt2.transformer.h[-1].output[0]),) + gpt2.transformer.h[-1].output[1:]

        with tracer.invoke(MSG_prompt):
            hs = gpt2.transformer.h[-1].output.save()
            out = gpt2.lm_head.output[0][-1].argmax(dim=-1).save()

    assert isinstance(hs, tuple)
    assert torch.all(hs[0] != 0)
    assert gpt2.tokenizer.decode(out) == " New"


def test_set5(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.trace() as tracer:
        
        with tracer.invoke(MSG_prompt):
            gpt2.transformer.h[0].output[0].requires_grad_(True)
            gpt2.transformer.h[0].output = (torch.zeros_like(gpt2.transformer.h[0].output[0]),) + gpt2.transformer.h[0].output[1:]
        
        with tracer.invoke(MSG_prompt):
            hs = gpt2.transformer.h[0].output[0].save()
            out = gpt2.lm_head.output[0][-1].argmax(dim=-1).save()
    
    assert torch.all(hs != 0)
    assert gpt2.tokenizer.decode(out) == " New"


@torch.no_grad()
def test_adhoc_module(gpt2: nnsight.LanguageModel):
    with gpt2.generate() as generator:
        with generator.invoke(
            "The Eiffel Tower is in the city of"
        ) as invoker:
            hidden_states = gpt2.transformer.h[-1].output[0]
            hidden_states = gpt2.lm_head(gpt2.transformer.ln_f(hidden_states))
            tokens = torch.softmax(hidden_states, dim=2).argmax(dim=2).save()

    output = gpt2.tokenizer.decode(tokens[0])

    assert output == "\n-el Tower is a the middle centre Paris"


@torch.no_grad()
def test_embeddings_set1(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.generate(
        max_new_tokens=3
    ) as generator:
        with generator.invoke(MSG_prompt) as invoker:
            embeddings = gpt2.transformer.wte.output


            output1 = gpt2.generator.output.save()

        with generator.invoke("_ _ _ _ _ _ _ _ _") as invoker:
            gpt2.transformer.wte.wait_for_output()
            gpt2.transformer.wte.output = embeddings

            output2 = gpt2.generator.output.save()


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
            embeddings = gpt2.transformer.wte.output.save()

            output = gpt2.generator.output.save()

    output1 = gpt2.tokenizer.decode(output[0])

    with gpt2.generate(
        max_new_tokens=3
    ) as generator:
        with generator.invoke("_ _ _ _ _ _ _ _ _") as invoker:
            gpt2.transformer.wte.output = embeddings

            output = gpt2.generator.output.save()

        _test_serialize(generator)

    output2 = gpt2.tokenizer.decode(output[0])

    assert output1 == "Madison Square Garden is located in the city of New York City"
    assert output2 == "_ _ _ _ _ _ _ _ _ New York City"


def test_retain_grad(gpt2: nnsight.LanguageModel):
    with gpt2.trace() as tracer:
        with tracer.invoke("Hello World") as invoker:
            hidden_states = gpt2.transformer.h[-1].output[0].save()
            hidden_states.retain_grad()

            logits = gpt2.lm_head.output

            logits.sum().backward()

        _test_serialize(tracer)

    assert hidden_states.grad is not None


def test_grad(gpt2: nnsight.LanguageModel):
    with gpt2.trace() as tracer:
        with tracer.invoke("Hello World") as invoker:
            hidden_states = gpt2.transformer.h[-1].output[0].save()
            
            logits = gpt2.lm_head.output.save()
            
            

            with logits.sum().backward():
                hidden_states_grad = hidden_states.grad.save()
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
                hidden_states_2grad = hidden_states.grad.save()
                
                

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
        z = y.save()

        # TODO
        # _test_serialize(tracer)

    z


def test_multi_grad(gpt2: nnsight.LanguageModel):
    with gpt2.trace() as tracer:
        with tracer.invoke("Hello World") as invoker:
            hidden_states = gpt2.transformer.h[-1].output[0]
            
            logits = gpt2.lm_head.output

            
            with logits.sum().backward(retain_graph=True):
                hidden_states_grad1 = hidden_states.grad.save()


            logits = logits * 2
            with logits.sum().backward():
                
                hidden_states_grad2 = hidden_states.grad.save()
                
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
        edited_output = gpt2_edited.output.logits.save()

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
        l1_out = gpt2.transformer.h[1].output[0].save()

    with gpt2_edited.trace(MSG_prompt):
        l1_out_edited = gpt2_edited.transformer.h[1].output[0].save()

    assert torch.all(l1_out[:, 0] == 0) and torch.all(l1_out[:, 1] != 0)
    assert torch.all(l1_out_edited[:, 0] == 0) and torch.all(l1_out_edited[:, 1] == 0)


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

    with gpt2.edit() as gpt2_edited:
        acts = l0.output[0]
        l0.output[0][:] = l0.attachment(acts, hook=True)

    with gpt2_edited.trace(batch):
        edited = l0.attachment.output.save()

    # Check that the batch size does not narrow
    assert edited.shape[0] == 2


def test_conditional_interventions(gpt2: nnsight.LanguageModel):
    with gpt2.session() as session:
        with gpt2.trace("Hello World") as tracer:
            if torch.all(gpt2.transformer.h[5].output[0] < 100000):
                gpt2.transformer.h[-1].output[0][:] = 0

            output = gpt2.transformer.h[-1].output[0].save()

    assert torch.all(output == 0)


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


@pytest.mark.order
def test_out_of_order(gpt2: nnsight.LanguageModel):
    with pytest.raises(ValueError):
        with gpt2.trace("_"):
            out = gpt2.transformer.h[2].output.save()
            out_2 = gpt2.transformer.h[1].inputs.save()


@pytest.mark.order
@pytest.mark.skips
def test_out_of_order_skip(gpt2: nnsight.LanguageModel):

    with pytest.raises(ValueError):
        with gpt2.trace("_"):

            gpt2.transformer.h[1].skip(gpt2.transformer.h[0].output)
            gpt2.transformer.h[1].input[:] = 0


# TODO - FIX THIS?
# def test_cleanup(gpt2: nnsight.LanguageModel):
#     with gpt2.trace("_", max_new_tokens=2) as tracer:
#         arr = list()

#         with tracer.all():
#             pass

#     with pytest.raises(UnboundLocalError):
#         arr

######################### SOURCE #################################

@pytest.mark.source
def test_source_output(gpt2: nnsight.LanguageModel):
    with gpt2.trace("_"):
        out = gpt2.transformer.h[0].attn.source.split_1.output.save()

    assert isinstance(out, tuple)


@pytest.mark.source
def test_source_inputs(gpt2: nnsight.LanguageModel):
    with gpt2.trace("_"):
        inp = gpt2.transformer.h[0].attn.source.attention_interface_0.inputs.save()

    assert isinstance(inp, tuple)


@pytest.mark.source
def test_source_safe_intervention(gpt2: nnsight.LanguageModel, MSG_prompt: str):

    input = gpt2.tokenizer(MSG_prompt, return_tensors="pt").to(gpt2.device)

    logits_0 = gpt2(**input)['logits']

    gpt2.transformer.h[0].attn.source
    with gpt2.trace("_"):
        gpt2.transformer.h[0].attn.c_attn.output = torch.zeros_like(gpt2.transformer.h[0].attn.c_attn.output)
        out = gpt2.transformer.h[0].attn.source.split_1.output.save()

    logits_1 = gpt2(**input)['logits']

    assert isinstance(out, tuple)
    assert torch.all(out[0] == 0) 
    assert torch.all(logits_0 == logits_1)


@pytest.mark.source
def test_multiple_source(gpt2: nnsight.LanguageModel):
    with gpt2.trace("_"):
        out_split_0 = gpt2.transformer.h[0].attn.source.split_1.output.save()
        out_attention_interface_0 = gpt2.transformer.h[0].attn.source.attention_interface_0.output.save()

        out_split_1 = gpt2.transformer.h[1].attn.source.split_1.output.save()
        out_attention_interface_1 = gpt2.transformer.h[1].attn.source.attention_interface_0.output.save()

    assert isinstance(out_split_0, tuple)
    assert isinstance(out_attention_interface_0, tuple)
    assert isinstance(out_split_1, tuple)
    assert isinstance(out_attention_interface_1, tuple)


@pytest.mark.source
def test_source_patching(gpt2: nnsight.LanguageModel):
    with gpt2.trace("_"):
        out = gpt2.transformer.h[0].attn.source.split_1.output
        out = (torch.zeros_like(out[0]),) + out[1:]

        gpt2.transformer.h[1].attn.source.split_1.output = out

        out_2 = gpt2.transformer.h[1].attn.source.split_1.output.save()

    assert isinstance(out_2, tuple)
    assert torch.all(out_2[0] == 0)


@pytest.mark.source
def test_recursive_source(gpt2: nnsight.LanguageModel):
    
    with gpt2.trace("_"):
        out = gpt2.transformer.h[0].attn.source.attention_interface_0.source.torch_nn_functional_scaled_dot_product_attention_0.output.save()

    assert isinstance(out, torch.Tensor)


@pytest.mark.source
def test_source_imported_function(gpt2: nnsight.LanguageModel):
    with gpt2.trace("_"):
        gpt2.transformer.h[0].attn.source.split_1.source
        out = gpt2.transformer.h[0].attn.source.split_1.output.save()

    assert isinstance(out, tuple)


@pytest.mark.source
def test_source_operation_not_found(gpt2: nnsight.LanguageModel):
    with pytest.raises(AttributeError):
        with gpt2.trace("_"):
            out = gpt2.transformer.h[0].attn.source.my_func.output.save()


######################### SKIP #################################

@pytest.mark.skips
def test_skip(gpt2: nnsight.LanguageModel):
    with gpt2.trace("_"):

        inp = gpt2.transformer.h[0].output.save()
        gpt2.transformer.h[1].skip(inp)
        out = gpt2.transformer.h[1].output.save()

    assert torch.equal(out[0], inp[0])


@pytest.mark.skips
def test_skip_2(gpt2: nnsight.LanguageModel):
    with gpt2.trace("_"):

        inp = gpt2.transformer.h[0].output.save()
        gpt2.transformer.h[1].inputs = ((torch.zeros_like(gpt2.transformer.h[1].input),)) + gpt2.transformer.h[1].inputs[1:]
        gpt2.transformer.h[1].skip(inp)

        out = gpt2.transformer.h[1].output.save()

    assert not torch.all(out[0] == 0)
    assert torch.equal(out[0], inp[0])


@pytest.mark.skips
def test_multiple_skip(gpt2: nnsight.LanguageModel):
    with gpt2.trace("_"):

        inp = gpt2.transformer.h[0].output

        gpt2.transformer.h[1].skip(inp)
        inp_2 = gpt2.transformer.h[1].output
        gpt2.transformer.h[2].skip(inp_2)
        inp_3 = gpt2.transformer.h[2].output.save()
        gpt2.transformer.h[3].skip(inp_3)
        
        out = gpt2.transformer.h[3].output.save()
    
    assert torch.equal(out[0], inp_3[0])


@pytest.mark.skips
def test_skip_inner_module(gpt2: nnsight.LanguageModel):
    with gpt2.trace("Hello World"):
        inp = gpt2.transformer.h[0].output
        gpt2.transformer.h[1].skip(inp)
        hs = gpt2.transformer.h[1].attn.output.save()

    with pytest.raises(UnboundLocalError):
        hs


@pytest.mark.skips
def test_skip_module_for_batch(gpt2: nnsight.LanguageModel, ET_prompt: str, MSG_prompt: str):
    with gpt2.trace() as tracer:
        with tracer.invoke(ET_prompt):
            inp = gpt2.transformer.h[0].output.save()

            gpt2.transformer.h[1].skip(inp)

            out = gpt2.transformer.h[1].output.save()

        with tracer.invoke(MSG_prompt):

            inp_2 = gpt2.transformer.h[0].output.save()

            gpt2.transformer.h[1].skip(inp_2)

            out_2 = gpt2.transformer.h[1].output.save()

    assert not torch.equal(out[0], out_2[0])
    assert torch.equal(out[0], inp[0])
    assert torch.equal(out_2[0], inp_2[0])


@pytest.mark.skips
def test_skip_module_for_batch_error(gpt2: nnsight.LanguageModel, ET_prompt: str, MSG_prompt: str):

    with pytest.raises(ValueError):
        with gpt2.trace() as tracer:
            with tracer.invoke(ET_prompt):
                inp = gpt2.transformer.h[0].output.save()

                gpt2.transformer.h[1].skip(inp)

            with tracer.invoke(MSG_prompt):
                inp_2 = gpt2.transformer.h[0].output.save()
                

###################### ITER #####################

@pytest.mark.iter
def test_iter(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.generate(MSG_prompt, max_new_tokens=3):
        logits_all = list().save()

        with gpt2.all():
            logits_all.append(gpt2.lm_head.output[0][-1].argmax(dim=-1))

    with gpt2.generate(MSG_prompt, max_new_tokens=3):
        logits_iter = list().save()

        with gpt2.iter[:]:
            logits_iter.append(gpt2.lm_head.output[0][-1].argmax(dim=-1))

    assert len(logits_all) == 3
    assert len(logits_iter) == 3
    
    assert gpt2.tokenizer.batch_decode(logits_all) == [" New", " York", " City"]
    assert gpt2.tokenizer.batch_decode(logits_iter) == [" New", " York", " City"]


@pytest.mark.iter
def test_iter_slice(gpt2: nnsight.LanguageModel, MSG_prompt: str):

    with gpt2.generate(MSG_prompt, max_new_tokens=5) as tracer:
        logits = list().save()

        with tracer.iter[1:3]:
            logits.append(gpt2.lm_head.output[0][-1].argmax(dim=-1))

    assert len(logits) == 2
    assert gpt2.tokenizer.batch_decode(logits) == [" York", " City"]


@pytest.mark.iter
def test_iter_module(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.trace(MSG_prompt, max_new_tokens=3) as tracer:
        logits = list()

        with tracer.iter[:]:
            logits.append(gpt2.lm_head.output[0][-1].argmax(dim=-1))


@pytest.mark.iter
def test_iter_idx(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.generate(MSG_prompt, max_new_tokens=3) as tracer:
        hs = list().save()

        with tracer.iter[0]:
            hs.append(gpt2.transformer.h[0].output[0])

        with tracer.iter[1]:
            hs.append(gpt2.transformer.h[1].output[0])

        with tracer.iter[2]:
            hs.append(gpt2.transformer.h[2].output[0])


    with gpt2.generate(MSG_prompt, max_new_tokens=3) as tracer:
        hs_2 = list().save()

        with tracer.iter[:3] as idx:
            hs_2.append(gpt2.transformer.h[idx].output[0])

    assert all([torch.equal(h, h_2) for h, h_2 in zip(hs, hs_2)])


@pytest.mark.iter
def test_batched_iter(gpt2: nnsight.LanguageModel, MSG_prompt: str):
    with gpt2.generate(max_new_tokens=5) as tracer:

        with tracer.invoke(MSG_prompt):
            logits_1 = list().save()

            with tracer.iter[1:3]:
                logits_1.append(gpt2.lm_head.output[0][-1].argmax(dim=-1))

        with tracer.invoke(MSG_prompt):
            logits_2 = list().save()

            with tracer.iter[:3]:
                logits_2.append(gpt2.lm_head.output[0][-1].argmax(dim=-1))

    assert len(logits_1) == 2
    assert len(logits_2) == 3

    assert gpt2.tokenizer.batch_decode(logits_1) == [" York", " City"]
    assert gpt2.tokenizer.batch_decode(logits_2) == [" New", " York", " City"]


# TODO - FIX THIS
@pytest.mark.iter
def test_one_iter(gpt2: nnsight.LanguageModel):
    with gpt2.generate("_", max_new_tokens=1) as tracer:
        arr_gen = list().save()

        with tracer.all():
            arr_gen.append(1)

    assert len(arr_gen) == 1


@pytest.mark.iter
@pytest.mark.skips
def test_iter_and_skip(gpt2: nnsight.LanguageModel):
    with gpt2.generate("_", max_new_tokens=3) as tracer:
        arr_gen = list().save()

        with tracer.iter[:] as it:
            if it != 1:
                arr_gen.append(gpt2.transformer.h[1].output[0])
            else:
                gpt2.transformer.h[1].skip((torch.zeros_like(gpt2.transformer.h[0].output[0]),) + gpt2.transformer.h[0].output[1:])

                arr_gen.append(gpt2.transformer.h[1].output[0])

    assert not torch.all(arr_gen[0] == 0)
    assert torch.all(arr_gen[1] == 0)
    assert not torch.all(arr_gen[2] == 0)


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

        output = model.generator.output.save()

    output
