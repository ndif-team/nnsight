"""
Tests for LanguageModel functionality with GPT-2.

These tests cover language model specific features:
- Generation and tracing
- Activation access and modification
- Embedding manipulation
- Gradient computation
- Model editing
- Scanning and validation
- Source tracing
- Module skipping
- Iteration and caching
- Module renaming
"""

import pytest
import torch
import nnsight


def _test_serialize(tracer):
    """Placeholder for serialization testing."""
    pass


# =============================================================================
# Basic Generation and Tracing
# =============================================================================


class TestGeneration:
    """Tests for text generation functionality."""

    @torch.no_grad()
    def test_basic_generation(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test basic multi-token generation."""
        with gpt2.generate(max_new_tokens=3) as generator:
            with generator.invoke(MSG_prompt):
                output = gpt2.generator.output.save()

        output = gpt2.tokenizer.decode(output[0])
        assert output == "Madison Square Garden is located in the city of New York City"

    @torch.no_grad()
    def test_save_hidden_states(self, gpt2: nnsight.LanguageModel):
        """Test saving hidden states and inputs."""
        with gpt2.generate("Hello world"):
            hs_input = gpt2.transformer.h[-1].input.save()
            hs = gpt2.transformer.h[-1].output[0].save()

        assert hs is not None
        assert isinstance(hs, torch.Tensor)
        assert hs.ndim == 3
        assert hs_input is not None
        assert isinstance(hs_input, torch.Tensor)
        assert hs_input.ndim == 3


# =============================================================================
# Activation Modification
# =============================================================================


class TestActivationModification:
    """Tests for modifying activations during forward pass."""

    @torch.no_grad()
    def test_inplace_modification(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test in-place activation modification with [:]."""
        with gpt2.generate() as tracer:
            with tracer.invoke(MSG_prompt):
                pre = gpt2.transformer.h[-1].output[0].clone().save()
                gpt2.transformer.h[-1].output[0][:] = 0
                post = gpt2.transformer.h[-1].output[0].save()
                output = gpt2.generator.output.save()

        output = gpt2.tokenizer.decode(output[0])
        assert not (pre == 0).all().item()
        assert (post == 0).all().item()
        assert output != "Madison Square Garden is located in the city of New"

    @torch.no_grad()
    def test_multiplication_modification(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test activation modification via multiplication."""
        with gpt2.generate() as generator:
            with generator.invoke(MSG_prompt):
                pre = gpt2.transformer.wte.output.clone().save()
                gpt2.transformer.wte.output = gpt2.transformer.wte.output * 0
                post = gpt2.transformer.wte.output.save()
                output = gpt2.generator.output.save()

        output = gpt2.tokenizer.decode(output[0])
        assert not (pre == 0).all().item()
        assert (post == 0).all().item()
        assert output != "Madison Square Garden is located in the city of New"

    @torch.no_grad()
    def test_tuple_replacement(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test replacing tuple outputs."""
        with gpt2.generate() as generator:
            with generator.invoke(MSG_prompt):
                pre = gpt2.transformer.h[-1].output.save()
                gpt2.transformer.h[-1].output = (
                    torch.zeros_like(gpt2.transformer.h[-1].output[0]),
                ) + gpt2.transformer.h[-1].output[1:]
                post = gpt2.transformer.h[-1].output.save()
                output = gpt2.generator.output.save()
            _test_serialize(generator)

        output = gpt2.tokenizer.decode(output[0])
        assert not (pre[0] == 0).all().item()
        assert (post[0] == 0).all().item()
        assert output != "Madison Square Garden is located in the city of New"

    @torch.no_grad()
    def test_cross_invoke_isolation(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test that modifications in one invoke don't affect another."""
        with gpt2.trace() as tracer:
            with tracer.invoke(MSG_prompt):
                gpt2.transformer.h[-1].output = (
                    torch.zeros_like(gpt2.transformer.h[-1].output[0]),
                ) + gpt2.transformer.h[-1].output[1:]

            with tracer.invoke(MSG_prompt):
                hs = gpt2.transformer.h[-1].output.save()
                out = gpt2.lm_head.output[0][-1].argmax(dim=-1).save()

        assert isinstance(hs, tuple)
        assert torch.all(hs[0] != 0)
        assert gpt2.tokenizer.decode(out) == " New"

    def test_modification_with_grad(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test modification with gradients enabled."""
        with gpt2.trace() as tracer:
            with tracer.invoke(MSG_prompt):
                gpt2.transformer.h[0].output[0].requires_grad_(True)
                gpt2.transformer.h[0].output = (
                    torch.zeros_like(gpt2.transformer.h[0].output[0]),
                ) + gpt2.transformer.h[0].output[1:]

            with tracer.invoke(MSG_prompt):
                hs = gpt2.transformer.h[0].output[0].save()
                out = gpt2.lm_head.output[0][-1].argmax(dim=-1).save()

        assert torch.all(hs != 0)
        assert gpt2.tokenizer.decode(out) == " New"


# =============================================================================
# Ad-hoc Module Application
# =============================================================================


class TestAdhocModules:
    """Tests for applying modules outside their normal execution order."""

    @torch.no_grad()
    def test_logit_lens(self, gpt2: nnsight.LanguageModel):
        """Test applying lm_head to intermediate hidden states (logit lens)."""
        with gpt2.generate() as generator:
            with generator.invoke("The Eiffel Tower is in the city of"):
                hidden_states = gpt2.transformer.h[-1].output[0]
                hidden_states = gpt2.lm_head(gpt2.transformer.ln_f(hidden_states))
                tokens = torch.softmax(hidden_states, dim=2).argmax(dim=2).save()

        output = gpt2.tokenizer.decode(tokens[0])
        assert output == "\n-el Tower is a the middle centre Paris"


# =============================================================================
# Embedding Manipulation
# =============================================================================


class TestEmbeddings:
    """Tests for embedding manipulation and transfer."""

    @torch.no_grad()
    def test_embedding_transfer_with_barrier(
        self, gpt2: nnsight.LanguageModel, MSG_prompt: str
    ):
        """Test transferring embeddings between invokes using barrier."""
        with gpt2.generate(max_new_tokens=3) as tracer:
            barrier = tracer.barrier(2)

            with tracer.invoke(MSG_prompt):
                embeddings = gpt2.transformer.wte.output
                barrier()
                output1 = gpt2.generator.output.save()

            with tracer.invoke("_ _ _ _ _ _ _ _ _"):
                barrier()
                gpt2.transformer.wte.output = embeddings
                output2 = gpt2.generator.output.save()

        output1 = gpt2.tokenizer.decode(output1[0])
        output2 = gpt2.tokenizer.decode(output2[0])

        assert output1 == "Madison Square Garden is located in the city of New York City"
        assert output2 == "_ _ _ _ _ _ _ _ _ New York City"

    @torch.no_grad()
    def test_embedding_transfer_separate_traces(
        self, gpt2: nnsight.LanguageModel, MSG_prompt: str
    ):
        """Test transferring embeddings between separate traces."""
        with gpt2.generate(max_new_tokens=3) as generator:
            with generator.invoke(MSG_prompt):
                embeddings = gpt2.transformer.wte.output.save()
                output = gpt2.generator.output.save()

        output1 = gpt2.tokenizer.decode(output[0])

        with gpt2.generate(max_new_tokens=3) as generator:
            with generator.invoke("_ _ _ _ _ _ _ _ _"):
                gpt2.transformer.wte.output = embeddings
                output = gpt2.generator.output.save()
            _test_serialize(generator)

        output2 = gpt2.tokenizer.decode(output[0])

        assert output1 == "Madison Square Garden is located in the city of New York City"
        assert output2 == "_ _ _ _ _ _ _ _ _ New York City"


# =============================================================================
# Gradients
# =============================================================================


class TestGradients:
    """Tests for gradient computation and modification."""

    def test_retain_grad(self, gpt2: nnsight.LanguageModel):
        """Test retaining gradients on intermediate tensors."""
        with gpt2.trace() as tracer:
            with tracer.invoke("Hello World"):
                hidden_states = gpt2.transformer.h[-1].output[0].save()
                hidden_states.retain_grad()
                logits = gpt2.lm_head.output
                logits.sum().backward()
            _test_serialize(tracer)

        assert hidden_states.grad is not None

    def test_grad_access_and_modify(self, gpt2: nnsight.LanguageModel):
        """Test accessing and modifying gradients in backward context."""
        with gpt2.trace() as tracer:
            with tracer.invoke("Hello World"):
                hidden_states = gpt2.transformer.h[-1].output[0].save()
                logits = gpt2.lm_head.output.save()

                with logits.sum().backward():
                    hidden_states_grad = hidden_states.grad.save()
                    hidden_states_grad[:] = 0

        assert (hidden_states_grad == 0).all().item()

        # Test gradient replacement
        with gpt2.trace() as tracer:
            with tracer.invoke("Hello World"):
                hidden_states = gpt2.transformer.h[-1].output[0]
                logits = gpt2.lm_head.output

                with logits.sum().backward():
                    grad = hidden_states.grad.clone()
                    grad[:] = 0
                    hidden_states.grad = grad
                    hidden_states_2grad = hidden_states.grad.save()
            _test_serialize(tracer)

        assert (hidden_states_2grad == 0).all().item()

    def test_multi_backward(self, gpt2: nnsight.LanguageModel):
        """Test multiple backward passes with retain_graph."""
        with gpt2.trace() as tracer:
            with tracer.invoke("Hello World"):
                hidden_states = gpt2.transformer.h[-1].output[0]
                logits = gpt2.lm_head.output

                with logits.sum().backward(retain_graph=True):
                    hidden_states_grad1 = hidden_states.grad.save()

                logits = logits * 2
                with logits.sum().backward():
                    hidden_states_grad2 = hidden_states.grad.save()
            _test_serialize(tracer)

        assert not torch.all(hidden_states_grad1.eq(hidden_states_grad2))

    def test_external_tensors(self, gpt2: nnsight.LanguageModel):
        """Test using external tensors in trace."""
        device = next(gpt2.parameters())

        lin = torch.nn.Linear(768, 768).to(device)
        bias = torch.randn(768).to(device)

        def fun(x):
            return torch.nn.ReLU()(lin(x) - bias)

        with gpt2.trace("fish"):
            x = gpt2.transformer.h[0].mlp.output
            y = fun(x)
            z = y.save()

        z  # Just verify it's accessible


# =============================================================================
# Model Editing
# =============================================================================


class TestEditing:
    """Tests for persistent model editing."""

    def test_edit_with_attachment(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test editing with attached modules."""
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

        with gpt2.trace(MSG_prompt):
            original = l0.output[0].clone().save()
            l0.output[0][:] *= 0.0
            original_output = gpt2.output.logits.save()

        with gpt2_edited.trace(MSG_prompt):
            one = l0.attachment.one.output.clone().save()
            l0.attachment.output *= 0.0
            edited_output = gpt2_edited.output.logits.save()

        assert torch.equal(original, one)
        assert torch.equal(original_output, edited_output)

    def test_non_inplace_editing(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test non-inplace editing creates separate model."""
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

    def test_clear_edits(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test clearing inplace edits."""
        with gpt2.edit(inplace=True):
            gpt2.transformer.h[1].output[0][:] = 0

        with gpt2.trace(MSG_prompt):
            l1_out = gpt2.transformer.h[1].output[0].save()

        gpt2.clear_edits()

        with gpt2.trace(MSG_prompt):
            l1_out_unedited = gpt2.transformer.h[1].output[0].save()

        assert torch.all(l1_out == 0)
        assert torch.all(l1_out_unedited != 0)

    def test_batched_editing(self, gpt2: nnsight.LanguageModel):
        """Test editing with batched inputs."""
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

        with gpt2.edit() as gpt2_edited:
            acts = l0.output[0]
            l0.output[0][:] = l0.attachment(acts, hook=True)

        with gpt2_edited.trace(batch):
            edited = l0.attachment.output.save()

        assert edited.shape[0] == 2


# =============================================================================
# Conditionals
# =============================================================================


class TestConditionals:
    """Tests for conditional interventions."""

    def test_conditional_intervention(self, gpt2: nnsight.LanguageModel):
        """Test conditional intervention based on tensor values."""
        with gpt2.session():
            with gpt2.trace("Hello World"):
                if torch.all(gpt2.transformer.h[5].output[0] < 100000):
                    gpt2.transformer.h[-1].output[0][:] = 0
                output = gpt2.transformer.h[-1].output[0].save()

        assert torch.all(output == 0)


# =============================================================================
# Input Setting
# =============================================================================


class TestInputSetting:
    """Tests for modifying module inputs."""

    def test_input_setting(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test setting module inputs."""
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


# =============================================================================
# Invoker Batching
# =============================================================================


class TestInvokerBatching:
    """Tests for batching with invokers."""

    @torch.no_grad()
    def test_invoker_group_batching(
        self, gpt2: nnsight.LanguageModel, ET_prompt: str, MSG_prompt: str
    ):
        """Test batching with multiple invoke groups."""
        with gpt2.trace() as tracer:
            with tracer.invoke(ET_prompt):
                out_1 = gpt2.lm_head.output[:, -1].save()

            with tracer.invoke([MSG_prompt, ET_prompt]):
                out_2 = gpt2.lm_head.output[:, -1].save()

            with tracer.invoke():
                out_3 = gpt2.lm_head.output[:, -1].save()

            with tracer.invoke():
                out_4 = gpt2.lm_head.output[:, -1].save()

            with tracer.invoke(MSG_prompt):
                out_5 = gpt2.lm_head.output[:, -1].save()

        assert out_1.shape[0] == 1
        assert out_2.shape[0] == 2
        assert out_3.shape[0] == 4
        assert out_4.shape[0] == 4
        assert out_5.shape[0] == 1

        assert torch.equal(out_1, out_2[1].unsqueeze(0))
        assert torch.equal(out_3, out_4)
        assert torch.equal(torch.concatenate([out_1, out_2, out_5]), out_3)


# =============================================================================
# Scanning and Validation
# =============================================================================


@pytest.mark.scan
class TestScan:
    """Tests for scan mode (shape inference)."""

    @torch.no_grad()
    def test_basic_scan(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test basic scanning for shapes."""
        with gpt2.scan(MSG_prompt):
            attn_input = gpt2.transformer.h[0].attn.input.save()
            attn_output = gpt2.transformer.h[0].attn.output[0].save()

        assert isinstance(attn_input, torch._subclasses.fake_tensor.FakeTensor)
        assert isinstance(attn_output, torch._subclasses.fake_tensor.FakeTensor)
        assert attn_input.shape == (1, 9, 768)
        assert attn_output.shape == (1, 9, 768)
        assert gpt2.transformer.h[1].output[0].shape == (1, 9, 768)

    @torch.no_grad()
    def test_scan_with_intervention(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test scanning with shape-changing intervention."""
        with gpt2.scan(MSG_prompt):
            gpt2.transformer.h[0].mlp.c_proj.output = torch.ones(1, 1, 768).to(gpt2.device)
            out = gpt2.transformer.h[0].mlp.c_proj.output.save()

        assert out.shape == (1, 1, 768)
        assert gpt2.transformer.h[1].output[0].shape == (1, 9, 768)

    @torch.no_grad()
    def test_scan_undispatched(self, MSG_prompt: str):
        """Test scanning on undispatched model."""
        gpt2_undispatched = nnsight.LanguageModel("openai-community/gpt2")

        with gpt2_undispatched.scan(MSG_prompt):
            pass

        assert gpt2_undispatched.dispatched == False
        assert gpt2_undispatched.transformer.h[1].output[0].shape == (1, 9, 768)


# =============================================================================
# Order Enforcement
# =============================================================================


@pytest.mark.order
class TestOrder:
    """Tests for execution order enforcement."""

    @torch.no_grad()
    def test_out_of_order_error(self, gpt2: nnsight.LanguageModel):
        """Test that accessing modules out of order raises error."""
        with pytest.raises(nnsight.intervention.interleaver.Mediator.OutOfOrderError):
            with gpt2.trace("_"):
                out = gpt2.transformer.h[2].output.save()
                out_2 = gpt2.transformer.h[1].inputs.save()

    @torch.no_grad()
    @pytest.mark.skips
    def test_out_of_order_skip(self, gpt2: nnsight.LanguageModel):
        """Test out of order error with skip."""
        with pytest.raises(nnsight.intervention.interleaver.Mediator.OutOfOrderError):
            with gpt2.trace("_"):
                gpt2.transformer.h[1].skip(gpt2.transformer.h[0].output)
                gpt2.transformer.h[1].input[:] = 0

    @torch.no_grad()
    @pytest.mark.skips
    def test_out_of_order_skip_2(self, gpt2: nnsight.LanguageModel):
        """Test out of order error with multiple skips."""
        with pytest.raises(nnsight.intervention.interleaver.Mediator.OutOfOrderError):
            with gpt2.trace("_"):
                inp = gpt2.transformer.h[0].output.save()
                gpt2.transformer.h[1].skip(inp)
                gpt2.transformer.h[0].skip(inp)


# =============================================================================
# Source Tracing
# =============================================================================


@pytest.mark.source
class TestSource:
    """Tests for source tracing (operation-level access)."""

    @torch.no_grad()
    def test_source_output(self, gpt2: nnsight.LanguageModel):
        """Test accessing operation output via source."""
        with gpt2.trace("_"):
            out = gpt2.transformer.h[0].attn.source.split_1.output.save()

        assert isinstance(out, tuple)

    @torch.no_grad()
    def test_source_inputs(self, gpt2: nnsight.LanguageModel):
        """Test accessing operation inputs via source."""
        with gpt2.trace("_"):
            inp = gpt2.transformer.h[0].attn.source.attention_interface_0.inputs.save()

        assert isinstance(inp, tuple)

    @torch.no_grad()
    def test_source_safe_intervention(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test that source tracing doesn't affect model."""
        input = gpt2.tokenizer(MSG_prompt, return_tensors="pt").to(gpt2.device)
        logits_0 = gpt2(**input)["logits"]

        gpt2.transformer.h[0].attn.source
        with gpt2.trace("_"):
            gpt2.transformer.h[0].attn.c_attn.output = torch.zeros_like(
                gpt2.transformer.h[0].attn.c_attn.output
            )
            out = gpt2.transformer.h[0].attn.source.split_1.output.save()

        logits_1 = gpt2(**input)["logits"]

        assert isinstance(out, tuple)
        assert torch.all(out[0] == 0)
        assert torch.all(logits_0 == logits_1)

    @torch.no_grad()
    def test_multiple_source(self, gpt2: nnsight.LanguageModel):
        """Test accessing multiple operations via source."""
        with gpt2.trace("_"):
            out_split_0 = gpt2.transformer.h[0].attn.source.split_1.output.save()
            out_attention_interface_0 = (
                gpt2.transformer.h[0].attn.source.attention_interface_0.output.save()
            )
            out_split_1 = gpt2.transformer.h[1].attn.source.split_1.output.save()
            out_attention_interface_1 = (
                gpt2.transformer.h[1].attn.source.attention_interface_0.output.save()
            )

        assert isinstance(out_split_0, tuple)
        assert isinstance(out_attention_interface_0, tuple)
        assert isinstance(out_split_1, tuple)
        assert isinstance(out_attention_interface_1, tuple)

    @torch.no_grad()
    def test_source_patching(self, gpt2: nnsight.LanguageModel):
        """Test patching operation outputs via source."""
        with gpt2.trace("_"):
            out = gpt2.transformer.h[0].attn.source.split_1.output
            out = (torch.zeros_like(out[0]),) + out[1:]
            gpt2.transformer.h[1].attn.source.split_1.output = out
            out_2 = gpt2.transformer.h[1].attn.source.split_1.output.save()

        assert isinstance(out_2, tuple)
        assert torch.all(out_2[0] == 0)

    @torch.no_grad()
    def test_recursive_source(self, gpt2: nnsight.LanguageModel):
        """Test recursive source tracing."""
        with gpt2.trace("_"):
            out = (
                gpt2.transformer.h[0]
                .attn.source.attention_interface_0.source.torch_nn_functional_scaled_dot_product_attention_0.output.save()
            )

        assert isinstance(out, torch.Tensor)

    @torch.no_grad()
    def test_source_imported_function(self, gpt2: nnsight.LanguageModel):
        """Test source tracing on imported functions."""
        with gpt2.trace("_"):
            gpt2.transformer.h[0].attn.source.split_1.source
            out = gpt2.transformer.h[0].attn.source.split_1.output.save()

        assert isinstance(out, tuple)

    @torch.no_grad()
    def test_source_operation_not_found(self, gpt2: nnsight.LanguageModel):
        """Test error when operation not found."""
        with pytest.raises(AttributeError):
            with gpt2.trace("_"):
                out = gpt2.transformer.h[0].attn.source.my_func.output.save()

    @torch.no_grad()
    def test_operation_envoy_update(self, MSG_prompt: str):
        """Test operation envoy updates with new model."""
        gpt2 = nnsight.LanguageModel("openai-community/gpt2")
        fn = gpt2.transformer.h[0].attn.source.split_1

        with gpt2.trace(MSG_prompt):
            out = fn.output.save()

        assert isinstance(out, tuple)

    @torch.no_grad()
    def test_source_output_with_barrier(
        self, gpt2: nnsight.LanguageModel, ET_prompt: str, MSG_prompt: str
    ):
        """Test source with barrier for cross-invoke patching."""
        with gpt2.trace() as tracer:
            barrier = tracer.barrier(2)

            with tracer.invoke(ET_prompt):
                attn_out = (
                    gpt2.transformer.h[0].attn.source.attention_interface_0.output[0].save()
                )
                barrier()

            with tracer.invoke(MSG_prompt):
                barrier()
                gpt2.transformer.h[0].attn.source.attention_interface_0.output[0][
                    :, -1, 0, :
                ] = attn_out[:, -1, 0, :]
                attn_out_2 = (
                    gpt2.transformer.h[0].attn.source.attention_interface_0.output[0].save()
                )

        assert torch.all(attn_out[:, -1, 0, :] == attn_out_2[:, -1, 0, :])


# =============================================================================
# Module Skipping
# =============================================================================


@pytest.mark.skips
class TestSkip:
    """Tests for module skipping."""

    @torch.no_grad()
    def test_basic_skip(self, gpt2: nnsight.LanguageModel):
        """Test basic module skip."""
        with gpt2.trace("_"):
            inp = gpt2.transformer.h[0].output.save()
            gpt2.transformer.h[1].skip(inp)
            out = gpt2.transformer.h[1].output.save()

        assert torch.equal(out[0], inp[0])

    @torch.no_grad()
    def test_skip_with_input_modification(self, gpt2: nnsight.LanguageModel):
        """Test skip overrides input modification."""
        with gpt2.trace("_"):
            inp = gpt2.transformer.h[0].output.save()
            gpt2.transformer.h[1].inputs = (
                (torch.zeros_like(gpt2.transformer.h[1].input),)
            ) + gpt2.transformer.h[1].inputs[1:]
            gpt2.transformer.h[1].skip(inp)
            out = gpt2.transformer.h[1].output.save()

        assert not torch.all(out[0] == 0)
        assert torch.equal(out[0], inp[0])

    @torch.no_grad()
    def test_multiple_skip(self, gpt2: nnsight.LanguageModel):
        """Test multiple consecutive skips."""
        with gpt2.trace("_"):
            inp = gpt2.transformer.h[0].output
            gpt2.transformer.h[1].skip(inp)
            inp_2 = gpt2.transformer.h[1].output
            gpt2.transformer.h[2].skip(inp_2)
            inp_3 = gpt2.transformer.h[2].output.save()
            gpt2.transformer.h[3].skip(inp_3)
            out = gpt2.transformer.h[3].output.save()

        assert torch.equal(out[0], inp_3[0])

    @torch.no_grad()
    def test_skip_inner_module_error(self, gpt2: nnsight.LanguageModel):
        """Test error when accessing inner module of skipped module."""
        with pytest.raises(ValueError):
            with gpt2.trace("Hello World"):
                inp = gpt2.transformer.h[0].output
                gpt2.transformer.h[1].skip(inp)
                hs = gpt2.transformer.h[1].attn.output.save()

    @torch.no_grad()
    def test_skip_batched(
        self, gpt2: nnsight.LanguageModel, ET_prompt: str, MSG_prompt: str
    ):
        """Test skip with batched inputs."""
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

    @torch.no_grad()
    def test_skip_partial_batch_error(
        self, gpt2: nnsight.LanguageModel, ET_prompt: str, MSG_prompt: str
    ):
        """Test error when skipping only partial batch."""
        with pytest.raises(ValueError):
            with gpt2.trace() as tracer:
                with tracer.invoke(ET_prompt):
                    inp = gpt2.transformer.h[0].output.save()
                    gpt2.transformer.h[1].skip(inp)

                with tracer.invoke(MSG_prompt):
                    inp_2 = gpt2.transformer.h[0].output.save()


# =============================================================================
# Iteration
# =============================================================================


@pytest.mark.iter
class TestIteration:
    """Tests for iteration over generation steps."""

    @torch.no_grad()
    def test_iter_and_all(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test tracer.iter and tracer.all() equivalence."""
        with gpt2.generate(MSG_prompt, max_new_tokens=3) as tracer:
            logits_all = list().save()
            with tracer.all():
                logits_all.append(gpt2.lm_head.output[0][-1].argmax(dim=-1))

        with gpt2.generate(MSG_prompt, max_new_tokens=3) as tracer:
            logits_iter = list().save()
            with tracer.iter[:]:
                logits_iter.append(gpt2.lm_head.output[0][-1].argmax(dim=-1))

        assert len(logits_all) == 3
        assert len(logits_iter) == 3
        assert gpt2.tokenizer.batch_decode(logits_all) == [" New", " York", " City"]
        assert gpt2.tokenizer.batch_decode(logits_iter) == [" New", " York", " City"]

    @torch.no_grad()
    def test_iter_slice(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test iteration with slice."""
        with gpt2.generate(MSG_prompt, max_new_tokens=5) as tracer:
            logits = list().save()
            with tracer.iter[1:3]:
                logits.append(gpt2.lm_head.output[0][-1].argmax(dim=-1))

        assert len(logits) == 2
        assert gpt2.tokenizer.batch_decode(logits) == [" York", " City"]

    @torch.no_grad()
    def test_iter_idx(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test iteration with index."""
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

    @torch.no_grad()
    def test_batched_iter(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test batched iteration."""
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

    @torch.no_grad()
    @pytest.mark.skips
    def test_iter_and_skip(self, gpt2: nnsight.LanguageModel):
        """Test iteration with skip."""
        with gpt2.generate("_", max_new_tokens=3) as tracer:
            arr_gen = list().save()

            with tracer.iter[:] as it:
                if it != 1:
                    arr_gen.append(gpt2.transformer.h[1].output[0])
                else:
                    gpt2.transformer.h[1].skip(
                        (torch.zeros_like(gpt2.transformer.h[0].output[0]),)
                        + gpt2.transformer.h[0].output[1:]
                    )
                    arr_gen.append(gpt2.transformer.h[1].output[0])

        assert not torch.all(arr_gen[0] == 0)
        assert torch.all(arr_gen[1] == 0)
        assert not torch.all(arr_gen[2] == 0)

    @torch.no_grad()
    def test_iter_with_promptless_invokers(
        self, gpt2: nnsight.LanguageModel, ET_prompt: str
    ):
        """Test iteration with promptless invokers."""
        with gpt2.trace() as tracer:
            with tracer.invoke(ET_prompt):
                pass

            with tracer.invoke():
                out = gpt2.transformer.h[0].output[0].clone().save()

            with tracer.invoke():
                with tracer.iter[0]:
                    gpt2.transformer.h[0].output[0][:] = 0

            with tracer.invoke():
                out_2 = gpt2.transformer.h[0].output[0].save()

        assert not torch.all(out[0] == 0)
        assert torch.all(out_2[0] == 0)

    @torch.no_grad()
    def test_slice_iter_with_envoy_called_before(self, gpt2: nnsight.LanguageModel):
        """Test slice iteration after accessing envoy."""
        logits = list()
        with gpt2.generate("_", max_new_tokens=5) as tracer:
            out = gpt2.transformer.h[0].output[0].clone().save()

            with tracer.iter[2:4]:
                logits.append(gpt2.lm_head.output.save())

        assert isinstance(out, torch.Tensor)
        assert len(logits) == 2

    @torch.no_grad()
    def test_iter_with_batched_interventions(
        self, gpt2: nnsight.LanguageModel, ET_prompt: str, MSG_prompt: str
    ):
        """Test iteration with batched interventions."""
        with gpt2.generate(max_new_tokens=3) as tracer:
            with tracer.invoke(ET_prompt):
                logits_1 = list().save()
                with tracer.iter[:]:
                    logits_1.append(gpt2.lm_head.output[0][-1].argmax(dim=-1))

            with tracer.invoke(MSG_prompt):
                logits_2 = list().save()
                with tracer.iter[0:3]:
                    logits_2.append(gpt2.lm_head.output[0][-1].argmax(dim=-1))

        assert all(
            [
                not torch.equal(logit_1, logit_2)
                for logit_1, logit_2 in zip(logits_1, logits_2)
            ]
        )

    @torch.no_grad()
    def test_iter_module(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test iteration accessing different layers."""
        with gpt2.trace(MSG_prompt, max_new_tokens=3) as tracer:
            logits = list()
            with tracer.iter[:]:
                logits.append(gpt2.lm_head.output[0][-1].argmax(dim=-1))


# =============================================================================
# Caching
# =============================================================================


@pytest.mark.cache
class TestCache:
    """Tests for activation caching."""

    @torch.no_grad()
    def test_basic_cache(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test basic caching."""
        with gpt2.trace(MSG_prompt) as tracer:
            cache = tracer.cache()

        assert cache["model.transformer.h.0"].output is not None
        assert cache["model.transformer.h.0"].inputs is None

    @torch.no_grad()
    def test_cache_output_and_inputs(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test caching both outputs and inputs."""
        with gpt2.trace(MSG_prompt) as tracer:
            cache = tracer.cache(include_inputs=True)

        assert torch.equal(
            cache["model.transformer.h.0"].output[0],
            cache["model.transformer.h.1"].inputs[0][0],
        )

    @torch.no_grad()
    def test_cache_inputs_only(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test caching inputs only."""
        with gpt2.trace(MSG_prompt) as tracer:
            cache = tracer.cache(include_inputs=True, include_output=False)

        assert cache["model.transformer.h.0"].inputs is not None
        assert cache["model.transformer.h.0"].output is None

    @torch.no_grad()
    def test_cache_generation(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test caching during generation."""
        with gpt2.generate(MSG_prompt, max_new_tokens=3) as tracer:
            cache = tracer.cache(modules=[gpt2.transformer.h[0].attn.c_attn])

        assert len(cache["model.transformer.h.0.attn.c_attn"]) == 3

    @torch.no_grad()
    def test_cache_with_intervention(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test cache captures intervened values."""
        with gpt2.trace(MSG_prompt) as tracer:
            cache = tracer.cache()
            gpt2.transformer.h[0].attn.c_attn.output = torch.zeros_like(
                gpt2.transformer.h[0].attn.c_attn.output
            )

        assert torch.all(cache["model.transformer.h.0.attn.c_attn"].output == 0)

    @torch.no_grad()
    def test_cache_single_invoker(
        self, gpt2: nnsight.LanguageModel, MSG_prompt: str, ET_prompt: str
    ):
        """Test cache for single invoker in batch."""
        with gpt2.trace() as tracer:
            with tracer.invoke(MSG_prompt):
                pass

            with tracer.invoke(ET_prompt):
                cache = tracer.cache()
                attn_out = gpt2.transformer.h[0].attn.c_attn.output.cpu().save()

        assert torch.equal(
            cache["model.transformer.h.0.attn.c_attn"].output[0], attn_out[0]
        )

    @torch.no_grad()
    def test_cache_after_intervention(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test cache started after some interventions."""
        with gpt2.trace(MSG_prompt) as tracer:
            gpt2.transformer.h[1].attn.c_attn.output = torch.zeros_like(
                gpt2.transformer.h[1].attn.c_attn.output
            )
            cache = tracer.cache(include_inputs=True)

        assert "model.transformer.h.0" not in cache
        assert cache["model.transformer"].inputs is None
        assert cache["model.transformer.h.1.attn.c_attn"].inputs is None
        assert torch.equal(
            cache["model.transformer.h.2"].output[0],
            cache["model.transformer.h.3"].inputs[0][0],
        )
        assert cache["model.transformer.h.1.attn.c_attn"].output is not None
        assert cache["model.transformer"].output is not None

    @torch.no_grad()
    def test_cache_skip(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test cache with module skipping."""
        with gpt2.trace(MSG_prompt) as tracer:
            cache = tracer.cache()
            out = gpt2.transformer.h[0].output.save()
            gpt2.transformer.h[1].skip(gpt2.transformer.h[0].output)

        assert torch.equal(cache["model.transformer.h.1"].output[0], out[0].cpu())

    @torch.no_grad()
    def test_cache_some_modules(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test caching specific modules by name."""
        with gpt2.trace(MSG_prompt) as tracer:
            cache = tracer.cache(
                modules=[
                    "model.transformer.h.0",
                    "model.transformer.h.1",
                    "model.transformer.h.2",
                ],
                include_inputs=True,
            )

        assert len(cache.keys()) == 3
        assert torch.equal(
            cache["model.transformer.h.0"].output[0],
            cache["model.transformer.h.1"].inputs[0][0],
        )

    @torch.no_grad()
    def test_cache_some_modules_by_ref(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test caching specific modules by reference."""
        with gpt2.trace(MSG_prompt) as tracer:
            cache = tracer.cache(
                modules=[module for module in gpt2.transformer.h], include_inputs=True
            )

        assert len(cache.keys()) == 12
        assert torch.equal(
            cache["model.transformer.h.0"].output[0],
            cache["model.transformer.h.1"].inputs[0][0],
        )

    @torch.no_grad()
    def test_cache_some_modules_generation(
        self, gpt2: nnsight.LanguageModel, MSG_prompt: str
    ):
        """Test caching specific modules during generation."""
        with gpt2.generate(MSG_prompt, max_new_tokens=2) as tracer:
            cache = tracer.cache(
                modules=[
                    "model.transformer.h.0",
                    "model.transformer.h.1",
                    "model.transformer.h.2",
                ]
            )

        assert len(cache.keys()) == 3
        assert len(cache["model.transformer.h.0"]) == 2

    @torch.no_grad()
    def test_multiple_caches(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test multiple separate caches."""
        with gpt2.trace(MSG_prompt) as tracer:
            attn_cache = tracer.cache(modules=[layer.attn for layer in gpt2.transformer.h])
            mlp_cache = tracer.cache(modules=[layer.mlp for layer in gpt2.transformer.h])

        assert len(attn_cache.keys()) == 12
        assert len(mlp_cache.keys()) == 12
        assert (
            "model.transformer.h.0.attn" in attn_cache
            and "model.transformer.h.0.mlp" not in attn_cache
        )
        assert (
            "model.transformer.h.0.mlp" in mlp_cache
            and "model.transformer.h.0.attn" not in mlp_cache
        )

    @torch.no_grad()
    def test_cache_attribute_access(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test cache with attribute-style access."""
        with gpt2.trace(MSG_prompt) as tracer:
            modules = [layer for layer in gpt2.transformer.h] + [gpt2.lm_head]
            cache = tracer.cache(modules=modules)

        assert len(cache) == 13
        assert cache["model.transformer.h.0"].output is not None
        assert cache.model.transformer.h[0].output is not None
        assert cache.model.transformer["h.0"].output is not None
        assert torch.equal(
            cache["model.transformer.h.0"].output[0],
            cache.model.transformer.h[0].output[0],
        )
        assert torch.equal(
            cache["model.transformer.h.0"].output[0],
            cache.model.transformer["h.0"].output[0],
        )
        assert cache.model.lm_head.output is not None

        with pytest.raises(IndexError):
            cache.model.transformer.h[12]

        with pytest.raises(AttributeError):
            cache.model.transformer.h[0].mlp

        with pytest.raises(KeyError):
            cache.model.transformer[0]

    @torch.no_grad()
    def test_cache_no_entry_input(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test cache with no input entry."""
        with gpt2.trace(MSG_prompt) as tracer:
            cache = tracer.cache(modules=[gpt2.transformer.h[0]])

        assert cache["model.transformer.h.0"].input is None

    @torch.no_grad()
    def test_cache_alias(self, MSG_prompt: str):
        """Test cache with module aliases."""
        gpt2 = nnsight.LanguageModel(
            "openai-community/gpt2",
            rename={"transformer": "model", "h.0": "first_layer", "1": "second_layer"},
        )

        with gpt2.trace(MSG_prompt) as tracer:
            cache = tracer.cache()

        assert torch.equal(
            cache["model.transformer.h.0"].output[0],
            cache.model.model.first_layer.output[0],
        )
        assert torch.equal(
            cache["model.transformer.h.1"].output[0],
            cache.model.model.h["second_layer"].output[0],
        )
        assert torch.equal(
            cache.model.transformer.h[0].output[0],
            cache.model.model.first_layer.output[0],
        )
        assert torch.equal(
            cache.model.transformer.h[1].output[0],
            cache.model.model.h["second_layer"].output[0],
        )
        assert torch.equal(
            cache["model.model.first_layer"].output[0],
            cache.model.model.first_layer.output[0],
        )
        assert torch.equal(
            cache["model.model.h.second_layer"].output[0],
            cache.model.model.h["second_layer"].output[0],
        )


# =============================================================================
# Module Renaming
# =============================================================================


@pytest.mark.rename
class TestRename:
    """Tests for module renaming/aliasing."""

    @torch.no_grad()
    def test_rename_module(self, MSG_prompt: str):
        """Test renaming a module."""
        gpt2 = nnsight.LanguageModel("openai-community/gpt2", rename={"mlp": "my_mlp"})

        with gpt2.trace(MSG_prompt):
            mlp_out_0 = gpt2.transformer.h[0].mlp.output.save()
            mlp_out_1 = gpt2.transformer.h[1].my_mlp.output.save()

        assert mlp_out_0 is not None
        assert mlp_out_1 is not None

    @torch.no_grad()
    def test_rename_path(self, MSG_prompt: str):
        """Test renaming a full path."""
        gpt2 = nnsight.LanguageModel(
            "openai-community/gpt2", rename={"transformer.h.3.mlp": "my_mlp"}
        )

        with gpt2.trace(MSG_prompt):
            mlp_out_0 = gpt2.transformer.h[0].mlp.output.save()
            mlp_out_3 = gpt2.transformer.h[3].mlp.output.save()
            my_mlp_out_3 = gpt2.my_mlp.output.save()
            mlp_out_5 = gpt2.transformer.h[5].mlp.output.save()

        assert mlp_out_0 is not None
        assert mlp_out_3 is not None
        assert my_mlp_out_3 is not None and torch.equal(mlp_out_3, my_mlp_out_3)
        assert mlp_out_5 is not None

    @torch.no_grad()
    def test_rename_module_list(self, MSG_prompt: str):
        """Test renaming a module list."""
        gpt2 = nnsight.LanguageModel("openai-community/gpt2", rename={".h": "layers"})

        with gpt2.trace(MSG_prompt):
            mlp_out_0 = gpt2.transformer.layers[0].mlp.output.save()
            mlp_out_1 = gpt2.transformer.layers[1].mlp.output.save()

        assert mlp_out_0 is not None
        assert mlp_out_1 is not None

    @torch.no_grad()
    def test_rename_module_list_path(self, MSG_prompt: str):
        """Test renaming a module list with full path."""
        gpt2 = nnsight.LanguageModel("openai-community/gpt2", rename={"transformer.h": "layers"})

        with gpt2.trace(MSG_prompt):
            mlp_out_0 = gpt2.layers[0].mlp.output.save()
            mlp_out_1 = gpt2.layers[1].mlp.output.save()

        assert mlp_out_0 is not None
        assert mlp_out_1 is not None


# =============================================================================
# Miscellaneous
# =============================================================================


class TestMiscellaneous:
    """Miscellaneous tests."""

    def test_undispatched_extra_module(self, device: str, MSG_prompt: str):
        """Test undispatched model with extra module access."""
        model = nnsight.LanguageModel(
            "openai-community/gpt2", device_map=device, dispatch=False
        )

        with model.generate(MSG_prompt, max_new_tokens=1):
            output = model.generator.output.save()

        output

    def test_iter_with_streamer1(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test iteration with streamer output."""
        with gpt2.generate(MSG_prompt, max_new_tokens=10) as tracer:
            hs = list()
            hs.save()

            strm = list()
            strm.save()

            with tracer.iter[2:6]:
                hs.append(gpt2.transformer.h[-2].output[0])
                strm.append(gpt2.generator.streamer.output)

            with tracer.iter[6:8]:
                hs.append(gpt2.transformer.h[-2].output[0])
                strm.append(gpt2.generator.streamer.output)

        assert len(hs) == 6
        assert len(strm) == 6

    def test_iter_with_streamer2(self, gpt2: nnsight.LanguageModel, MSG_prompt: str):
        """Test iteration with streamer output accessed before iter."""
        with gpt2.generate(MSG_prompt, max_new_tokens=10) as tracer:
            hs = list()
            hs.save()

            strm = list()
            strm.save()

            strm.append(gpt2.generator.streamer.output)

            with tracer.iter[2:6]:
                hs.append(gpt2.transformer.h[-2].output[0])
                strm.append(gpt2.generator.streamer.output)

            with tracer.iter[6:8]:
                hs.append(gpt2.transformer.h[-2].output[0])
                strm.append(gpt2.generator.streamer.output)

        assert len(hs) == 6
        assert len(strm) == 7
