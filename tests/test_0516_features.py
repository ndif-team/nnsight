"""
Tests for nnsight v0.5.16 features.

These tests cover the major new features in 0.5.16:
- Keyword-only trace arguments (smart invoker detection)
- Custom functions in traces
- Local simulation mode (remote='local')
- Multi-invoker generation
- Clean exception tracebacks
"""

import pytest
import torch
import nnsight


# =============================================================================
# Keyword-Only Trace Arguments (Smart Invoker Detection)
# =============================================================================


class TestKeywordOnlyTraceArgs:
    """Tests for the new trace behavior with keyword-only arguments.

    In 0.5.16, nnsight detects whether arguments affect batch size to determine
    if invokers are needed, rather than just checking for positional args.
    """

    @torch.no_grad()
    def test_input_ids_keyword_only(self, gpt2: nnsight.LanguageModel):
        """Test that input_ids as keyword-only arg creates implicit invoker."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        with gpt2.trace(input_ids=input_ids):
            hidden = gpt2.transformer.h[0].output[0].save()

        assert hidden is not None
        assert isinstance(hidden, torch.Tensor)
        assert hidden.shape[1] == 5  # sequence length matches input

    @torch.no_grad()
    def test_attention_mask_keyword(self, gpt2: nnsight.LanguageModel):
        """Test with both input_ids and attention_mask as keywords."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1]])

        with gpt2.trace(input_ids=input_ids, attention_mask=attention_mask):
            hidden = gpt2.transformer.h[0].output[0].save()

        assert hidden is not None
        assert hidden.shape[1] == 5


# =============================================================================
# Custom Functions in Traces
# =============================================================================


class TestCustomFunctions:
    """Tests for using custom functions inside traces."""

    @torch.no_grad()
    def test_custom_analysis_function(self, gpt2: nnsight.LanguageModel):
        """Test that custom functions work inside traces."""

        def analyze_layer(model, layer_idx=0):
            """Custom analysis function."""
            return model.transformer.h[layer_idx].output[0].mean(dim=-1)

        with gpt2.trace("The quick brown fox"):
            result = analyze_layer(gpt2, layer_idx=5).save()

        assert result is not None
        assert isinstance(result, torch.Tensor)

    @torch.no_grad()
    def test_custom_function_with_loop(self, gpt2: nnsight.LanguageModel):
        """Test custom function that iterates over layers."""

        def multi_layer_mean(model, num_layers=3):
            """Compute mean across multiple layers."""
            outputs = []
            for i in range(num_layers):
                outputs.append(model.transformer.h[i].output[0].mean())
            return sum(outputs)

        with gpt2.trace("Test input"):
            result = multi_layer_mean(gpt2, num_layers=3).save()

        assert result is not None


# =============================================================================
# Local Simulation Mode
# =============================================================================


class TestLocalSimulation:
    """Tests for remote='local' serialization simulation."""

    @torch.no_grad()
    def test_basic_local_simulation(self, gpt2: nnsight.LanguageModel):
        """Test that remote='local' simulates serialization correctly."""
        with gpt2.trace("Hello world", remote="local"):
            hidden = gpt2.transformer.h[0].output[0].save()

        assert hidden is not None
        assert isinstance(hidden, torch.Tensor)
        assert hidden.ndim == 3

    @torch.no_grad()
    def test_local_simulation_with_custom_function(self, gpt2: nnsight.LanguageModel):
        """Test that custom functions serialize correctly in local simulation."""

        def custom_analysis(model):
            """A function to test serialization."""
            layer_outputs = []
            for i in range(3):
                layer_outputs.append(model.transformer.h[i].output[0].mean())
            return sum(layer_outputs)

        with gpt2.trace("Test input", remote="local"):
            result = custom_analysis(gpt2).save()

        assert result is not None


# =============================================================================
# Multi-Invoker Generation
# =============================================================================


class TestMultiInvoker:
    """Tests for multi-invoker generation with config kwargs."""

    @torch.no_grad()
    def test_generate_with_invokers(self, gpt2: nnsight.LanguageModel):
        """Test that config kwargs allow sub-invokers."""
        with gpt2.generate(max_new_tokens=3) as tracer:
            with tracer.invoke("Hello"):
                hidden_1 = gpt2.transformer.h[0].output[0].save()
            with tracer.invoke("Goodbye"):
                hidden_2 = gpt2.transformer.h[0].output[0].save()

        assert hidden_1 is not None
        assert hidden_2 is not None
        assert isinstance(hidden_1, torch.Tensor)
        assert isinstance(hidden_2, torch.Tensor)

    @torch.no_grad()
    def test_multiple_invokers_independent(self, gpt2: nnsight.LanguageModel):
        """Test that multiple invokers can access different layers independently."""
        with gpt2.trace() as tracer:
            with tracer.invoke("Hello world"):
                hidden_layer_0 = gpt2.transformer.h[0].output[0].save()
            with tracer.invoke("Goodbye world"):
                hidden_layer_5 = gpt2.transformer.h[5].output[0].save()

        # Both should be valid tensors with correct hidden dim
        assert hidden_layer_0 is not None
        assert hidden_layer_5 is not None
        assert hidden_layer_0.shape[-1] == 768  # GPT-2 hidden dim
        assert hidden_layer_5.shape[-1] == 768


# =============================================================================
# Exception Handling
# =============================================================================


class TestExceptionHandling:
    """Tests for clean exception traceback handling."""

    def test_index_error_traceback(self, gpt2: nnsight.LanguageModel):
        """Test that IndexError is raised cleanly for invalid layer index."""
        with pytest.raises(IndexError):
            with gpt2.trace("Hello world"):
                # Layer 999 doesn't exist
                output = gpt2.transformer.h[999].output.save()

    def test_exception_wrapper_type(self, gpt2: nnsight.LanguageModel):
        """Test that exceptions are wrapped with ExceptionWrapper."""
        from nnsight.intervention.tracing.util import ExceptionWrapper

        try:
            with gpt2.trace("Hello world"):
                output = gpt2.transformer.h[999].output.save()
        except IndexError as e:
            # Should be wrapped in ExceptionWrapper
            assert isinstance(e, ExceptionWrapper)

    def test_attribute_error_traceback(self, gpt2: nnsight.LanguageModel):
        """Test that AttributeError is raised cleanly."""
        with pytest.raises(AttributeError):
            with gpt2.trace("Hello world"):
                # 'nonexistent' attribute doesn't exist
                output = gpt2.transformer.nonexistent.output.save()
