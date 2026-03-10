"""Tests for the .source feature — operation-level tracing and intervention.

Covers basic source access, interventions, the forward wrapper chain
(nnsight skip + accelerate device_map), and recursive source tracing.
"""

import pytest
import torch

import nnsight


@pytest.fixture(scope="module")
def gpt2(device: str):
    return nnsight.LanguageModel(
        "openai-community/gpt2", device_map=device, dispatch=True
    )


@pytest.fixture
def prompt():
    return "The Eiffel Tower is in"


class TestSourceAccess:
    """Basic .source access — reading operation outputs and inputs."""

    @torch.no_grad()
    def test_source_output(self, gpt2: nnsight.LanguageModel):
        with gpt2.trace("_"):
            out = gpt2.transformer.h[0].attn.source.split_1.output.save()

        assert isinstance(out, tuple)

    @torch.no_grad()
    def test_source_inputs(self, gpt2: nnsight.LanguageModel):
        with gpt2.trace("_"):
            inp = (
                gpt2.transformer.h[0]
                .attn.source.attention_interface_0.inputs.save()
            )

        assert isinstance(inp, tuple)

    @torch.no_grad()
    def test_multiple_source_across_layers(self, gpt2: nnsight.LanguageModel):
        with gpt2.trace("_"):
            out_0 = gpt2.transformer.h[0].attn.source.split_1.output.save()
            out_1 = gpt2.transformer.h[1].attn.source.split_1.output.save()

        assert isinstance(out_0, tuple)
        assert isinstance(out_1, tuple)

    @torch.no_grad()
    def test_recursive_source(self, gpt2: nnsight.LanguageModel):
        with gpt2.trace("_"):
            out = (
                gpt2.transformer.h[0]
                .attn.source.attention_interface_0
                .source.torch_nn_functional_scaled_dot_product_attention_0
                .output.save()
            )

        assert isinstance(out, torch.Tensor)


class TestSourceIntervention:
    """Modifying operation-level values via .source."""

    @torch.no_grad()
    def test_source_patching(self, gpt2: nnsight.LanguageModel):
        with gpt2.trace("_"):
            out = gpt2.transformer.h[0].attn.source.split_1.output
            out = (torch.zeros_like(out[0]),) + out[1:]
            gpt2.transformer.h[1].attn.source.split_1.output = out
            patched = gpt2.transformer.h[1].attn.source.split_1.output.save()

        assert isinstance(patched, tuple)
        assert torch.all(patched[0] == 0)

    @torch.no_grad()
    def test_source_does_not_affect_plain_forward(
        self, gpt2: nnsight.LanguageModel, prompt: str
    ):
        """Accessing .source and running a trace should not change plain model output."""
        tokens = gpt2.tokenizer(prompt, return_tensors="pt").to(gpt2.device)
        logits_before = gpt2(**tokens)["logits"]

        # Trigger source injection
        gpt2.transformer.h[0].attn.source
        with gpt2.trace("_"):
            _ = gpt2.transformer.h[0].attn.source.split_1.output.save()

        logits_after = gpt2(**tokens)["logits"]
        assert torch.allclose(logits_before, logits_after)


class TestSourceWrapperChain:
    """Verify .source preserves the nnsight forward wrapper chain."""

    @torch.no_grad()
    def test_nnsight_forward_preserved(self, gpt2: nnsight.LanguageModel):
        """After .source injection, __nnsight_forward__ should exist and be swapped."""
        module = gpt2.transformer.h[0].attn._module
        assert hasattr(module, "__nnsight_forward__")

        # Trigger source injection
        gpt2.transformer.h[0].attn.source

        # __nnsight_forward__ should still exist — .source swaps it, not module.forward
        assert hasattr(module, "__nnsight_forward__")

    @torch.no_grad()
    def test_nnsight_skip_preserved(self, gpt2: nnsight.LanguageModel):
        """After .source injection, __nnsight_skip__ should still exist."""
        module = gpt2.transformer.h[0].attn._module
        assert hasattr(module, "__nnsight_skip__")

        gpt2.transformer.h[0].attn.source

        assert hasattr(module, "__nnsight_skip__")

    @torch.no_grad()
    def test_source_then_trace_works(self, gpt2: nnsight.LanguageModel):
        """A normal trace should still work after .source has been accessed."""
        gpt2.transformer.h[0].attn.source

        with gpt2.trace("_"):
            out = gpt2.transformer.h[0].attn.output[0].save()

        assert isinstance(out, torch.Tensor)
        assert out.ndim == 3


class TestSourceErrors:
    """Edge cases and expected errors."""

    @torch.no_grad()
    def test_nonexistent_operation_raises(self, gpt2: nnsight.LanguageModel):
        with pytest.raises(AttributeError):
            with gpt2.trace("_"):
                gpt2.transformer.h[0].attn.source.does_not_exist_xyz.output.save()
