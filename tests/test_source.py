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
            inp = gpt2.transformer.h[0].attn.source.attention_interface_0.inputs.save()

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
            out = gpt2.transformer.h[
                0
            ].attn.source.attention_interface_0.source.torch_nn_functional_scaled_dot_product_attention_0.output.save()

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


class TestSourceIter:
    """`.source` combined with ``tracer.iter[...]`` — operation-level
    intervention across multi-step generation.

    These tests exercise the per-mediator iteration tracker for
    operation paths, which the persistent iter hook in
    :func:`_register_iter_hooks` keeps in sync with the parent module's
    tracker by walking the module's ``SourceAccessor`` after each forward
    pass (recursing into nested accessors for recursive ``.source``).
    """

    @torch.no_grad()
    def test_iter_source_output_all_steps(self, gpt2: nnsight.LanguageModel):
        """Save an op output on every generation step via ``iter[:]``."""
        # Touch .source before iter to keep this a Phase-1 case (the
        # mid-loop-first-access edge case is a known limitation).
        gpt2.transformer.h[0].attn.source

        with gpt2.generate("The Eiffel Tower is in", max_new_tokens=3) as tracer:
            outs = list().save()
            for step in tracer.iter[:]:
                outs.append(gpt2.transformer.h[0].attn.source.split_1.output)

        assert len(outs) == 3
        for out in outs:
            assert isinstance(out, tuple)

    @torch.no_grad()
    def test_iter_source_input_all_steps(self, gpt2: nnsight.LanguageModel):
        """Save an op input on every generation step via ``iter[:]``."""
        gpt2.transformer.h[0].attn.source

        with gpt2.generate("Hello", max_new_tokens=3) as tracer:
            inputs = list().save()
            for step in tracer.iter[:]:
                inputs.append(
                    gpt2.transformer.h[0].attn.source.attention_interface_0.inputs
                )

        assert len(inputs) == 3
        for inp in inputs:
            assert isinstance(inp, tuple)

    @torch.no_grad()
    def test_iter_source_specific_step(self, gpt2: nnsight.LanguageModel):
        """``iter[N]`` should fire op hooks only on step N."""
        gpt2.transformer.h[0].attn.source

        with gpt2.generate("Hello", max_new_tokens=3) as tracer:
            outs = list().save()
            for step in tracer.iter[1]:
                outs.append(gpt2.transformer.h[0].attn.source.split_1.output)

        assert len(outs) == 1

    @torch.no_grad()
    def test_iter_source_slice(self, gpt2: nnsight.LanguageModel):
        """``iter[a:b]`` should fire op hooks only on the requested range."""
        gpt2.transformer.h[0].attn.source

        with gpt2.generate("Hello", max_new_tokens=4) as tracer:
            outs = list().save()
            for step in tracer.iter[1:3]:
                outs.append(gpt2.transformer.h[0].attn.source.split_1.output)

        assert len(outs) == 2

    @torch.no_grad()
    def test_iter_source_intervention(self, gpt2: nnsight.LanguageModel):
        """Modifying an op output every step should change generation.

        Zeroes ``attention_interface_0`` (the main attention computation)
        on every layer at every generation step. This is heavy enough to
        flip at least one sampled token vs. the unperturbed baseline.
        """
        for i in range(len(gpt2.transformer.h)):
            gpt2.transformer.h[i].attn.source

        with gpt2.generate("The Eiffel Tower is in", max_new_tokens=3) as tracer:
            base = tracer.result.save()

        with gpt2.generate("The Eiffel Tower is in", max_new_tokens=3) as tracer:
            for step in tracer.iter[:]:
                for i in range(len(gpt2.transformer.h)):
                    attn_out = (
                        gpt2.transformer.h[i].attn.source.attention_interface_0.output
                    )
                    gpt2.transformer.h[i].attn.source.attention_interface_0.output = (
                        torch.zeros_like(attn_out[0]),
                    ) + attn_out[1:]
            patched = tracer.result.save()

        assert not torch.equal(base, patched)

    @torch.no_grad()
    def test_iter_recursive_source(self, gpt2: nnsight.LanguageModel):
        """Recursive ``.source`` (op-of-op) under ``iter[:]``.

        This exercises the recursive op-path tracker bumping in
        :func:`_bump_source_paths` — ``...attn.attention_interface_0`` and
        its nested ``...scaled_dot_product_attention_0`` paths must both
        advance in lockstep with the parent module each step.
        """
        gpt2.transformer.h[0].attn.source

        with gpt2.generate("Hello", max_new_tokens=3) as tracer:
            outs = list().save()
            for step in tracer.iter[:]:
                outs.append(
                    gpt2.transformer.h[
                        0
                    ].attn.source.attention_interface_0.source.torch_nn_functional_scaled_dot_product_attention_0.output
                )

        assert len(outs) == 3
        for out in outs:
            assert isinstance(out, torch.Tensor)

    @torch.no_grad()
    def test_iter_source_sparse(self, gpt2: nnsight.LanguageModel):
        """Skipping op access on some steps should still work on later steps.

        The persistent iter hook bumps op-path trackers every forward
        pass regardless of whether the user registered an op hook for
        that step, so a step-N hook still finds its tracker at N.
        """
        gpt2.transformer.h[0].attn.source

        with gpt2.generate("Hello", max_new_tokens=4) as tracer:
            outs = list().save()
            for step in tracer.iter[:]:
                if step in (0, 2):
                    outs.append(gpt2.transformer.h[0].attn.source.split_1.output)

        assert len(outs) == 2
        for out in outs:
            assert isinstance(out, tuple)
