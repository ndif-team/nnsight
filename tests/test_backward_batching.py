"""Tests for backward/gradient support with batched (multi-invoke) traces.

These tests verify that `.grad` access works correctly when multiple input
invokes are batched together.  The core issue is that `Batcher._narrow()`
creates views that are not in the autograd forward path, so backward hooks
on those views never fire.  The fix in `wrap_grad` registers hooks on the
base tensor instead and narrows the gradient in the hook.

A secondary fix in `Batcher._swap` prevents a segfault when a user assigns
a narrow view back into its own base tensor (self-referential autograd).
"""

import pytest
import torch
from nnsight import LanguageModel


@pytest.fixture(scope="module")
def model():
    return LanguageModel(
        "openai-community/gpt2", device_map="cpu", dispatch=True
    )


class TestBackwardWithBatchedInvokes:
    """Gradient access with two input invokes (needs_batching=True)."""

    def test_grad_in_second_invoke(self, model):
        """User's original attribution patching pattern: clean run + corrupted
        run with backward in the second invoke."""
        with model.trace() as tracer:
            with tracer.invoke("Hello"):
                clean = model.transformer.h[-1].output[0].save()
            with tracer.invoke("World"):
                x = model.transformer.h[-1].output[0]
                x.requires_grad_(True)
                logits = model.lm_head.output
                with logits.sum().backward():
                    grad = x.grad.save()

        assert grad.shape[0] == 1
        assert grad.shape[-1] == 768
        assert (grad != 0).any()

    def test_grad_in_first_invoke(self, model):
        """Backward in the first invoke of two."""
        with model.trace() as tracer:
            with tracer.invoke("Hello"):
                x = model.transformer.h[-1].output[0]
                x.requires_grad_(True)
                logits = model.lm_head.output
                with logits.sum().backward():
                    grad = x.grad.save()
            with tracer.invoke("World"):
                clean = model.transformer.h[-1].output[0].save()

        assert grad.shape[0] == 1
        assert (grad != 0).any()

    def test_attribution_patching(self, model):
        """Full attribution patching: logit-diff metric with gradient."""
        paris = model.tokenizer.encode(" Paris")[0]
        rome = model.tokenizer.encode(" Rome")[0]

        with model.trace() as tracer:
            with tracer.invoke("The Eiffel Tower is in"):
                pass
            with tracer.invoke("The Colosseum is in"):
                hs = model.transformer.h[5].output[0]
                hs.requires_grad_(True)
                logits = model.lm_head.output[:, -1]
                metric = logits[0, paris] - logits[0, rome]
                with metric.backward():
                    attr = hs.grad.save()

        assert attr.shape[-1] == 768
        assert (attr != 0).any()

    def test_grad_deterministic(self, model):
        """Same setup run twice produces identical gradients."""
        grads = []
        for _ in range(2):
            with model.trace() as tracer:
                with tracer.invoke("Hello"):
                    pass
                with tracer.invoke("World"):
                    x = model.transformer.h[-1].output[0]
                    x.requires_grad_(True)
                    logits = model.lm_head.output
                    with logits.sum().backward():
                        grads.append(x.grad.save())

        assert torch.allclose(grads[0], grads[1])

    def test_grad_modification(self, model):
        """User can read and modify gradients inside backward context."""
        with model.trace() as tracer:
            with tracer.invoke("Hello"):
                pass
            with tracer.invoke("World"):
                x = model.transformer.h[-1].output[0]
                x.requires_grad_(True)
                logits = model.lm_head.output
                with logits.sum().backward():
                    orig = x.grad.clone().save()
                    x.grad[:] = 0
                    zeroed = x.grad.save()

        assert (orig != 0).any()
        assert (zeroed == 0).all()


class TestBackwardRegressions:
    """Ensure existing single-invoke and empty-invoke patterns still work."""

    def test_single_invoke(self, model):
        with model.trace("Hello"):
            x = model.transformer.h[-1].output[0]
            x.requires_grad_(True)
            logits = model.lm_head.output
            with logits.sum().backward():
                grad = x.grad.save()

        assert (grad != 0).any()

    def test_input_plus_empty_invoke(self, model):
        with model.trace() as tracer:
            with tracer.invoke("Hello"):
                x = model.transformer.h[-1].output[0]
                x.requires_grad_(True)
                logits = model.lm_head.output
                with logits.sum().backward():
                    grad = x.grad.save()
            with tracer.invoke():
                out = model.transformer.h[-1].output[0].save()

        assert (grad != 0).any()

    def test_retain_graph(self, model):
        with model.trace("Hello"):
            x = model.transformer.h[-1].output[0]
            x.requires_grad_(True)
            logits = model.lm_head.output
            with logits.sum().backward(retain_graph=True):
                g1 = x.grad.save()
            modified = logits * 2
            with modified.sum().backward():
                g2 = x.grad.save()

        assert torch.allclose(g2, g1 * 2, atol=1e-5)


class TestSwapSelfReferentialGuard:
    """Batcher._swap must use concat when swap_value is a view of
    current_value, otherwise in-place assignment segfaults during backward."""

    def test_pop_back_pattern(self, model):
        """User reads output, sets requires_grad, assigns back — was segfault."""
        with model.trace() as tracer:
            with tracer.invoke("Hello"):
                clean = model.transformer.h[-1].output[0].save()
            with tracer.invoke("World"):
                out = model.transformer.h[-1].output
                x = out[0]
                x.requires_grad_(True)
                model.transformer.h[-1].output = (x,) + out[1:]
                logits = model.lm_head.output
                with logits.sum().backward():
                    grad = x.grad.save()

        assert (grad != 0).any()

    def test_normal_swap_unaffected(self, model):
        """Swapping a new tensor (not a view) still uses in-place path."""
        with model.trace() as tracer:
            with tracer.invoke("Hello"):
                pass
            with tracer.invoke("World"):
                out = model.transformer.h[-1].output
                modified = out[0] * 2
                model.transformer.h[-1].output = (modified,) + out[1:]
                logits = model.lm_head.output.save()

        assert logits.shape[-1] == 50257
