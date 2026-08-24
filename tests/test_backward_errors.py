"""Tests for the diagnostics around a freed autograd graph.

All invokes in a trace contribute to a single batched forward pass, so they
also share a single autograd graph. One `.backward()` per invoke therefore
still frees the graph for every invoke after it -- which surfaced as
autograd's own "backward through the graph a second time" `RuntimeError`, a
message that never mentions invokes and reads as unrelated to the code that
triggered it.
"""

import pytest
import torch

import nnsight

PROMPT_A = "The quick brown fox jumps"
PROMPT_B = "Madison Square Garden is located in"


def _rel_err(got: torch.Tensor, expected: torch.Tensor) -> float:
    """Scale-free gradient comparison; batching left-pads the shorter prompt."""

    length = min(got.shape[1], expected.shape[1])
    got, expected = got[:, -length:], expected[:, -length:]

    return ((got - expected).norm() / expected.norm()).item()


def _single_invoke_grad(gpt2: nnsight.LanguageModel, prompt: str) -> torch.Tensor:
    with gpt2.trace() as tracer:
        with tracer.invoke(prompt):
            x = gpt2.transformer.h[5].attn.c_proj.output
            with gpt2.lm_head.output.sum().backward():
                grad = x.grad.save()
    return grad


@pytest.mark.usefixtures("gpt2")
class TestFreedGraphDiagnostic:
    """The raised error must name the cause and the fix."""

    def test_backward_in_two_invokes_explains_the_shared_graph(
        self, gpt2: nnsight.LanguageModel
    ):
        with pytest.raises(Exception) as info:
            with gpt2.trace() as tracer:
                with tracer.invoke(PROMPT_A):
                    x = gpt2.transformer.h[5].attn.c_proj.output
                    with gpt2.lm_head.output.sum().backward():
                        x.grad.save()
                with tracer.invoke(PROMPT_B):
                    y = gpt2.transformer.h[5].attn.c_proj.output
                    with gpt2.lm_head.output.sum().backward():
                        y.grad.save()

        message = str(info.value)

        # The cause, stated in nnsight's terms rather than autograd's.
        assert "share one autograd graph" in message
        assert "single batched forward pass" in message
        # The fix, and the original error for anyone who needs it.
        assert "retain_graph=True" in message
        assert "Original error from torch.autograd" in message

    def test_message_notes_when_retain_graph_was_already_passed(
        self, gpt2: nnsight.LanguageModel
    ):
        # The failing call already retains; it is the *earlier* one that did not.
        # The message should say so rather than suggesting a flag that is set.
        with pytest.raises(Exception) as info:
            with gpt2.trace() as tracer:
                with tracer.invoke(PROMPT_A):
                    x = gpt2.transformer.h[5].attn.c_proj.output
                    with gpt2.lm_head.output.sum().backward():
                        x.grad.save()
                with tracer.invoke(PROMPT_B):
                    y = gpt2.transformer.h[5].attn.c_proj.output
                    with gpt2.lm_head.output.sum().backward(retain_graph=True):
                        y.grad.save()

        assert "did not" in str(info.value)

    def test_unrelated_runtime_errors_are_not_reworded(
        self, gpt2: nnsight.LanguageModel
    ):
        # Only the freed-graph message is translated; everything else must
        # propagate with its own text intact.
        with pytest.raises(Exception) as info:
            with gpt2.trace(PROMPT_A):
                logits = gpt2.lm_head.output
                with logits.sum().backward(gradient=torch.zeros(3, 3)):
                    pass

        assert "share one autograd graph" not in str(info.value)


@pytest.mark.usefixtures("gpt2")
class TestRetainGraphAcrossInvokes:
    """The documented fix must actually produce each invoke's own gradient."""

    def test_retain_graph_on_all_but_last(self, gpt2: nnsight.LanguageModel):
        reference_a = _single_invoke_grad(gpt2, PROMPT_A)
        reference_b = _single_invoke_grad(gpt2, PROMPT_B)

        with gpt2.trace() as tracer:
            with tracer.invoke(PROMPT_A):
                x = gpt2.transformer.h[5].attn.c_proj.output
                with gpt2.lm_head.output.sum().backward(retain_graph=True):
                    grad_a = x.grad.save()
            with tracer.invoke(PROMPT_B):
                y = gpt2.transformer.h[5].attn.c_proj.output
                with gpt2.lm_head.output.sum().backward():
                    grad_b = y.grad.save()

        assert _rel_err(grad_a, reference_a) < 1e-4
        assert _rel_err(grad_b, reference_b) < 1e-4

        # Each invoke gets its own rows, not the other invoke's.
        assert _rel_err(grad_a, reference_b) > 1e-2
        assert _rel_err(grad_b, reference_a) > 1e-2

    def test_three_invokes(self, gpt2: nnsight.LanguageModel):
        prompts = [PROMPT_A, PROMPT_B, "A completely different sentence appears"]
        references = [_single_invoke_grad(gpt2, prompt) for prompt in prompts]

        grads = []
        with gpt2.trace() as tracer:
            for index, prompt in enumerate(prompts):
                last = index == len(prompts) - 1
                with tracer.invoke(prompt):
                    x = gpt2.transformer.h[5].attn.c_proj.output
                    with gpt2.lm_head.output.sum().backward(retain_graph=not last):
                        grads.append(x.grad.save())

        for index, (grad, reference) in enumerate(zip(grads, references)):
            assert _rel_err(grad, reference) < 1e-4, f"invoke {index}"
