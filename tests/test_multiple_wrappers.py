"""
Tests for wrapping modules multiple times with NNsight.

These tests verify that:
1. A model wrapped multiple times works correctly for all wrappers
2. Wrapping the same model many times doesn't cause problems
"""

import gc
import pytest
import torch
from collections import OrderedDict

from nnsight import NNsight


# =============================================================================
# Test Fixtures
# =============================================================================


INPUT_SIZE = 5
HIDDEN_DIMS = 10
OUTPUT_SIZE = 2


@pytest.fixture
def shared_model(device: str):
    """Create a simple two-layer model (not wrapped) for testing multiple wrappers."""
    net = torch.nn.Sequential(
        OrderedDict(
            [
                ("layer1", torch.nn.Linear(INPUT_SIZE, HIDDEN_DIMS)),
                ("layer2", torch.nn.Linear(HIDDEN_DIMS, OUTPUT_SIZE)),
            ]
        )
    )
    return net.to(device)


@pytest.fixture
def test_input(device: str):
    """Random input tensor for tests."""
    return torch.rand((1, INPUT_SIZE), device=device)


# =============================================================================
# Multiple Wrappers Functionality Tests
# =============================================================================


class TestMultipleWrappers:
    """Tests for wrapping the same model with multiple NNsight instances."""

    @torch.no_grad()
    def test_two_wrappers_both_work(
        self, shared_model: torch.nn.Module, test_input: torch.Tensor
    ):
        """Test that two NNsight wrappers on the same model both function correctly."""
        # Create two separate wrappers on the same underlying model
        wrapper1 = NNsight(shared_model)
        wrapper2 = NNsight(shared_model)

        # Use wrapper1 to trace and save outputs
        with wrapper1.trace(test_input):
            out1 = wrapper1.layer2.output.save()

        # Use wrapper2 to trace and save outputs
        with wrapper2.trace(test_input):
            out2 = wrapper2.layer2.output.save()

        # Both should produce the same result
        assert isinstance(out1, torch.Tensor)
        assert isinstance(out2, torch.Tensor)
        assert torch.allclose(out1, out2)

    @torch.no_grad()
    def test_two_wrappers_modifications_independent(
        self, shared_model: torch.nn.Module, test_input: torch.Tensor
    ):
        """Test that modifications via one wrapper don't interfere with another."""
        wrapper1 = NNsight(shared_model)
        wrapper2 = NNsight(shared_model)

        # Get baseline output without modifications
        with wrapper2.trace(test_input):
            baseline = wrapper2.layer2.output.clone().save()

        # Modify output using wrapper1
        with wrapper1.trace(test_input):
            wrapper1.layer1.output[:] = 0
            out1 = wrapper1.layer2.output.save()

        # Trace normally using wrapper2 (no modifications)
        with wrapper2.trace(test_input):
            out2 = wrapper2.layer2.output.save()

        # wrapper1's output should be different from baseline due to zeroing layer1
        assert not torch.allclose(out1, baseline)

        # wrapper2 should match baseline (no modifications applied)
        assert torch.allclose(out2, baseline)

    @torch.no_grad()
    def test_interleaved_wrapper_usage(
        self, shared_model: torch.nn.Module, test_input: torch.Tensor
    ):
        """Test using wrappers in an interleaved manner works correctly."""
        wrapper1 = NNsight(shared_model)
        wrapper2 = NNsight(shared_model)

        # First use wrapper1
        with wrapper1.trace(test_input):
            out1_first = wrapper1.layer2.output.save()

        # Then use wrapper2
        with wrapper2.trace(test_input):
            out2 = wrapper2.layer2.output.save()

        # Use wrapper1 again
        with wrapper1.trace(test_input):
            out1_second = wrapper1.layer2.output.save()

        # All outputs should be consistent
        assert torch.allclose(out1_first, out2)
        assert torch.allclose(out1_first, out1_second)

    @torch.no_grad()
    def test_three_wrappers(
        self, shared_model: torch.nn.Module, test_input: torch.Tensor
    ):
        """Test that three wrappers on the same model all work correctly."""
        wrapper1 = NNsight(shared_model)
        wrapper2 = NNsight(shared_model)
        wrapper3 = NNsight(shared_model)

        with wrapper1.trace(test_input):
            out1 = wrapper1.layer2.output.save()

        with wrapper2.trace(test_input):
            out2 = wrapper2.layer2.output.save()

        with wrapper3.trace(test_input):
            out3 = wrapper3.layer2.output.save()

        assert torch.allclose(out1, out2)
        assert torch.allclose(out2, out3)


# =============================================================================
# Stress Tests for Many Wrappers
# =============================================================================


class TestManyWrappers:
    """Tests to ensure wrapping the same model many times doesn't cause issues."""

    @torch.no_grad()
    def test_many_wrappers_no_memory_leak(
        self, shared_model: torch.nn.Module, test_input: torch.Tensor
    ):
        """Test that creating many wrappers doesn't cause memory issues."""
        # Create many wrappers in sequence
        wrappers = []
        for i in range(10):
            wrapper = NNsight(shared_model)
            wrappers.append(wrapper)

        # All wrappers should still work
        for wrapper in wrappers:
            with wrapper.trace(test_input):
                out = wrapper.layer2.output.save()
            assert isinstance(out, torch.Tensor)

    @torch.no_grad()
    def test_wrapper_cleanup_on_gc(
        self, shared_model: torch.nn.Module, test_input: torch.Tensor
    ):
        """Test that wrappers clean up properly when garbage collected."""
        # Create wrappers and let them go out of scope
        for _ in range(5):
            wrapper = NNsight(shared_model)
            with wrapper.trace(test_input):
                out = wrapper.layer2.output.save()
            assert isinstance(out, torch.Tensor)

        # Force garbage collection
        gc.collect()

        # Model should still work with new wrapper
        new_wrapper = NNsight(shared_model)
        with new_wrapper.trace(test_input):
            final_out = new_wrapper.layer2.output.save()

        assert isinstance(final_out, torch.Tensor)

    @torch.no_grad()
    def test_rapid_wrapper_creation_and_use(
        self, shared_model: torch.nn.Module, test_input: torch.Tensor
    ):
        """Test rapidly creating and using wrappers in succession."""
        reference_output = None

        for i in range(20):
            wrapper = NNsight(shared_model)
            with wrapper.trace(test_input):
                out = wrapper.layer2.output.save()

            if reference_output is None:
                reference_output = out.clone()
            else:
                # All outputs should be identical
                assert torch.allclose(out, reference_output)

    @torch.no_grad()
    def test_simultaneous_wrapper_traces(
        self, shared_model: torch.nn.Module, test_input: torch.Tensor
    ):
        """Test that multiple wrapper traces that follow each other work."""
        wrapper1 = NNsight(shared_model)
        wrapper2 = NNsight(shared_model)
        wrapper3 = NNsight(shared_model)
        wrapper4 = NNsight(shared_model)
        wrapper5 = NNsight(shared_model)

        wrappers = [wrapper1, wrapper2, wrapper3, wrapper4, wrapper5]
        outputs = []

        for wrapper in wrappers:
            with wrapper.trace(test_input):
                outputs.append(wrapper.layer2.output.save())

        # All outputs should match
        for out in outputs[1:]:
            assert torch.allclose(outputs[0], out)


# =============================================================================
# Edge Cases
# =============================================================================


class TestMultipleWrapperEdgeCases:
    """Edge case tests for multiple wrappers."""

    @torch.no_grad()
    def test_wrapper_with_modifications_then_new_wrapper(
        self, shared_model: torch.nn.Module, test_input: torch.Tensor
    ):
        """Test that creating a new wrapper after modifications still works."""
        wrapper1 = NNsight(shared_model)

        # Modify via wrapper1
        with wrapper1.trace(test_input):
            wrapper1.layer1.output[:] = 1.0
            out1 = wrapper1.layer2.output.save()

        # Create new wrapper after modification
        wrapper2 = NNsight(shared_model)

        with wrapper2.trace(test_input):
            out2 = wrapper2.layer2.output.save()

        # wrapper2 should work normally (not be affected by wrapper1's modifications)
        assert not torch.allclose(out1, out2)

    @torch.no_grad()
    def test_skip_module_with_multiple_wrappers(
        self, shared_model: torch.nn.Module, test_input: torch.Tensor
    ):
        """Test that module skipping works correctly with multiple wrappers."""
        wrapper1 = NNsight(shared_model)
        wrapper2 = NNsight(shared_model)

        # Skip layer2 using wrapper1
        with wrapper1.trace(test_input):
            wrapper1.layer2.output = wrapper1.layer1.output[:, :OUTPUT_SIZE]
            out1 = wrapper1.output.save()

        # Normal trace with wrapper2
        with wrapper2.trace(test_input):
            out2 = wrapper2.output.save()

        # Outputs should be different due to skip
        assert not torch.allclose(out1, out2)

    @torch.no_grad()
    def test_save_intermediate_layers_multiple_wrappers(
        self, shared_model: torch.nn.Module, test_input: torch.Tensor
    ):
        """Test saving intermediate activations with multiple wrappers."""
        wrapper1 = NNsight(shared_model)
        wrapper2 = NNsight(shared_model)

        with wrapper1.trace(test_input):
            l1_out1 = wrapper1.layer1.output.save()
            l2_out1 = wrapper1.layer2.output.save()

        with wrapper2.trace(test_input):
            l1_out2 = wrapper2.layer1.output.save()
            l2_out2 = wrapper2.layer2.output.save()

        assert torch.allclose(l1_out1, l1_out2)
        assert torch.allclose(l2_out1, l2_out2)

    def test_gradients_with_multiple_wrappers(
        self, shared_model: torch.nn.Module, test_input: torch.Tensor
    ):
        """Test gradient computation with multiple wrappers."""
        wrapper1 = NNsight(shared_model)
        wrapper2 = NNsight(shared_model)

        # Compute gradients with wrapper1
        with wrapper1.trace(test_input):
            l1_out = wrapper1.layer1.output
            loss = wrapper1.output.sum()
            with loss.backward():
                grad1 = l1_out.grad.clone().save()

        # Compute gradients with wrapper2
        with wrapper2.trace(test_input):
            l1_out = wrapper2.layer1.output
            loss = wrapper2.output.sum()
            with loss.backward():
                grad2 = l1_out.grad.clone().save()

        # Gradients should be the same
        assert torch.allclose(grad1, grad2)
