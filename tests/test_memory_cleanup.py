"""
Tests for memory cleanup behavior in NNsight.

These tests verify that:
1. Deleting the model after a trace clears the memory of the model
2. Objects NOT saved in the trace do not persist after the trace
3. Deleting saved objects frees their memory

These tests are designed to run on CPU only using weakref and gc
to verify object cleanup rather than measuring GPU memory.

NOTE: Many tests are marked with @pytest.mark.xfail to document known memory
retention issues that need to be fixed. When memory cleanup is properly
implemented, these tests will start passing and the xfail marks can be removed.
"""

import gc
import weakref
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


def create_model():
    """Create a simple two-layer model for testing."""
    return torch.nn.Sequential(
        OrderedDict(
            [
                ("layer1", torch.nn.Linear(INPUT_SIZE, HIDDEN_DIMS)),
                ("layer2", torch.nn.Linear(HIDDEN_DIMS, OUTPUT_SIZE)),
            ]
        )
    )


@pytest.fixture
def test_input():
    """Random input tensor for tests."""
    return torch.rand((1, INPUT_SIZE))


def force_gc():
    """Force garbage collection with multiple cycles.

    Python's GC sometimes needs multiple passes to clean up reference cycles.
    This function runs gc.collect() multiple times to ensure thorough cleanup.
    """
    for _ in range(1):
        gc.collect()


# =============================================================================
# Model Cleanup Tests
# =============================================================================


class TestModelCleanup:
    """Tests for model memory cleanup after tracing."""

    def test_model_freed_after_trace_and_delete(self, test_input: torch.Tensor):
        """Test that the underlying model is freed when the NNsight wrapper is deleted."""

        def create_and_trace():
            model = create_model()
            model_ref = weakref.ref(model)
            wrapper = NNsight(model)
            with torch.no_grad():
                with wrapper.trace(test_input):
                    _ = wrapper.layer2.output.save()
            return model_ref

        model_ref = create_and_trace()
        force_gc()

        # Model should be freed
        assert model_ref() is None, "Model was not freed after deleting wrapper"

    def test_model_freed_without_trace(self):
        """Test that the model is freed even if no trace was run."""
        model = create_model()
        model_ref = weakref.ref(model)

        wrapper = NNsight(model)

        del wrapper
        del model
        gc.collect()

        assert (
            model_ref() is None
        ), "Model was not freed after deleting wrapper (no trace)"

    def test_layer_freed_after_model_delete(self, test_input: torch.Tensor):
        """Test that individual layers are freed when the model is deleted."""

        def create_and_trace():
            model = create_model()
            layer1_ref = weakref.ref(model.layer1)
            layer2_ref = weakref.ref(model.layer2)
            wrapper = NNsight(model)
            with torch.no_grad():
                with wrapper.trace(test_input):
                    _ = wrapper.layer2.output.save()
            return layer1_ref, layer2_ref

        layer1_ref, layer2_ref = create_and_trace()
        force_gc()

        assert layer1_ref() is None, "Layer1 was not freed after deleting model"
        assert layer2_ref() is None, "Layer2 was not freed after deleting model"

    def test_multiple_traces_no_accumulation(self, test_input: torch.Tensor):
        """Test that running multiple traces doesn't accumulate references."""
        model = create_model()
        wrapper = NNsight(model)

        # Track tensors from each trace
        tensor_refs = []

        for _ in range(5):
            with torch.no_grad():
                with wrapper.trace(test_input):
                    out = wrapper.layer2.output.save()
            tensor_refs.append(weakref.ref(out))
            del out

        gc.collect()

        # All tensors from previous traces should be freed (except possibly the last one)
        freed_count = sum(1 for ref in tensor_refs[:-1] if ref() is None)
        assert (
            freed_count >= len(tensor_refs) - 1
        ), f"Only {freed_count}/{len(tensor_refs)-1} intermediate tensors were freed"


# =============================================================================
# Unsaved Object Cleanup Tests
# =============================================================================


class TestUnsavedObjectCleanup:
    """Tests that objects not saved in traces don't persist."""

    def test_unsaved_intermediate_not_retained(self, test_input: torch.Tensor):
        """Test that intermediate values not saved are not retained after trace."""
        model = create_model()
        wrapper = NNsight(model)

        # We can't get a weakref to the proxy object directly,
        # but we can verify that the interleaver doesn't hold references
        with torch.no_grad():
            with wrapper.trace(test_input):
                # Access but don't save
                _ = wrapper.layer1.output
                _ = wrapper.layer2.output
                # Only save final output
                final = wrapper.output.save()

        # The trace should complete without holding intermediate values
        assert isinstance(final, torch.Tensor)

        # Clean up
        del final
        del wrapper
        del model
        gc.collect()

    def test_unsaved_clones_not_retained(self, test_input: torch.Tensor):
        """Test that cloned tensors not saved are not retained."""
        model = create_model()
        wrapper = NNsight(model)

        with torch.no_grad():
            with wrapper.trace(test_input):
                # Clone but don't save
                _ = wrapper.layer1.output.clone()
                _ = wrapper.layer2.output.clone()
                # Only save final
                final = wrapper.output.save()

        assert isinstance(final, torch.Tensor)

        del final
        del wrapper
        del model
        gc.collect()

    def test_only_saved_objects_are_tensors(self, test_input: torch.Tensor):
        """Test that only explicitly saved objects become real tensors after trace."""
        model = create_model()
        wrapper = NNsight(model)

        unsaved_ref = None  # Store ref to check after trace

        with torch.no_grad():
            with wrapper.trace(test_input):
                unsaved = wrapper.layer1.output
                saved = wrapper.layer2.output.save()
                # Try to store a reference to the unsaved object
                unsaved_ref = unsaved

        # Saved object should be a real tensor
        assert isinstance(saved, torch.Tensor)

        # Unsaved object should NOT be a real tensor (it's a proxy or None)
        assert not isinstance(
            unsaved_ref, torch.Tensor
        ), "Unsaved proxy should not become a tensor after trace"


# =============================================================================
# Saved Object Cleanup Tests
# =============================================================================


class TestSavedObjectCleanup:
    """Tests that saved objects can be properly freed.

    NOTE: These tests document known memory retention issues.
    Even with torch.no_grad(), tensors are not being freed properly.
    """

    def test_saved_tensor_freed_on_delete(self, test_input: torch.Tensor):
        """Test that saved tensors are freed when deleted (no grad)."""

        def create_and_trace():
            model = create_model()
            wrapper = NNsight(model)
            with torch.no_grad():
                with wrapper.trace(test_input):
                    saved = wrapper.layer2.output.save()
            return weakref.ref(saved)

        saved_ref = create_and_trace()
        force_gc()

        assert saved_ref() is None, "Saved tensor was not freed after deletion"

    def test_multiple_saved_tensors_freed_independently(self, test_input: torch.Tensor):
        """Test that multiple saved tensors can be freed independently (no grad)."""

        def create_and_trace():
            model = create_model()
            wrapper = NNsight(model)
            with torch.no_grad():
                with wrapper.trace(test_input):
                    saved1 = wrapper.layer1.output.save()
                    saved2 = wrapper.layer2.output.save()
            return weakref.ref(saved1), weakref.ref(saved2)

        ref1, ref2 = create_and_trace()
        force_gc()

        # Both should be freed after the function scope ends
        assert ref1() is None, "First saved tensor was not freed"
        assert ref2() is None, "Second saved tensor was not freed"

    def test_saved_tensor_in_list_freed(self, test_input: torch.Tensor):
        """Test that saved tensors in a list are freed when list is deleted (no grad)."""

        def create_and_trace():
            model = create_model()
            wrapper = NNsight(model)
            with torch.no_grad():
                with wrapper.trace(test_input):
                    t1 = wrapper.layer1.output.save()
                    t2 = wrapper.layer2.output.save()
            return [weakref.ref(t1), weakref.ref(t2)]

        refs = create_and_trace()
        force_gc()

        for i, ref in enumerate(refs):
            assert ref() is None, f"Tensor {i} in list was not freed"


# =============================================================================
# Interleaver Cleanup Tests
# =============================================================================


class TestInterleaverCleanup:
    """Tests that interleaver resources are properly cleaned up."""

    def test_interleaver_freed_with_wrapper(self, test_input: torch.Tensor):
        """Test that the interleaver is freed when the wrapper is deleted."""

        def create_and_trace():
            model = create_model()
            wrapper = NNsight(model)
            interleaver_ref = weakref.ref(wrapper._interleaver)
            with torch.no_grad():
                with wrapper.trace(test_input):
                    _ = wrapper.output.save()
            return interleaver_ref

        interleaver_ref = create_and_trace()
        force_gc()

        assert interleaver_ref() is None, "Interleaver was not freed with wrapper"


# =============================================================================
# Tracer Cleanup Tests
# =============================================================================


class TestTracerCleanup:
    """Tests that tracer objects are properly cleaned up."""

    def test_tracer_freed_after_trace(self, test_input: torch.Tensor):
        """Test that the tracer object is freed after the trace context exits."""

        def create_and_trace():
            model = create_model()
            wrapper = NNsight(model)
            refs = []  # Define outside with block
            with torch.no_grad():
                with wrapper.trace(test_input) as tracer:
                    refs.append(weakref.ref(tracer))
                    _ = wrapper.output.save()
            return refs[0]

        tracer_ref = create_and_trace()
        force_gc()

        assert tracer_ref() is None, "Tracer was not freed after trace"

    def test_tracer_no_reference_to_saved_tensors(self, test_input: torch.Tensor):
        """Test that the tracer doesn't retain references to saved tensors."""

        def create_and_trace():
            model = create_model()
            wrapper = NNsight(model)
            refs = []  # Define outside with block
            with torch.no_grad():
                with wrapper.trace(test_input) as tracer:
                    saved = wrapper.layer2.output.save()
                    refs.append(weakref.ref(tracer))
                    refs.append(weakref.ref(saved))
            # After trace, tracer should not keep saved tensor alive
            return refs[0], refs[1]

        tracer_ref, saved_ref = create_and_trace()
        force_gc()

        assert tracer_ref() is None, "Tracer was not freed"
        assert saved_ref() is None, "Saved tensor retained by tracer"

    def test_tracer_freed_with_modifications(self, test_input: torch.Tensor):
        """Test that tracer is freed even after making modifications."""

        def create_and_trace():
            model = create_model()
            wrapper = NNsight(model)
            refs = []  # Define outside with block
            with torch.no_grad():
                with wrapper.trace(test_input) as tracer:
                    wrapper.layer1.output[:] = 0
                    _ = wrapper.output.save()
                    refs.append(weakref.ref(tracer))
            return refs[0]

        tracer_ref = create_and_trace()
        force_gc()

        assert tracer_ref() is None, "Tracer with modifications was not freed"

    @pytest.mark.xfail(reason="Tracer not freed after early stop - needs cleanup fix")
    def test_tracer_freed_after_early_stop(self, test_input: torch.Tensor):
        """Test that tracer is freed after early stopping.

        NOTE: This test documents that tracer.stop() causes a reference leak.
        """

        def create_and_trace():
            model = create_model()
            wrapper = NNsight(model)
            refs = []  # Define outside with block
            with torch.no_grad():
                with wrapper.trace(test_input) as tracer:
                    refs.append(weakref.ref(tracer))  # Append before stop
                    _ = wrapper.layer1.output.save()
                    tracer.stop()
            return refs[0]

        tracer_ref = create_and_trace()

        force_gc()

        assert tracer_ref() is None, "Tracer was not freed after early stop"

    def test_multiple_tracers_freed(self, test_input: torch.Tensor):
        """Test that multiple sequential tracers are all freed."""

        def create_and_trace():
            model = create_model()
            wrapper = NNsight(model)
            tracer_refs = []
            with torch.no_grad():
                for _ in range(5):
                    with wrapper.trace(test_input) as tracer:
                        _ = wrapper.output.save()
                        tracer_refs.append(weakref.ref(tracer))
            return tracer_refs

        tracer_refs = create_and_trace()
        force_gc()

        for i, ref in enumerate(tracer_refs):
            assert ref() is None, f"Tracer {i} was not freed"

    def test_nested_session_tracer_freed(self, test_input: torch.Tensor):
        """Test that tracers in sessions are freed."""

        def create_and_trace():
            model = create_model()
            wrapper = NNsight(model)
            tracer_refs = []
            with torch.no_grad():
                with wrapper.session():
                    with wrapper.trace(test_input) as tracer1:
                        _ = wrapper.layer1.output.save()
                        tracer_refs.append(weakref.ref(tracer1))
                    with wrapper.trace(test_input) as tracer2:
                        _ = wrapper.layer2.output.save()
                        tracer_refs.append(weakref.ref(tracer2))
            return tracer_refs

        tracer_refs = create_and_trace()
        force_gc()

        for i, ref in enumerate(tracer_refs):
            assert ref() is None, f"Session tracer {i} was not freed"


# =============================================================================
# Session Cleanup Tests
# =============================================================================


class TestSessionCleanup:
    """Tests for memory cleanup in session contexts."""

    def test_session_values_freed_after_session(self, test_input: torch.Tensor):
        """Test that values created in a session are freed after the session."""

        def create_and_trace():
            model = create_model()
            wrapper = NNsight(model)
            with torch.no_grad():
                with wrapper.session():
                    with wrapper.trace(test_input):
                        saved = wrapper.layer2.output.save()
            return weakref.ref(saved)

        saved_ref = create_and_trace()
        force_gc()

        assert saved_ref() is None, "Session value was not freed after session"

    def test_cross_trace_values_freed(self, test_input: torch.Tensor):
        """Test that cross-trace values are freed after session."""

        def create_and_trace():
            model = create_model()
            wrapper = NNsight(model)
            with torch.no_grad():
                with wrapper.session():
                    with wrapper.trace(test_input):
                        val1 = wrapper.layer1.output.save()
                    with wrapper.trace(test_input):
                        val2 = wrapper.layer2.output.save()
            return weakref.ref(val1), weakref.ref(val2)

        ref1, ref2 = create_and_trace()
        force_gc()

        assert ref1() is None, "First cross-trace value was not freed"
        assert ref2() is None, "Second cross-trace value was not freed"


# =============================================================================
# Stress Tests
# =============================================================================


class TestMemoryStress:
    """Stress tests for memory cleanup."""

    def test_many_traces_no_leak(self, test_input: torch.Tensor):
        """Test that running many traces doesn't leak memory."""
        model = create_model()
        wrapper = NNsight(model)

        # Track all tensor refs
        all_refs = []

        for i in range(50):
            with torch.no_grad():
                with wrapper.trace(test_input):
                    out = wrapper.layer2.output.save()
            all_refs.append(weakref.ref(out))
            del out

        gc.collect()

        # Most tensors should be freed (allow some slack for recent ones)
        freed_count = sum(1 for ref in all_refs if ref() is None)
        assert (
            freed_count >= len(all_refs) - 2
        ), f"Memory leak: only {freed_count}/{len(all_refs)} tensors were freed"

        del wrapper
        del model

    def test_large_tensor_freed(self, test_input: torch.Tensor):
        """Test that large tensors are properly freed (no grad)."""

        def create_and_trace():
            """Create model, trace, and return weakref to saved tensor."""
            large_model = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("layer1", torch.nn.Linear(INPUT_SIZE, 1000)),
                        ("layer2", torch.nn.Linear(1000, 1000)),
                        ("layer3", torch.nn.Linear(1000, OUTPUT_SIZE)),
                    ]
                )
            )
            wrapper = NNsight(large_model)

            with torch.no_grad():
                with wrapper.trace(test_input):
                    large_tensor = wrapper.layer2.output.save()

            return weakref.ref(large_tensor)

        # Get the weakref - all local vars in create_and_trace are now out of scope
        large_ref = create_and_trace()
        force_gc()

        assert large_ref() is None, "Large tensor was not freed"

    def test_repeated_wrapper_creation_no_leak(self, test_input: torch.Tensor):
        """Test that repeatedly creating wrappers doesn't leak memory (no grad)."""

        def create_wrappers_and_trace(model):
            wrapper_refs = []
            for _ in range(10):
                wrapper = NNsight(model)
                wrapper_refs.append(weakref.ref(wrapper))
                with torch.no_grad():
                    with wrapper.trace(test_input):
                        _ = wrapper.output.save()
            return wrapper_refs

        model = create_model()
        model_ref = weakref.ref(model)

        wrapper_refs = create_wrappers_and_trace(model)
        force_gc()

        # All wrappers should be freed
        freed_wrappers = sum(1 for ref in wrapper_refs if ref() is None)
        assert freed_wrappers == len(
            wrapper_refs
        ), f"Only {freed_wrappers}/{len(wrapper_refs)} wrappers were freed"

        # Model should still exist (we still hold reference)
        assert model_ref() is not None

        del model
        force_gc()

        # Now model should be freed
        assert model_ref() is None, "Model was not freed after all wrappers deleted"


# =============================================================================
# Gradient Retention Tests (documenting expected behavior with gradients)
# =============================================================================


class TestGradientRetention:
    """Tests documenting memory behavior when gradients are involved.

    Tensors with grad_fn retain references to the computation graph,
    which can prevent cleanup. These tests verify expected behavior.
    """

    def test_tensor_with_grad_fn_retains_references(self, test_input: torch.Tensor):
        """Document that tensors with grad_fn retain computation graph references."""
        model = create_model()
        wrapper = NNsight(model)

        # With gradients enabled, tensors retain grad_fn
        with wrapper.trace(test_input):
            saved = wrapper.layer2.output.save()

        # Tensor should have grad_fn when not using torch.no_grad()
        assert saved.grad_fn is not None, "Expected tensor to have grad_fn"

        del saved
        del wrapper
        del model

    def test_tensor_without_grad_fn_frees_cleanly(self, test_input: torch.Tensor):
        """Verify that tensors without grad_fn free cleanly."""

        def create_and_trace():
            model = create_model()
            wrapper = NNsight(model)
            with torch.no_grad():
                with wrapper.trace(test_input):
                    saved = wrapper.layer2.output.save()
            # Tensor should NOT have grad_fn with torch.no_grad()
            assert saved.grad_fn is None, "Expected tensor to not have grad_fn"
            return weakref.ref(saved)

        saved_ref = create_and_trace()
        force_gc()

        assert saved_ref() is None, "Tensor without grad_fn should be freed"
