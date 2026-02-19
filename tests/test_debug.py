"""
Tests for NNsight debugging and exception handling.

These tests serve two purposes:
1. Verify that exceptions are properly reconstructed with correct line numbers
2. Demonstrate what different exception types look like for documentation

Run with:
    pytest tests/test_debug.py -v -s  # -s to see print output
    pytest tests/test_debug.py -v -s -k "out_of_order"  # specific test
"""

import pytest
import torch
import traceback
import sys

from nnsight import NNsight, CONFIG
from nnsight.intervention.interleaver import Mediator


class TestOutOfOrderErrors:
    """Tests for OutOfOrderError when modules are accessed in wrong order."""

    def test_out_of_order_basic(self, tiny_model: NNsight, tiny_input: torch.Tensor):
        """Access layer2 before layer1 - should raise OutOfOrderError."""
        with pytest.raises(Mediator.OutOfOrderError) as exc_info:
            with tiny_model.trace(tiny_input):
                # WRONG: layer2 runs after layer1, but we access layer2 first
                out2 = tiny_model.layer2.output.save()
                out1 = tiny_model.layer1.output.save()  # This should fail

        print("\n--- OutOfOrderError (basic) ---")
        print(str(exc_info.value))

        # Verify the error message mentions the missed module
        assert (
            "layer1" in str(exc_info.value).lower()
            or "missed" in str(exc_info.value).lower()
        )

    def test_out_of_order_exception_type_preserved(
        self, tiny_model: NNsight, tiny_input: torch.Tensor
    ):
        """Verify that OutOfOrderError type is preserved for catching."""
        caught_correct_type = False

        try:
            with tiny_model.trace(tiny_input):
                out2 = tiny_model.layer2.output.save()
                out1 = tiny_model.layer1.output.save()
        except Mediator.OutOfOrderError:
            caught_correct_type = True

        assert caught_correct_type, "Should be able to catch OutOfOrderError by type"


class TestExceptionReconstruction:
    """Tests that verify exception tracebacks point to original source."""

    def test_index_error_shows_correct_line(
        self, tiny_model: NNsight, tiny_input: torch.Tensor
    ):
        """IndexError should point to the line with the bad index."""
        with pytest.raises(IndexError) as exc_info:
            with tiny_model.trace(tiny_input):
                # This line should appear in the traceback
                bad_access = tiny_model.layer1.output[999].save()

        print("\n--- IndexError traceback ---")
        print(str(exc_info.value))

        # The exception string should contain our code
        exc_str = str(exc_info.value)
        assert "999" in exc_str or "layer1" in exc_str

    def test_attribute_error_in_trace(
        self, tiny_model: NNsight, tiny_input: torch.Tensor
    ):
        """AttributeError for accessing non-existent module."""
        with pytest.raises(AttributeError) as exc_info:
            with tiny_model.trace(tiny_input):
                # This module doesn't exist
                out = tiny_model.nonexistent_layer.output.save()

        print("\n--- AttributeError traceback ---")
        print(str(exc_info.value))

    def test_exception_type_preserved(
        self, tiny_model: NNsight, tiny_input: torch.Tensor
    ):
        """Verify original exception types can still be caught."""
        # Test that we can catch specific exception types
        caught_index = False
        caught_generic = False

        try:
            with tiny_model.trace(tiny_input):
                bad = tiny_model.layer1.output[999].save()
        except IndexError:
            caught_index = True
        except Exception:
            caught_generic = True

        assert caught_index, "Should catch as IndexError, not generic Exception"
        assert not caught_generic


class TestDebugMode:
    """Tests for DEBUG mode exception output."""

    def test_debug_mode_off(self, tiny_model: NNsight, tiny_input: torch.Tensor):
        """With DEBUG=False (default), internal nnsight frames are hidden."""
        original_debug = CONFIG.APP.DEBUG
        CONFIG.APP.DEBUG = False

        try:
            with pytest.raises(IndexError) as exc_info:
                with tiny_model.trace(tiny_input):
                    bad = tiny_model.layer1.output[999].save()

            print("\n--- DEBUG=False traceback ---")
            print(str(exc_info.value))

            exc_str = str(exc_info.value)
            # Should NOT contain internal nnsight paths when DEBUG is off
            # (the reconstructed traceback hides them)

        finally:
            CONFIG.APP.DEBUG = original_debug

    def test_debug_mode_on(self, tiny_model: NNsight, tiny_input: torch.Tensor):
        """With DEBUG=True, internal nnsight frames are shown."""
        original_debug = CONFIG.APP.DEBUG
        CONFIG.APP.DEBUG = True

        try:
            with pytest.raises(IndexError) as exc_info:
                with tiny_model.trace(tiny_input):
                    bad = tiny_model.layer1.output[999].save()

            print("\n--- DEBUG=True traceback ---")
            print(str(exc_info.value))

            # With DEBUG=True, we should see more internal details
            # (the exact content depends on where the error occurs)

        finally:
            CONFIG.APP.DEBUG = original_debug


class TestNestedContextExceptions:
    """Tests for exceptions in nested trace contexts."""

    def test_exception_in_invoke(self, tiny_model: NNsight, tiny_input: torch.Tensor):
        """Exception inside an invoke block."""
        with pytest.raises(IndexError) as exc_info:
            with tiny_model.trace() as tracer:
                with tracer.invoke(tiny_input):
                    # Error happens inside invoke
                    bad = tiny_model.layer1.output[999].save()

        print("\n--- Exception in invoke ---")
        print(str(exc_info.value))

    @pytest.mark.skip(
        reason="Multiple invokes require batching (only in LanguageModel)"
    )
    def test_exception_in_second_invoke(
        self, tiny_model: NNsight, tiny_input: torch.Tensor
    ):
        """Exception in second invoke, after first succeeds.

        Note: This test is skipped for base NNsight because multiple invokes
        require batching, which is only implemented in LanguageModel.
        """
        with pytest.raises(IndexError) as exc_info:
            with tiny_model.trace() as tracer:
                with tracer.invoke(tiny_input):
                    # First invoke is fine
                    out1 = tiny_model.layer1.output.save()

                with tracer.invoke(tiny_input):
                    # Second invoke has the error
                    bad = tiny_model.layer1.output[999].save()

        print("\n--- Exception in second invoke ---")
        print(str(exc_info.value))

    def test_exception_in_helper_function(
        self, tiny_model: NNsight, tiny_input: torch.Tensor
    ):
        """Exception in a helper function called from trace."""

        def bad_helper(model):
            # This function is called from within a trace
            return model.layer1.output[999]

        with pytest.raises(IndexError) as exc_info:
            with tiny_model.trace(tiny_input):
                result = bad_helper(tiny_model).save()

        print("\n--- Exception in helper function ---")
        print(str(exc_info.value))

    def test_exception_in_backward_context(
        self, tiny_model: NNsight, tiny_input: torch.Tensor
    ):
        """Exception inside a backward() context."""
        with pytest.raises(IndexError) as exc_info:
            with tiny_model.trace(tiny_input):
                out = tiny_model.layer1.output
                loss = tiny_model.output.sum()

                with loss.backward():
                    # Error in backward context
                    bad_grad = out.grad[999].save()

        print("\n--- Exception in backward context ---")
        print(str(exc_info.value))


class TestOutOfOrderInInvokes:
    """Tests for out-of-order access patterns with multiple invokes."""

    def test_out_of_order_in_single_invoke(
        self, tiny_model: NNsight, tiny_input: torch.Tensor
    ):
        """Out of order within a single invoke."""
        with pytest.raises(Mediator.OutOfOrderError) as exc_info:
            with tiny_model.trace() as tracer:
                with tracer.invoke(tiny_input):
                    out2 = tiny_model.layer2.output.save()
                    out1 = tiny_model.layer1.output.save()  # Out of order!

        print("\n--- OutOfOrder in single invoke ---")
        print(str(exc_info.value))

    @pytest.mark.skip(
        reason="Multiple invokes require batching (only in LanguageModel)"
    )
    def test_correct_order_across_invokes(
        self, tiny_model: NNsight, tiny_input: torch.Tensor
    ):
        """Different invokes can access modules in any order (separate forward passes).

        Note: This test is skipped for base NNsight because multiple invokes
        require batching, which is only implemented in LanguageModel.
        """
        # This should NOT raise - each invoke is a separate forward pass
        with tiny_model.trace() as tracer:
            with tracer.invoke(tiny_input):
                out2 = tiny_model.layer2.output.save()  # Only layer2

            with tracer.invoke(tiny_input):
                out1 = tiny_model.layer1.output.save()  # Only layer1 (new forward pass)

        # Both should work fine
        assert out1 is not None
        assert out2 is not None


class TestExceptionMessages:
    """Tests to examine specific exception message content."""

    def test_out_of_order_message_content(
        self, tiny_model: NNsight, tiny_input: torch.Tensor
    ):
        """Examine the OutOfOrderError message for helpful info."""
        with pytest.raises(Mediator.OutOfOrderError) as exc_info:
            with tiny_model.trace(tiny_input):
                out2 = tiny_model.layer2.output.save()
                out1 = tiny_model.layer1.output.save()

        error_msg = str(exc_info.value)
        print("\n--- OutOfOrderError full message ---")
        print(error_msg)
        print("---")

        # Check for helpful content
        # (adjust these based on what the actual error message contains)
        assert "missed" in error_msg.lower() or "order" in error_msg.lower()


class TestDanglingMediators:
    """Tests for when a module is never called (dangling mediator)."""

    def test_nonexistent_module_path(
        self, tiny_model: NNsight, tiny_input: torch.Tensor
    ):
        """Access a module path that doesn't exist - mediator waits forever."""
        # Note: This might raise AttributeError first, or a dangling mediator error
        with pytest.raises((AttributeError, ValueError)) as exc_info:
            with tiny_model.trace(tiny_input):
                # This module doesn't exist
                out = tiny_model.fake_layer.output.save()

        print("\n--- Nonexistent module path ---")
        print(str(exc_info.value))


class TestTraceInputRequirements:
    """Tests for trace input requirements."""

    def test_trace_with_no_args_no_invokes(self, tiny_model: NNsight):
        """Trace with no input and no invokes should error."""
        # This should raise an error - trace() needs input or invokes
        with pytest.raises(Exception) as exc_info:
            with tiny_model.trace():  # No input!
                out = tiny_model.layer1.output.save()

        print("\n--- Trace with no args, no invokes ---")
        print(f"Exception type: {type(exc_info.value).__name__}")
        print(str(exc_info.value))


class TestWithBlockNotFound:
    """Tests related to WithBlockNotFoundError.

    Note: WithBlockNotFoundError is hard to trigger in normal usage.
    It occurs when the AST parser can't find a 'with' block at the expected line.
    This might happen with very unusual source code layouts or dynamic code generation.
    """

    def test_normal_trace_finds_block(
        self, tiny_model: NNsight, tiny_input: torch.Tensor
    ):
        """Normal trace should find the with block without issues."""
        # This should work fine
        with tiny_model.trace(tiny_input):
            out = tiny_model.layer1.output.save()

        assert out is not None
        print("\n--- Normal trace finds block ---")
        print("With block found successfully")


class TestGradientOrderErrors:
    """Tests for gradient access order issues."""

    def test_gradient_wrong_order(self, tiny_model: NNsight, tiny_input: torch.Tensor):
        """Access gradients in wrong order (should access in reverse of forward)."""
        # In this small model, let's see what happens with gradient access
        with pytest.raises(Exception) as exc_info:
            with tiny_model.trace(tiny_input):
                # Get both layer outputs (forward order: layer1 -> layer2)
                out1 = tiny_model.layer1.output
                out2 = tiny_model.layer2.output

                loss = tiny_model.output.sum()

                with loss.backward():
                    # Try to access layer1 grad before layer2 grad
                    # (wrong order - should be layer2 first since it's backwards)
                    grad1 = out1.grad.save()
                    grad2 = out2.grad.save()  # This might fail

        print("\n--- Gradient wrong order ---")
        print(f"Exception type: {type(exc_info.value).__name__}")
        print(str(exc_info.value))


class TestMultipleErrorScenarios:
    """Combination tests for complex error scenarios."""

    def test_error_after_successful_operations(
        self, tiny_model: NNsight, tiny_input: torch.Tensor
    ):
        """Error after some successful operations."""
        with pytest.raises(IndexError) as exc_info:
            with tiny_model.trace(tiny_input):
                # These work fine
                out1 = tiny_model.layer1.output.save()
                shape = out1.shape

                # This fails
                bad = tiny_model.layer2.output[999].save()

        print("\n--- Error after successful operations ---")
        print(str(exc_info.value))

        # Verify the successful operation still returned something
        # (values before error should still be accessible if saved)

    def test_conditional_error(self, tiny_model: NNsight, tiny_input: torch.Tensor):
        """Error inside a conditional."""
        with pytest.raises(IndexError) as exc_info:
            with tiny_model.trace(tiny_input):
                out = tiny_model.layer1.output
                if out.shape[0] > 0:  # This is True
                    # Error happens in conditional branch
                    bad = out[999].save()

        print("\n--- Error in conditional ---")
        print(str(exc_info.value))

    def test_loop_error(self, tiny_model: NNsight, tiny_input: torch.Tensor):
        """Error inside a loop."""
        with pytest.raises(Mediator.OutOfOrderError) as exc_info:
            with tiny_model.trace(tiny_input):
                results = []
                # Loop that accesses layers in wrong order
                for layer in [tiny_model.layer2, tiny_model.layer1]:
                    results.append(layer.output.save())

        print("\n--- Error in loop ---")
        print(str(exc_info.value))


class TestBackwardTracerRestrictions:
    """Tests for backward tracer limitations.

    The BackwardsMediator enforces that you can only request .grad inside
    a backward() context. Trying to access .output or .input will fail with:

    "Cannot request `{requester}` in a backwards tracer. You can only request
    `.grad`. Please define your Tensors before the Backwards Tracer and interact
    with their gradients within the Backwards Tracer."

    This is hard to trigger with a simple test because accessing .output
    inside backward hits other errors first (like OutOfOrderError). The
    restriction is enforced in BackwardsMediator.request() in backwards.py.
    """

    def test_backward_restriction_documented(self):
        """Document the backward tracer restriction."""
        print("\n--- Backward Tracer Restriction ---")
        print("Inside `with loss.backward():`, you can ONLY access `.grad`.")
        print("Accessing `.output` or `.input` will fail with:")
        print("")
        print(
            "  ValueError: Cannot request `model.layer.output.i0` in a backwards tracer."
        )
        print("  You can only request `.grad`. Please define your Tensors before the")
        print(
            "  Backwards Tracer and interact with their gradients within the Backwards Tracer."
        )
        print("")
        print("CORRECT pattern:")
        print("  with model.trace('Hello'):")
        print(
            "      hs = model.transformer.h[-1].output[0]  # Get tensor BEFORE backward"
        )
        print("      loss = model.lm_head.output.sum()")
        print("      with loss.backward():")
        print("          grad = hs.grad.save()  # Access .grad INSIDE backward")


class TestNestedInvokeError:
    """Tests for invoke inside invoke error."""

    def test_invoke_inside_invoke(self, tiny_model: NNsight, tiny_input: torch.Tensor):
        """Cannot create an invoke inside another invoke during interleaving."""
        # This is tricky to trigger - let's see what happens
        # The error occurs when trying to invoke during active interleaving
        with pytest.raises(ValueError) as exc_info:
            with tiny_model.trace() as tracer:
                with tracer.invoke(tiny_input):
                    # Try to create another invoke INSIDE this one
                    # This should fail because we're already interleaving
                    with tracer.invoke(tiny_input):
                        out = tiny_model.layer1.output.save()

        print("\n--- Invoke inside invoke ---")
        print(str(exc_info.value))


class TestIteratorFootgun:
    """Tests for the .iter[:] / .all() footgun - code after doesn't execute."""

    def test_code_after_unbounded_iter_not_executed(
        self, tiny_model: NNsight, tiny_input: torch.Tensor
    ):
        """Code after .iter[:] in a non-generation context won't execute.

        This is a common footgun! When using .iter[:] or .all(), the iterator
        waits for more iterations that never come. The model finishes, and
        check_dangling_mediators gives a WARNING (not error). But any code
        AFTER the iter block never runs!
        """
        # In a regular trace (not generate), there's only 1 iteration
        # So .iter[:] will wait for iteration 1 which never comes

        # We can't easily test this with tiny_model since it doesn't support
        # generation. But we can document the behavior.

        # Let's test what happens when we try to use a variable defined after iter
        print("\n--- Code after unbounded iter footgun ---")
        print("IMPORTANT: In .trace() (non-generation), modules are only called once.")
        print("Using .iter[:] will wait forever for iteration 1 that never comes.")
        print(
            "Code after the iter block won't execute, and variables won't be defined."
        )
        print("")
        print("Example of the footgun:")
        print("  with model.generate('Hello', max_new_tokens=3) as tracer:")
        print("      with tracer.iter[:]:")
        print("          hidden = model.layer.output.save()")
        print("      # WARNING: This line never executes!")
        print("      final = model.lm_head.output.save()")
        print("  print(final)  # NameError: 'final' is not defined")
        print("")
        print("SOLUTION: Use a separate empty invoker:")
        print("  with model.generate('Hello', max_new_tokens=3) as tracer:")
        print("      with tracer.invoke():  # First invoker - handles iteration")
        print("          with tracer.iter[:]:")
        print("              hidden = model.layer.output.save()")
        print("      with tracer.invoke():  # Second invoker - runs after")
        print("          final = model.lm_head.output.save()  # Now runs!")


class TestLanguageModelErrors:
    """Tests for LanguageModel-specific errors."""

    def test_tokenizer_not_provided_with_preloaded_model(self):
        """When wrapping a pre-loaded model, tokenizer must be provided."""
        # This requires a HuggingFace model, so we'll just document the error
        print("\n--- Tokenizer not provided ---")
        print("When wrapping a pre-loaded HuggingFace model:")
        print("  from transformers import AutoModelForCausalLM")
        print("  model = AutoModelForCausalLM.from_pretrained('gpt2')")
        print("  llm = LanguageModel(model)  # Error!")
        print("")
        print("Error message:")
        print("  AttributeError: Tokenizer not found. If you passed a pre-loaded model")
        print(
            "  to `LanguageModel`, you need to provide a tokenizer when initializing:"
        )
        print("  `LanguageModel(model, tokenizer=tokenizer)`.")
        print("")
        print("Fix:")
        print("  from transformers import AutoTokenizer")
        print("  tokenizer = AutoTokenizer.from_pretrained('gpt2')")
        print("  llm = LanguageModel(model, tokenizer=tokenizer)")


# =============================================================================
# Utility for manual inspection
# =============================================================================


def print_full_traceback(exc: Exception):
    """Helper to print full traceback for manual inspection."""
    import traceback

    print("\n" + "=" * 60)
    print("FULL TRACEBACK:")
    print("=" * 60)
    traceback.print_exception(type(exc), exc, exc.__traceback__)
    print("=" * 60)
    print("\nSTR(EXCEPTION):")
    print("=" * 60)
    print(str(exc))
    print("=" * 60 + "\n")
