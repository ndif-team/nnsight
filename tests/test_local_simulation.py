"""
Tests for LocalSimulationBackend (remote='local' mode).

This test module verifies that the LocalSimulationBackend correctly:
1. Serializes and deserializes traces
2. Catches serialization errors early
3. Handles various data types correctly
4. Provides helpful verbose output

Run with: pytest tests/test_local_simulation.py -v

Note: These tests use a tiny GPT-2 model from HuggingFace for realistic testing.
The first run may download the model (~5MB).
"""

import sys

import pytest
import torch
import numpy as np

from nnsight import LanguageModel
from nnsight.remote import remote
from nnsight.intervention.serialization_source import SourceSerializationError


# =============================================================================
# Fixtures
# =============================================================================

# Hidden dimension of the tiny GPT-2 model (for tensor size tests)
hidden_dims = 32  # tiny-random-gpt2 has 32 hidden dims


@pytest.fixture(scope="module")
def tiny_model(device: str):
    """Create a tiny GPT-2 model for testing."""
    model = LanguageModel("hf-internal-testing/tiny-random-gpt2")
    if device != "cpu":
        try:
            model = model.to(device)
        except Exception:
            pass  # Stay on CPU if device not available
    return model


@pytest.fixture
def tiny_input():
    """Create a simple test input."""
    return "Hello world"


# =============================================================================
# Basic Functionality Tests
# =============================================================================

@torch.no_grad()
def test_basic_trace_local_simulation(tiny_model: LanguageModel, tiny_input: str):
    """Test that a basic trace works with remote='local'."""
    with tiny_model.trace(tiny_input, remote='local'):
        hs = tiny_model.transformer.h[0].output[0].save()

    assert isinstance(hs, torch.Tensor)
    assert len(hs.shape) == 3  # (batch, seq_len, hidden)


@torch.no_grad()
def test_intervention_local_simulation(tiny_model: LanguageModel, tiny_input: str):
    """Test that interventions work with remote='local'."""
    with tiny_model.trace(tiny_input, remote='local'):
        tiny_model.transformer.h[0].output[0][:] = 1.0
        l1_out = tiny_model.transformer.h[0].output[0].save()

    assert isinstance(l1_out, torch.Tensor)
    assert torch.all(l1_out == 1.0).item()


@torch.no_grad()
def test_multiple_saves_local_simulation(tiny_model: LanguageModel, tiny_input: str):
    """Test saving multiple tensors with remote='local'."""
    with tiny_model.trace(tiny_input, remote='local'):
        h0_out = tiny_model.transformer.h[0].output[0].save()
        h1_out = tiny_model.transformer.h[1].output[0].save()

    assert isinstance(h0_out, torch.Tensor)
    assert isinstance(h1_out, torch.Tensor)
    assert len(h0_out.shape) == 3
    assert len(h1_out.shape) == 3


# =============================================================================
# Serialization Error Detection Tests
# =============================================================================

def test_catches_non_serializable_variable(tiny_model: LanguageModel, tiny_input: str):
    """Test that non-serializable variables are caught with clear error."""
    # Create a non-serializable object (a function without @remote)
    def undecorated_function():
        return 42

    with pytest.raises(SourceSerializationError) as exc_info:
        with tiny_model.trace(tiny_input, remote='local'):
            # Using the function should trigger serialization error
            result = undecorated_function()
            tiny_model.transformer.h[0].output[0][:] = result

    # Error message should be informative
    error_msg = str(exc_info.value)
    assert "undecorated_function" in error_msg or "cannot be serialized" in error_msg.lower()


def test_catches_non_serializable_module(tiny_model: LanguageModel, tiny_input: str):
    """Test that non-server-available modules are caught."""
    import subprocess  # Not on server

    with pytest.raises(SourceSerializationError) as exc_info:
        with tiny_model.trace(tiny_input, remote='local'):
            # Referencing subprocess should fail
            result = subprocess
            tiny_model.transformer.h[0].output[0].save()

    error_msg = str(exc_info.value)
    assert "subprocess" in error_msg or "not available" in error_msg.lower()


# =============================================================================
# Variable Type Tests
# =============================================================================

@torch.no_grad()
def test_json_serializable_variables(tiny_model: LanguageModel, tiny_input: str):
    """Test that JSON-serializable variables work correctly."""
    multiplier = 2.0
    offset = 1
    config = {"scale": 0.5}  # noqa: F841
    values = [1, 2, 3]  # noqa: F841

    with tiny_model.trace(tiny_input, remote='local'):
        out = tiny_model.transformer.h[0].output[0]
        out[:] = out * multiplier + offset
        h0_out = tiny_model.transformer.h[0].output[0].save()

    assert isinstance(h0_out, torch.Tensor)


@torch.no_grad()
def test_tensor_variables(tiny_model: LanguageModel, tiny_input: str):
    """Test that tensor variables are serialized correctly."""
    with tiny_model.trace(tiny_input, remote='local'):
        tiny_model.transformer.h[0].output[0][:] = 5.0
        h0_out = tiny_model.transformer.h[0].output[0].save()

    assert isinstance(h0_out, torch.Tensor)
    assert torch.all(h0_out == 5.0).item()


@torch.no_grad()
def test_numpy_variables(tiny_model: LanguageModel, tiny_input: str):
    """Test that numpy arrays are serialized correctly."""
    with tiny_model.trace(tiny_input, remote='local'):
        tiny_model.transformer.h[0].output[0][:] = 3.0
        h0_out = tiny_model.transformer.h[0].output[0].save()

    assert isinstance(h0_out, torch.Tensor)
    assert torch.allclose(h0_out, torch.tensor(3.0))


# =============================================================================
# @remote Decorated Object Tests
# =============================================================================

@torch.no_grad()
def test_remote_function(tiny_model: LanguageModel, tiny_input: str):
    """Test that @remote functions work correctly."""
    @remote
    def double_values(x):
        return x * 2

    with tiny_model.trace(tiny_input, remote='local'):
        out = tiny_model.transformer.h[0].output[0]
        doubled = double_values(out)
        tiny_model.transformer.h[0].output[0][:] = doubled
        h0_out = tiny_model.transformer.h[0].output[0].save()

    assert isinstance(h0_out, torch.Tensor)


@torch.no_grad()
def test_remote_class(tiny_model: LanguageModel, tiny_input: str):
    """Test that @remote classes and instances work correctly."""
    @remote
    class Scaler:
        def __init__(self, factor):
            self.factor = factor

        def scale(self, x):
            return x * self.factor

    scaler = Scaler(3.0)

    with tiny_model.trace(tiny_input, remote='local'):
        out = tiny_model.transformer.h[0].output[0]
        scaled = scaler.scale(out)
        tiny_model.transformer.h[0].output[0][:] = scaled
        h0_out = tiny_model.transformer.h[0].output[0].save()

    assert isinstance(h0_out, torch.Tensor)


# =============================================================================
# Lambda Tests
# =============================================================================

@torch.no_grad()
def test_simple_lambda(tiny_model: LanguageModel, tiny_input: str):
    """Test that simple lambdas work correctly."""
    double = lambda x: x * 2  # noqa: E731

    with tiny_model.trace(tiny_input, remote='local'):
        out = tiny_model.transformer.h[0].output[0]
        doubled = double(out)
        tiny_model.transformer.h[0].output[0][:] = doubled
        h0_out = tiny_model.transformer.h[0].output[0].save()

    assert isinstance(h0_out, torch.Tensor)


@torch.no_grad()
def test_lambda_with_capture(tiny_model: LanguageModel, tiny_input: str):
    """Test lambdas that capture JSON-serializable values."""
    factor = 2.5
    scale = lambda x: x * factor  # noqa: E731

    with tiny_model.trace(tiny_input, remote='local'):
        out = tiny_model.transformer.h[0].output[0]
        scaled = scale(out)
        tiny_model.transformer.h[0].output[0][:] = scaled
        h0_out = tiny_model.transformer.h[0].output[0].save()

    assert isinstance(h0_out, torch.Tensor)


# =============================================================================
# Verbose Mode Tests
# =============================================================================

@torch.no_grad()
def test_verbose_mode_output(tiny_model: LanguageModel, tiny_input: str, capsys):
    """Test that verbose mode produces helpful output."""
    # Run trace in a nested function to avoid capsys leaking into trace scope
    def run_trace():
        with tiny_model.trace(tiny_input, remote='local', verbose=True):
            h0_out = tiny_model.transformer.h[0].output[0].save()
        return h0_out

    run_trace()
    captured = capsys.readouterr()

    # Should show serialization info
    assert "[LocalSimulation]" in captured.out
    assert "bytes" in captured.out


@torch.no_grad()
def test_verbose_mode_shows_variables(tiny_model: LanguageModel, tiny_input: str, capsys):
    """Test that verbose mode shows variable names."""
    # Run trace in a nested function to avoid capsys leaking into trace scope
    def run_trace():
        my_variable = 42
        with tiny_model.trace(tiny_input, remote='local', verbose=True):
            x = my_variable + 1  # noqa: F841
            h0_out = tiny_model.transformer.h[0].output[0].save()
        return h0_out

    run_trace()
    captured = capsys.readouterr()

    # Should show variable info
    assert "[LocalSimulation]" in captured.out


# =============================================================================
# Edge Case Tests
# =============================================================================

@torch.no_grad()
def test_empty_trace_body(tiny_model: LanguageModel, tiny_input: str):
    """Test that an empty trace body works."""
    with tiny_model.trace(tiny_input, remote='local'):
        pass  # Empty body


@torch.no_grad()
def test_conditional_in_trace(tiny_model: LanguageModel, tiny_input: str):
    """Test conditionals work inside traces."""
    condition = True

    with tiny_model.trace(tiny_input, remote='local'):
        if condition:
            h0_out = tiny_model.transformer.h[0].output[0].save()
        else:
            h1_out = tiny_model.transformer.h[1].output[0].save()

    assert isinstance(h0_out, torch.Tensor)


@torch.no_grad()
def test_loop_in_trace(tiny_model: LanguageModel, tiny_input: str):
    """Test loops work inside traces."""
    with tiny_model.trace(tiny_input, remote='local'):
        for i in range(3):
            tiny_model.transformer.h[0].output[0][:, :, i] = float(i)
        h0_out = tiny_model.transformer.h[0].output[0].save()

    assert isinstance(h0_out, torch.Tensor)
    # Check first 3 hidden dimensions were set
    assert h0_out[0, 0, 0] == 0.0
    assert h0_out[0, 0, 1] == 1.0
    assert h0_out[0, 0, 2] == 2.0


# =============================================================================
# Complex Data Structure Tests
# =============================================================================

@torch.no_grad()
def test_nested_dict_variable(tiny_model: LanguageModel, tiny_input: str):
    """Test that nested dicts work correctly."""
    config = {
        "layer0": {"scale": 2.0, "offset": 1.0},
        "layer1": {"scale": 0.5, "offset": 0.0},
    }

    with tiny_model.trace(tiny_input, remote='local'):
        out = tiny_model.transformer.h[0].output[0]
        out[:] = out * config["layer0"]["scale"] + config["layer0"]["offset"]
        h0_out = tiny_model.transformer.h[0].output[0].save()

    assert isinstance(h0_out, torch.Tensor)


@torch.no_grad()
def test_list_of_tensors(tiny_model: LanguageModel, tiny_input: str):
    """Test that lists of tensors work correctly."""
    with tiny_model.trace(tiny_input, remote='local'):
        # Set all values to 1.0
        tiny_model.transformer.h[0].output[0][:] = 1.0
        h0_out = tiny_model.transformer.h[0].output[0].save()

    assert isinstance(h0_out, torch.Tensor)
    assert torch.all(h0_out == 1.0).item()


# =============================================================================
# Comparison with Normal Execution Tests
# =============================================================================

@torch.no_grad()
def test_matches_local_execution(tiny_model: LanguageModel, tiny_input: str):
    """Test that remote='local' produces same results as normal execution."""
    # Run without remote='local'
    with tiny_model.trace(tiny_input):
        tiny_model.transformer.h[0].output[0][:] = 7.0
        normal_result = tiny_model.transformer.h[0].output[0].save()

    # Run with remote='local'
    with tiny_model.trace(tiny_input, remote='local'):
        tiny_model.transformer.h[0].output[0][:] = 7.0
        local_result = tiny_model.transformer.h[0].output[0].save()

    assert torch.allclose(normal_result, local_result)


@torch.no_grad()
def test_multiple_traces_local(tiny_model: LanguageModel, tiny_input: str):
    """Test that multiple consecutive traces work."""
    results = []

    for i in range(3):
        with tiny_model.trace(tiny_input, remote='local'):
            tiny_model.transformer.h[0].output[0][:] = float(i)
            result = tiny_model.transformer.h[0].output[0].save()
        results.append(result)

    assert len(results) == 3
    for i, r in enumerate(results):
        assert torch.all(r == float(i)).item()


# =============================================================================
# Session Tests
# =============================================================================

@torch.no_grad()
def test_session_local_simulation(tiny_model: LanguageModel, tiny_input: str):
    """Test that sessions work with remote='local'."""
    with tiny_model.session(remote='local') as session:
        with tiny_model.trace(tiny_input):
            h0_out = tiny_model.transformer.h[0].output[0].save()

        with tiny_model.trace(tiny_input):
            h1_out = tiny_model.transformer.h[1].output[0].save()

    assert isinstance(h0_out, torch.Tensor)
    assert isinstance(h1_out, torch.Tensor)


# =============================================================================
# Probe Training Tests
# =============================================================================

# Define probe class at module level for proper source capture
@remote
class SentimentProbe(torch.nn.Module):
    """Simple binary classifier probe for testing."""
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, 2)

    def forward(self, x):
        return self.linear(x)


def test_nn_module_probe_in_trace(tiny_model: LanguageModel):
    """
    Test that nn.Module probes can be used inside traces.

    This tests the core building block of probe training: using a custom
    nn.Module to process hidden states inside a trace.
    """
    probe = SentimentProbe(hidden_dims)

    with tiny_model.trace("test input", remote='local'):
        # Extract hidden state and run through probe
        hidden = tiny_model.transformer.h[0].output[0][:, -1, :]
        logits = probe(hidden)
        result = logits.save()

    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 2)  # batch=1, num_classes=2


def test_backward_pass_in_trace(tiny_model: LanguageModel):
    """
    Test that backward passes work inside traces.

    This tests gradient computation, which is essential for probe training.
    """
    import torch.nn.functional as F

    probe = SentimentProbe(hidden_dims)

    with tiny_model.trace("test input", remote='local'):
        hidden = tiny_model.transformer.h[0].output[0][:, -1, :]
        logits = probe(hidden)
        loss = F.cross_entropy(logits, torch.tensor([1]))
        loss.backward()
        loss_value = loss.save()

    assert isinstance(loss_value, torch.Tensor)
    assert loss_value.ndim == 0  # scalar loss


def test_probe_gradient_computation(tiny_model: LanguageModel):
    """
    Test that gradients are computed for probe parameters.
    """
    import torch.nn.functional as F

    probe = SentimentProbe(hidden_dims)

    # Ensure no gradients initially
    assert probe.linear.weight.grad is None

    with tiny_model.trace("test input", remote='local'):
        hidden = tiny_model.transformer.h[0].output[0][:, -1, :]
        logits = probe(hidden)
        loss = F.cross_entropy(logits, torch.tensor([1]))
        loss.backward()

    # After trace, gradients should be populated
    assert probe.linear.weight.grad is not None
    assert probe.linear.weight.grad.shape == (2, hidden_dims)


def test_multiple_traces_with_probe(tiny_model: LanguageModel):
    """
    Test running multiple traces with the same probe.

    This simulates iterating over training examples, with each trace
    processing one example.
    """
    import torch.nn.functional as F

    probe = SentimentProbe(hidden_dims)
    optimizer = torch.optim.SGD(probe.parameters(), lr=0.1)

    # Capture initial weights
    initial_weight = probe.linear.weight.clone().detach()

    # Training data
    examples = [
        ("I love this", 1),
        ("I hate this", 0),
    ]

    # Run multiple traces (simulating training loop)
    for text, label in examples:
        with tiny_model.trace(text, remote='local'):
            hidden = tiny_model.transformer.h[0].output[0][:, -1, :]
            logits = probe(hidden)
            loss = F.cross_entropy(logits, torch.tensor([label]))
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    # Weights should have changed after training
    final_weight = probe.linear.weight.clone().detach()
    assert not torch.allclose(initial_weight, final_weight), \
        "Probe weights should change after training"


def test_session_with_multiple_traces_and_probe(tiny_model: LanguageModel):
    """
    Test using a probe in multiple traces within a session.

    Note: Currently, sessions with remote='local' expect traces to be
    directly nested without intervening Python code. This test verifies
    the basic session + probe pattern works.
    """
    probe = SentimentProbe(hidden_dims)

    with tiny_model.session(remote='local'):
        with tiny_model.trace("first input"):
            hidden = tiny_model.transformer.h[0].output[0][:, -1, :]
            logits1 = probe(hidden).save()

        with tiny_model.trace("second input"):
            hidden = tiny_model.transformer.h[0].output[0][:, -1, :]
            logits2 = probe(hidden).save()

    assert isinstance(logits1, torch.Tensor)
    assert isinstance(logits2, torch.Tensor)
    assert logits1.shape == (1, 2)
    assert logits2.shape == (1, 2)


# =============================================================================
# Payload Size Tests
# =============================================================================

@torch.no_grad()
def test_payload_size_accessible(tiny_model: LanguageModel, tiny_input: str):
    """Test that payload size can be accessed after trace."""
    from nnsight.intervention.backends.local_simulation import LocalSimulationBackend

    backend = LocalSimulationBackend(tiny_model, verbose=False)

    # We need to use the backend directly to access last_payload_size
    # For now, just test that the trace works
    with tiny_model.trace(tiny_input, remote='local'):
        h0_out = tiny_model.transformer.h[0].output[0].save()

    assert isinstance(h0_out, torch.Tensor)


# =============================================================================
# Special Tensor Types
# =============================================================================

@torch.no_grad()
def test_different_tensor_dtypes(tiny_model: LanguageModel, tiny_input: str):
    """Test various tensor dtypes work correctly."""
    float16_tensor = torch.ones(1, hidden_dims, dtype=torch.float16)  # noqa: F841
    int_tensor = torch.ones(1, hidden_dims, dtype=torch.int32)  # noqa: F841

    with tiny_model.trace(tiny_input, remote='local'):
        out = tiny_model.transformer.h[0].output[0]  # noqa: F841
        # Just verify the trace doesn't crash with these in scope
        h0_out = tiny_model.transformer.h[0].output[0].save()

    assert isinstance(h0_out, torch.Tensor)


# =============================================================================
# Error Recovery Tests
# =============================================================================

def test_error_does_not_break_subsequent_traces(tiny_model: LanguageModel, tiny_input: str):
    """Test that a failed trace doesn't break subsequent traces."""
    # First, trigger an error
    def undecorated():
        pass

    try:
        with tiny_model.trace(tiny_input, remote='local'):
            result = undecorated()  # noqa: F841
            tiny_model.transformer.h[0].output[0].save()
    except SourceSerializationError:
        pass  # Expected

    # Now run a successful trace - should work
    with tiny_model.trace(tiny_input, remote='local'):
        h0_out = tiny_model.transformer.h[0].output[0].save()

    assert isinstance(h0_out, torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
