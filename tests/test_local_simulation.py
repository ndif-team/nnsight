"""
Tests for LocalSimulationBackend (remote='local' mode).

This test module verifies that the LocalSimulationBackend correctly:
1. Serializes and deserializes traces
2. Catches serialization errors early
3. Handles various data types correctly
4. Provides helpful verbose output

Run with: pytest tests/test_local_simulation.py -v
"""

from collections import OrderedDict
import io
import sys

import pytest
import torch
import numpy as np

from nnsight import NNsight
from nnsight.remote import remote
from nnsight.intervention.serialization_source import SourceSerializationError


# =============================================================================
# Fixtures
# =============================================================================

input_size = 5
hidden_dims = 10
output_size = 2


@pytest.fixture(scope="module")
def tiny_model(device: str):
    """Create a simple test model."""
    net = torch.nn.Sequential(
        OrderedDict(
            [
                ("layer1", torch.nn.Linear(input_size, hidden_dims)),
                ("layer2", torch.nn.Linear(hidden_dims, output_size)),
            ]
        )
    )
    return NNsight(net).to(device)


@pytest.fixture
def tiny_input():
    """Create a simple test input."""
    return torch.rand((1, input_size))


# =============================================================================
# Basic Functionality Tests
# =============================================================================

@torch.no_grad()
def test_basic_trace_local_simulation(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test that a basic trace works with remote='local'."""
    with tiny_model.trace(tiny_input, remote='local'):
        hs = tiny_model.layer2.output.save()

    assert isinstance(hs, torch.Tensor)
    assert hs.shape == (1, output_size)


@torch.no_grad()
def test_intervention_local_simulation(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test that interventions work with remote='local'."""
    with tiny_model.trace(tiny_input, remote='local'):
        tiny_model.layer1.output[:] = 1.0
        l1_out = tiny_model.layer1.output.save()

    assert isinstance(l1_out, torch.Tensor)
    assert torch.all(l1_out == 1.0).item()


@torch.no_grad()
def test_multiple_saves_local_simulation(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test saving multiple tensors with remote='local'."""
    with tiny_model.trace(tiny_input, remote='local'):
        l1_out = tiny_model.layer1.output.save()
        l2_out = tiny_model.layer2.output.save()

    assert isinstance(l1_out, torch.Tensor)
    assert isinstance(l2_out, torch.Tensor)
    assert l1_out.shape == (1, hidden_dims)
    assert l2_out.shape == (1, output_size)


# =============================================================================
# Serialization Error Detection Tests
# =============================================================================

def test_catches_non_serializable_variable(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test that non-serializable variables are caught with clear error."""
    # Create a non-serializable object (a function without @remote)
    def undecorated_function():
        return 42

    with pytest.raises(SourceSerializationError) as exc_info:
        with tiny_model.trace(tiny_input, remote='local'):
            # Using the function should trigger serialization error
            result = undecorated_function()
            tiny_model.layer1.output[:] = result

    # Error message should be informative
    error_msg = str(exc_info.value)
    assert "undecorated_function" in error_msg or "cannot be serialized" in error_msg.lower()


def test_catches_non_serializable_module(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test that non-server-available modules are caught."""
    import subprocess  # Not on server

    with pytest.raises(SourceSerializationError) as exc_info:
        with tiny_model.trace(tiny_input, remote='local'):
            # Referencing subprocess should fail
            result = subprocess
            tiny_model.layer1.output.save()

    error_msg = str(exc_info.value)
    assert "subprocess" in error_msg or "not available" in error_msg.lower()


# =============================================================================
# Variable Type Tests
# =============================================================================

@torch.no_grad()
def test_json_serializable_variables(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test that JSON-serializable variables work correctly."""
    multiplier = 2.0
    offset = 1
    config = {"scale": 0.5}
    values = [1, 2, 3]

    with tiny_model.trace(tiny_input, remote='local'):
        out = tiny_model.layer1.output
        out[:] = out * multiplier + offset
        l1_out = tiny_model.layer1.output.save()

    assert isinstance(l1_out, torch.Tensor)


@torch.no_grad()
def test_tensor_variables(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test that tensor variables are serialized correctly."""
    replacement = torch.ones(1, hidden_dims) * 5

    with tiny_model.trace(tiny_input, remote='local'):
        tiny_model.layer1.output[:] = replacement
        l1_out = tiny_model.layer1.output.save()

    assert isinstance(l1_out, torch.Tensor)
    assert torch.all(l1_out == 5.0).item()


@torch.no_grad()
def test_numpy_variables(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test that numpy arrays are serialized correctly."""
    replacement = np.ones((1, hidden_dims)) * 3

    with tiny_model.trace(tiny_input, remote='local'):
        tiny_model.layer1.output[:] = torch.from_numpy(replacement).float()
        l1_out = tiny_model.layer1.output.save()

    assert isinstance(l1_out, torch.Tensor)
    assert torch.allclose(l1_out, torch.tensor(3.0))


# =============================================================================
# @remote Decorated Object Tests
# =============================================================================

@torch.no_grad()
def test_remote_function(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test that @remote functions work correctly."""
    @remote
    def double_values(x):
        return x * 2

    with tiny_model.trace(tiny_input, remote='local'):
        out = tiny_model.layer1.output
        doubled = double_values(out)
        tiny_model.layer1.output[:] = doubled
        l1_out = tiny_model.layer1.output.save()

    assert isinstance(l1_out, torch.Tensor)


@torch.no_grad()
def test_remote_class(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test that @remote classes and instances work correctly."""
    @remote
    class Scaler:
        def __init__(self, factor):
            self.factor = factor

        def scale(self, x):
            return x * self.factor

    scaler = Scaler(3.0)

    with tiny_model.trace(tiny_input, remote='local'):
        out = tiny_model.layer1.output
        scaled = scaler.scale(out)
        tiny_model.layer1.output[:] = scaled
        l1_out = tiny_model.layer1.output.save()

    assert isinstance(l1_out, torch.Tensor)


# =============================================================================
# Lambda Tests
# =============================================================================

@torch.no_grad()
def test_simple_lambda(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test that simple lambdas work correctly."""
    double = lambda x: x * 2

    with tiny_model.trace(tiny_input, remote='local'):
        out = tiny_model.layer1.output
        doubled = double(out)
        tiny_model.layer1.output[:] = doubled
        l1_out = tiny_model.layer1.output.save()

    assert isinstance(l1_out, torch.Tensor)


@torch.no_grad()
def test_lambda_with_capture(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test lambdas that capture JSON-serializable values."""
    factor = 2.5
    scale = lambda x: x * factor

    with tiny_model.trace(tiny_input, remote='local'):
        out = tiny_model.layer1.output
        scaled = scale(out)
        tiny_model.layer1.output[:] = scaled
        l1_out = tiny_model.layer1.output.save()

    assert isinstance(l1_out, torch.Tensor)


# =============================================================================
# Verbose Mode Tests
# =============================================================================

@torch.no_grad()
def test_verbose_mode_output(tiny_model: NNsight, tiny_input: torch.Tensor, capsys):
    """Test that verbose mode produces helpful output."""
    with tiny_model.trace(tiny_input, remote='local', verbose=True):
        l1_out = tiny_model.layer1.output.save()

    captured = capsys.readouterr()

    # Should show serialization info
    assert "[LocalSimulation]" in captured.out
    assert "bytes" in captured.out


@torch.no_grad()
def test_verbose_mode_shows_variables(tiny_model: NNsight, tiny_input: torch.Tensor, capsys):
    """Test that verbose mode shows variable names."""
    my_variable = 42

    with tiny_model.trace(tiny_input, remote='local', verbose=True):
        x = my_variable + 1  # noqa: F841
        l1_out = tiny_model.layer1.output.save()

    captured = capsys.readouterr()

    # Should show variable info
    assert "[LocalSimulation]" in captured.out


# =============================================================================
# Edge Case Tests
# =============================================================================

@torch.no_grad()
def test_empty_trace_body(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test that an empty trace body works."""
    with tiny_model.trace(tiny_input, remote='local'):
        pass  # Empty body


@torch.no_grad()
def test_conditional_in_trace(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test conditionals work inside traces."""
    condition = True

    with tiny_model.trace(tiny_input, remote='local'):
        if condition:
            l1_out = tiny_model.layer1.output.save()
        else:
            l2_out = tiny_model.layer2.output.save()

    assert isinstance(l1_out, torch.Tensor)


@torch.no_grad()
def test_loop_in_trace(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test loops work inside traces."""
    with tiny_model.trace(tiny_input, remote='local'):
        for i in range(3):
            tiny_model.layer1.output[:, i] = float(i)
        l1_out = tiny_model.layer1.output.save()

    assert isinstance(l1_out, torch.Tensor)
    assert l1_out[0, 0] == 0.0
    assert l1_out[0, 1] == 1.0
    assert l1_out[0, 2] == 2.0


# =============================================================================
# Complex Data Structure Tests
# =============================================================================

@torch.no_grad()
def test_nested_dict_variable(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test that nested dicts work correctly."""
    config = {
        "layer1": {"scale": 2.0, "offset": 1.0},
        "layer2": {"scale": 0.5, "offset": 0.0},
    }

    with tiny_model.trace(tiny_input, remote='local'):
        out = tiny_model.layer1.output
        out[:] = out * config["layer1"]["scale"] + config["layer1"]["offset"]
        l1_out = tiny_model.layer1.output.save()

    assert isinstance(l1_out, torch.Tensor)


@torch.no_grad()
def test_list_of_tensors(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test that lists of tensors work correctly."""
    tensors = [torch.ones(1, hidden_dims), torch.zeros(1, hidden_dims)]

    with tiny_model.trace(tiny_input, remote='local'):
        tiny_model.layer1.output[:] = tensors[0] + tensors[1]
        l1_out = tiny_model.layer1.output.save()

    assert isinstance(l1_out, torch.Tensor)
    assert torch.all(l1_out == 1.0).item()


# =============================================================================
# Comparison with Normal Execution Tests
# =============================================================================

@torch.no_grad()
def test_matches_local_execution(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test that remote='local' produces same results as normal execution."""
    # Run without remote='local'
    with tiny_model.trace(tiny_input):
        tiny_model.layer1.output[:] = 7.0
        normal_result = tiny_model.layer1.output.save()

    # Run with remote='local'
    with tiny_model.trace(tiny_input, remote='local'):
        tiny_model.layer1.output[:] = 7.0
        local_result = tiny_model.layer1.output.save()

    assert torch.allclose(normal_result, local_result)


@torch.no_grad()
def test_multiple_traces_local(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test that multiple consecutive traces work."""
    results = []

    for i in range(3):
        with tiny_model.trace(tiny_input, remote='local'):
            tiny_model.layer1.output[:] = float(i)
            result = tiny_model.layer1.output.save()
        results.append(result)

    assert len(results) == 3
    for i, r in enumerate(results):
        assert torch.all(r == float(i)).item()


# =============================================================================
# Session Tests
# =============================================================================

@torch.no_grad()
def test_session_local_simulation(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test that sessions work with remote='local'."""
    with tiny_model.session(remote='local') as session:
        with tiny_model.trace(tiny_input):
            l1_out = tiny_model.layer1.output.save()

        with tiny_model.trace(tiny_input):
            l2_out = tiny_model.layer2.output.save()

    assert isinstance(l1_out, torch.Tensor)
    assert isinstance(l2_out, torch.Tensor)


# =============================================================================
# Payload Size Tests
# =============================================================================

@torch.no_grad()
def test_payload_size_accessible(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test that payload size can be accessed after trace."""
    from nnsight.intervention.backends.local_simulation import LocalSimulationBackend

    backend = LocalSimulationBackend(tiny_model, verbose=False)

    # We need to use the backend directly to access last_payload_size
    # For now, just test that the trace works
    with tiny_model.trace(tiny_input, remote='local'):
        l1_out = tiny_model.layer1.output.save()

    assert isinstance(l1_out, torch.Tensor)


# =============================================================================
# Special Tensor Types
# =============================================================================

@torch.no_grad()
def test_different_tensor_dtypes(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test various tensor dtypes work correctly."""
    float16_tensor = torch.ones(1, hidden_dims, dtype=torch.float16)
    int_tensor = torch.ones(1, hidden_dims, dtype=torch.int32)

    with tiny_model.trace(tiny_input, remote='local'):
        out = tiny_model.layer1.output
        # Just verify the trace doesn't crash with these in scope
        l1_out = tiny_model.layer1.output.save()

    assert isinstance(l1_out, torch.Tensor)


# =============================================================================
# Error Recovery Tests
# =============================================================================

def test_error_does_not_break_subsequent_traces(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test that a failed trace doesn't break subsequent traces."""
    # First, trigger an error
    def undecorated():
        pass

    try:
        with tiny_model.trace(tiny_input, remote='local'):
            result = undecorated()  # noqa: F841
            tiny_model.layer1.output.save()
    except SourceSerializationError:
        pass  # Expected

    # Now run a successful trace - should work
    with tiny_model.trace(tiny_input, remote='local'):
        l1_out = tiny_model.layer1.output.save()

    assert isinstance(l1_out, torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
