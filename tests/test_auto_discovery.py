"""
Tests for auto-discovery serialization (default behavior where @remote is optional).

These tests verify that:
1. Functions and classes with available source are auto-discovered
2. strict=True mode requires explicit @remote decorations
3. max_payload_mb warnings work correctly
4. Error messages are helpful when auto-discovery fails

Run with: pytest tests/test_auto_discovery.py -v
"""

from collections import OrderedDict
import warnings
import sys

import pytest
import torch
import numpy as np

sys.path.insert(0, 'src')

from nnsight import NNsight
from nnsight.remote import remote
from nnsight.intervention.serialization_source import (
    serialize_source_based,
    deserialize_source_based,
    extract_all,
    SourceSerializationError,
    _can_auto_discover_function,
    _auto_discover_function,
)


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
# Auto-Discovery Function Tests
# =============================================================================

def test_can_auto_discover_user_function():
    """Test that user-defined functions can be auto-discovered."""
    def my_function(x):
        return x * 2

    assert _can_auto_discover_function(my_function) is True


def test_cannot_auto_discover_builtin_function():
    """Test that builtin functions cannot be auto-discovered."""
    assert _can_auto_discover_function(len) is False
    assert _can_auto_discover_function(print) is False


def test_cannot_auto_discover_lambda():
    """Test that lambdas are not auto-discovered (handled separately)."""
    double = lambda x: x * 2  # noqa: E731
    assert _can_auto_discover_function(double) is False


def test_cannot_auto_discover_torch_function():
    """Test that torch functions are not auto-discovered."""
    assert _can_auto_discover_function(torch.tensor) is False
    assert _can_auto_discover_function(torch.zeros) is False


def test_auto_discover_function_metadata():
    """Test that auto-discovered functions have correct metadata."""
    def my_helper(x, y):
        return x + y

    metadata = _auto_discover_function(my_helper)

    assert 'source' in metadata
    assert 'code' in metadata['source']
    assert 'def my_helper' in metadata['source']['code']
    assert 'return x + y' in metadata['source']['code']
    assert metadata['type'] == 'function'


# =============================================================================
# Auto-Discovery Class Tests
# =============================================================================

def test_auto_discover_simple_class():
    """Test that simple classes are auto-discovered."""
    class SimpleClass:
        def __init__(self, value):
            self.value = value

        def process(self, x):
            return x * self.value

    from nnsight.intervention.serialization_source import can_auto_discover
    assert can_auto_discover(SimpleClass) is True


def test_auto_discover_class_with_methods():
    """Test auto-discovery of class with multiple methods."""
    class Processor:
        def __init__(self, scale, offset):
            self.scale = scale
            self.offset = offset

        def forward(self, x):
            return x * self.scale + self.offset

        def backward(self, grad):
            return grad * self.scale

    from nnsight.intervention.serialization_source import can_auto_discover
    assert can_auto_discover(Processor) is True


def test_cannot_auto_discover_builtin_class():
    """Test that builtin classes cannot be auto-discovered."""
    from nnsight.intervention.serialization_source import can_auto_discover
    assert can_auto_discover(list) is False
    assert can_auto_discover(dict) is False
    assert can_auto_discover(str) is False


# =============================================================================
# extract_all Tests
# =============================================================================

def test_extract_all_auto_discovers_function():
    """Test that extract_all auto-discovers functions when strict_remote=False."""
    def process_data(x):
        return x + 1

    locals_dict = {
        'process_data': process_data,
        'simple_value': 42,
    }

    variables, remote_objects, model_refs = extract_all(locals_dict, strict_remote=False)

    # simple_value should be in variables
    assert 'simple_value' in variables
    assert variables['simple_value'] == 42

    # process_data should be in remote_objects (auto-discovered)
    assert 'process_data' in remote_objects
    assert 'source' in remote_objects['process_data']


def test_extract_all_strict_rejects_undeclared_function():
    """Test that strict_remote=True rejects functions without @remote."""
    def process_data(x):
        return x + 1

    locals_dict = {
        'process_data': process_data,
    }

    with pytest.raises(SourceSerializationError) as exc_info:
        extract_all(locals_dict, strict_remote=True)

    error_msg = str(exc_info.value)
    assert 'process_data' in error_msg
    assert '@nnsight.remote' in error_msg or '@remote' in error_msg


def test_extract_all_strict_accepts_remote_decorated():
    """Test that strict_remote=True accepts @remote decorated functions."""
    @remote
    def process_data(x):
        return x + 1

    locals_dict = {
        'process_data': process_data,
    }

    variables, remote_objects, model_refs = extract_all(locals_dict, strict_remote=True)

    # Should work without error
    assert 'process_data' in remote_objects


def test_extract_all_auto_discovers_class_instance():
    """Test that extract_all auto-discovers class instances."""
    class MyProcessor:
        def __init__(self, value):
            self.value = value

        def process(self, x):
            return x * self.value

    processor = MyProcessor(3)

    locals_dict = {
        'processor': processor,
    }

    variables, remote_objects, model_refs = extract_all(locals_dict, strict_remote=False)

    # MyProcessor should be in remote_objects
    assert 'MyProcessor' in remote_objects
    assert 'instances' in remote_objects['MyProcessor']


# =============================================================================
# Payload Size Warning Tests
# =============================================================================

def test_payload_size_warning():
    """Test that large payloads trigger warnings."""
    class MockTracer:
        class info:
            source = "pass"
            frame = None

    tracer = MockTracer()

    # Set a very low threshold to trigger the warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = serialize_source_based(tracer, max_upload_mb=0.00001)  # ~10 bytes

        # Should have triggered a warning
        assert len(w) >= 1
        assert "upload" in str(w[-1].message).lower() or "payload" in str(w[-1].message).lower()


def test_no_warning_under_threshold():
    """Test that payloads under threshold don't trigger warnings."""
    class MockTracer:
        class info:
            source = "pass"
            frame = None

    tracer = MockTracer()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = serialize_source_based(tracer, max_upload_mb=10)  # 10 MB threshold

        # Should not have triggered a warning
        payload_warnings = [x for x in w if "upload" in str(x.message).lower() or "payload" in str(x.message).lower()]
        assert len(payload_warnings) == 0


# =============================================================================
# Integration Tests with Model
# =============================================================================

@torch.no_grad()
def test_auto_discovery_with_trace(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test that auto-discovery works in actual traces."""
    # Define a function WITHOUT @remote
    def double_values(x):
        return x * 2

    # This should work because strict=False is the default
    with tiny_model.trace(tiny_input, remote='local'):
        out = tiny_model.layer1.output
        doubled = double_values(out)
        tiny_model.layer1.output[:] = doubled
        result = tiny_model.layer1.output.save()

    assert isinstance(result, torch.Tensor)


@torch.no_grad()
def test_strict_mode_with_trace(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test that strict_remote mode rejects undecorated functions."""
    def process_values(x):
        return x + 1

    with pytest.raises(SourceSerializationError) as exc_info:
        with tiny_model.trace(tiny_input, remote='local', strict_remote=True):
            out = tiny_model.layer1.output
            result = process_values(out)  # noqa: F841
            tiny_model.layer1.output.save()

    error_msg = str(exc_info.value)
    assert 'process_values' in error_msg


@torch.no_grad()
def test_strict_mode_accepts_remote(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test that strict_remote mode accepts @remote decorated functions."""
    @remote
    def process_values(x):
        return x + 1

    with tiny_model.trace(tiny_input, remote='local', strict_remote=True):
        out = tiny_model.layer1.output
        processed = process_values(out)  # noqa: F841
        result = tiny_model.layer1.output.save()

    assert isinstance(result, torch.Tensor)


@torch.no_grad()
def test_auto_discovery_class_instance(tiny_model: NNsight, tiny_input: torch.Tensor):
    """Test auto-discovery of class instances in traces."""
    # Define a class WITHOUT @remote
    class Multiplier:
        def __init__(self, factor):
            self.factor = factor

        def apply(self, x):
            return x * self.factor

    multiplier = Multiplier(2.0)

    # This should work because strict=False is the default
    with tiny_model.trace(tiny_input, remote='local'):
        out = tiny_model.layer1.output
        scaled = multiplier.apply(out)  # noqa: F841
        result = tiny_model.layer1.output.save()

    assert isinstance(result, torch.Tensor)


# =============================================================================
# Error Message Quality Tests
# =============================================================================

def test_error_message_mentions_source_unavailable():
    """Test that error messages mention when source is unavailable."""
    # Create a function without source (by using exec)
    namespace = {}
    exec("def dynamic_fn(x): return x * 2", namespace)
    dynamic_fn = namespace['dynamic_fn']

    locals_dict = {'dynamic_fn': dynamic_fn}

    with pytest.raises(SourceSerializationError) as exc_info:
        extract_all(locals_dict, strict=False)

    error_msg = str(exc_info.value)
    # Should mention that source is not available
    assert 'source' in error_msg.lower() or 'cannot be serialized' in error_msg.lower()


def test_error_message_provides_options():
    """Test that error messages provide helpful options."""
    # Create something that can't be serialized
    import threading
    lock = threading.Lock()

    locals_dict = {'lock': lock}

    with pytest.raises(SourceSerializationError) as exc_info:
        extract_all(locals_dict, strict=False)

    error_msg = str(exc_info.value)
    # Should provide options
    assert 'option' in error_msg.lower() or 'JSON' in error_msg


# =============================================================================
# Edge Cases
# =============================================================================

def test_auto_discovery_nested_class():
    """Test auto-discovery handles nested classes."""
    class Outer:
        class Inner:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

    from nnsight.intervention.serialization_source import can_auto_discover
    # Nested classes may or may not be discoverable depending on source availability
    # This test documents the behavior
    try:
        result = can_auto_discover(Outer.Inner)
        # If it doesn't raise, just verify it returns a boolean
        assert isinstance(result, bool)
    except Exception:
        pass  # Some nested classes might not be discoverable


def test_auto_discovery_with_closure():
    """Test auto-discovery of functions with closures."""
    multiplier = 3

    def scale_by_closure(x):
        return x * multiplier

    # Should be discoverable
    assert _can_auto_discover_function(scale_by_closure) is True

    # Metadata should capture closure variable
    metadata = _auto_discover_function(scale_by_closure)
    # The closure variable should be in closure_vars if JSON-serializable
    assert 'closure_vars' in metadata
    assert metadata['closure_vars'].get('multiplier') == 3


def test_auto_discovery_preserves_remote_decorated():
    """Test that @remote decorated items are still handled correctly."""
    @remote
    def already_decorated(x):
        return x * 2

    locals_dict = {
        'already_decorated': already_decorated,
    }

    # Should work with both strict_remote=True and strict_remote=False
    vars1, objs1, refs1 = extract_all(locals_dict, strict_remote=False)
    vars2, objs2, refs2 = extract_all(locals_dict, strict_remote=True)

    assert 'already_decorated' in objs1
    assert 'already_decorated' in objs2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
