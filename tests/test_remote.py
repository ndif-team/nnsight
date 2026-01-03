"""
Tests for @nnsight.remote decorator and source-based serialization.
"""

import json
import pytest
import numpy as np
import torch

from nnsight import remote, RemoteValidationError
from nnsight.intervention.serialization_source import (
    is_json_serializable,
    SourceSerializationError,
)


# =============================================================================
# Test: Basic decorator functionality
# =============================================================================

def test_remote_valid_function():
    """Test that @remote decorator works on simple functions."""
    @remote
    def normalize(x):
        return x / x.norm()

    assert hasattr(normalize, '_remote_source')
    assert hasattr(normalize, '_remote_module_refs')
    assert normalize._remote_validated is True
    assert 'def normalize' in normalize._remote_source


def test_remote_valid_class():
    """Test that @remote decorator works on simple classes."""
    @remote
    class ValidClass:
        def __init__(self, x):
            self.x = x

        def compute(self, y):
            return self.x + y

    assert hasattr(ValidClass, '_remote_source')
    assert ValidClass._remote_validated is True
    assert 'class ValidClass' in ValidClass._remote_source


def test_remote_function_with_torch():
    """Test that functions using torch work."""
    @remote
    def torch_func(x):
        import torch
        return torch.nn.functional.softmax(x, dim=-1)

    assert torch_func._remote_validated is True


def test_remote_function_with_numpy():
    """Test that functions using numpy work."""
    @remote
    def numpy_func(x):
        import numpy as np
        return np.mean(x)

    assert numpy_func._remote_validated is True


# =============================================================================
# Test: Module-level reference capture
# =============================================================================

TOP_K = 10
VOCAB_SIZE = 50257
LAYERS = [5, 10, 15]


def test_remote_captures_constants():
    """Test that JSON-serializable constants are captured."""
    @remote
    def use_constant():
        return TOP_K

    assert 'TOP_K' in use_constant._remote_module_refs
    assert use_constant._remote_module_refs['TOP_K'] == 10


def test_remote_captures_list_constants():
    """Test that list constants are captured."""
    @remote
    def use_layers():
        return LAYERS

    assert 'LAYERS' in use_layers._remote_module_refs
    assert use_layers._remote_module_refs['LAYERS'] == [5, 10, 15]


# =============================================================================
# Test: Module alias handling
# =============================================================================

def test_remote_allows_numpy_alias():
    """Test that np alias for numpy is allowed."""
    @remote
    def use_np():
        return np.array([1, 2, 3])

    # np should NOT be in module_refs (it's available on server)
    assert 'np' not in use_np._remote_module_refs
    assert use_np._remote_validated is True


def test_remote_allows_torch_alias():
    """Test that torch is allowed."""
    @remote
    def use_torch():
        return torch.tensor([1, 2, 3])

    assert 'torch' not in use_torch._remote_module_refs
    assert use_torch._remote_validated is True


# =============================================================================
# Test: Validation errors
# =============================================================================

def test_remote_rejects_disallowed_import():
    """Test that importing disallowed modules raises error."""
    with pytest.raises(RemoteValidationError) as exc_info:
        @remote
        def bad_import():
            import pandas
            return pandas.DataFrame()

    assert "imports 'pandas'" in str(exc_info.value)


def test_remote_rejects_open():
    """Test that calling open() raises error."""
    with pytest.raises(RemoteValidationError) as exc_info:
        @remote
        def file_io():
            return open("file.txt").read()

    assert "calls 'open()'" in str(exc_info.value)


def test_remote_rejects_exec():
    """Test that calling exec() raises error."""
    with pytest.raises(RemoteValidationError) as exc_info:
        @remote
        def use_exec():
            exec("print('hello')")

    assert "calls 'exec()'" in str(exc_info.value)


def test_remote_rejects_eval():
    """Test that calling eval() raises error."""
    with pytest.raises(RemoteValidationError) as exc_info:
        @remote
        def use_eval():
            return eval("1 + 1")

    assert "calls 'eval()'" in str(exc_info.value)


# =============================================================================
# Test: Class-specific validation
# =============================================================================

def test_remote_rejects_non_remote_base():
    """Test that classes with non-remote base classes are rejected."""
    class NonRemoteBase:
        pass

    with pytest.raises(RemoteValidationError) as exc_info:
        @remote
        class BadClass(NonRemoteBase):
            pass

    assert "Base class 'NonRemoteBase'" in str(exc_info.value)


def test_remote_allows_remote_base():
    """Test that classes with @remote base classes are allowed."""
    @remote
    class RemoteBase:
        pass

    @remote
    class DerivedClass(RemoteBase):
        pass

    assert DerivedClass._remote_validated is True


def test_remote_rejects_slots():
    """Test that classes with __slots__ are rejected."""
    with pytest.raises(RemoteValidationError) as exc_info:
        @remote
        class SlottedClass:
            __slots__ = ['x', 'y']

    assert "__slots__" in str(exc_info.value)


# =============================================================================
# Test: Version metadata
# =============================================================================

def test_remote_with_version():
    """Test that version metadata can be specified."""
    @remote(library="mylib", version="1.0.0")
    class VersionedClass:
        pass

    assert VersionedClass._remote_library == "mylib"
    assert VersionedClass._remote_version == "1.0.0"


def test_remote_without_version():
    """Test that version metadata is auto-detected or None."""
    @remote
    class UnversionedClass:
        pass

    # May be None or auto-detected from package
    assert hasattr(UnversionedClass, '_remote_library')
    assert hasattr(UnversionedClass, '_remote_version')


# =============================================================================
# Test: JSON serialization helpers
# =============================================================================

def test_is_json_serializable_primitives():
    """Test JSON serialization detection for primitives."""
    assert is_json_serializable(None) is True
    assert is_json_serializable(True) is True
    assert is_json_serializable(42) is True
    assert is_json_serializable(3.14) is True
    assert is_json_serializable("hello") is True


def test_is_json_serializable_collections():
    """Test JSON serialization detection for collections."""
    assert is_json_serializable([1, 2, 3]) is True
    assert is_json_serializable((1, 2, 3)) is True
    assert is_json_serializable({"a": 1, "b": 2}) is True
    assert is_json_serializable([{"nested": [1, 2]}]) is True


def test_is_json_serializable_rejects_complex():
    """Test that complex objects are not JSON serializable."""
    assert is_json_serializable(torch.tensor([1, 2])) is False
    assert is_json_serializable(lambda x: x) is False

    class CustomClass:
        pass
    assert is_json_serializable(CustomClass()) is False


# =============================================================================
# Test: Introspection is allowed
# =============================================================================

def test_remote_allows_introspection():
    """Test that introspection builtins are allowed."""
    @remote
    class IntrospectiveClass:
        def __init__(self, model):
            self.model = model

        def check_model(self):
            # These should all be allowed
            has_transformer = hasattr(self.model, 'transformer')
            config = getattr(self.model, 'config', None)
            is_model = isinstance(self.model, type(self.model))
            return has_transformer, config, is_model

    assert IntrospectiveClass._remote_validated is True


# =============================================================================
# Test: Complex valid examples
# =============================================================================

def test_remote_complex_class():
    """Test a realistic helper class."""
    @remote
    class LogitLensKit:
        def __init__(self, model, top_k=10):
            self.model = model
            self.top_k = top_k
            self.layers = []

        def get_hidden(self, layer):
            return self.model.transformer.h[layer].output[0]

        def project(self, h):
            return h @ self.model.lm_head.weight.T

        def analyze(self, layers):
            results = []
            for layer in layers:
                h = self.get_hidden(layer)
                logits = self.project(h)
                results.append(logits.topk(self.top_k))
            return results

    assert LogitLensKit._remote_validated is True
    assert 'class LogitLensKit' in LogitLensKit._remote_source
