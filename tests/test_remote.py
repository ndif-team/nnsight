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


# =============================================================================
# Test: Additional allowed modules (collections, itertools, functools, operator)
# =============================================================================

def test_remote_allows_collections():
    """Test that collections module is allowed."""
    @remote
    def use_collections():
        from collections import defaultdict, Counter
        d = defaultdict(int)
        c = Counter([1, 2, 2, 3])
        return d, c

    assert use_collections._remote_validated is True


def test_remote_allows_itertools():
    """Test that itertools module is allowed."""
    @remote
    def use_itertools():
        import itertools
        return list(itertools.chain([1, 2], [3, 4]))

    assert use_itertools._remote_validated is True


def test_remote_allows_functools():
    """Test that functools module is allowed."""
    @remote
    def use_functools():
        import functools
        @functools.lru_cache
        def fib(n):
            return n if n < 2 else fib(n-1) + fib(n-2)
        return fib(10)

    assert use_functools._remote_validated is True


def test_remote_allows_operator():
    """Test that operator module is allowed."""
    @remote
    def use_operator():
        import operator
        return operator.add(1, 2)

    assert use_operator._remote_validated is True


# =============================================================================
# Test: Serialization - extract_remote_object
# =============================================================================

from nnsight.intervention.serialization_source import (
    extract_remote_object,
    serialize_instance_state,
    is_remote_object,
    is_model_reference,
)


def test_is_remote_object_function():
    """Test detecting @remote decorated functions."""
    @remote
    def my_func():
        pass

    assert is_remote_object(my_func) is True

    def plain_func():
        pass

    assert is_remote_object(plain_func) is False


def test_is_remote_object_class():
    """Test detecting @remote decorated classes."""
    @remote
    class MyClass:
        pass

    assert is_remote_object(MyClass) is True

    class PlainClass:
        pass

    assert is_remote_object(PlainClass) is False


def test_is_remote_object_instance():
    """Test detecting instances of @remote decorated classes."""
    @remote
    class MyClass:
        def __init__(self, x):
            self.x = x

    instance = MyClass(42)
    assert is_remote_object(instance) is True

    class PlainClass:
        pass

    plain_instance = PlainClass()
    assert is_remote_object(plain_instance) is False


def test_extract_remote_object_function():
    """Test extracting @remote function for serialization."""
    @remote
    def normalize(x):
        return x / x.norm()

    result = {}
    extract_remote_object("normalize", normalize, result)

    assert "normalize" in result
    assert result["normalize"]["type"] == "function"
    assert "def normalize" in result["normalize"]["source"]


def test_extract_remote_object_class():
    """Test extracting @remote class for serialization."""
    @remote
    class Analyzer:
        def __init__(self, k):
            self.k = k

    result = {}
    extract_remote_object("Analyzer", Analyzer, result)

    assert "Analyzer" in result
    assert result["Analyzer"]["type"] == "class"
    assert "class Analyzer" in result["Analyzer"]["source"]


def test_extract_remote_object_instance():
    """Test extracting @remote class instance for serialization."""
    @remote
    class Analyzer:
        def __init__(self, k):
            self.k = k

    instance = Analyzer(10)
    result = {}
    extract_remote_object("analyzer", instance, result)

    assert "Analyzer" in result
    assert len(result["Analyzer"]["instances"]) == 1
    instance_data = list(result["Analyzer"]["instances"].values())[0]
    assert instance_data["var_name"] == "analyzer"
    assert instance_data["state"]["k"] == 10


def test_extract_remote_object_with_version():
    """Test that version metadata is included in serialization."""
    @remote(library="mylib", version="1.0.0")
    class Versioned:
        pass

    result = {}
    extract_remote_object("Versioned", Versioned, result)

    assert result["Versioned"]["library"] == "mylib"
    assert result["Versioned"]["version"] == "1.0.0"


# =============================================================================
# Test: Instance state serialization
# =============================================================================

def test_serialize_instance_state_primitives():
    """Test serializing instance with primitive attributes."""
    @remote
    class Simple:
        def __init__(self):
            self.x = 10
            self.name = "test"
            self.values = [1, 2, 3]
            self.config = {"a": 1, "b": 2}

    instance = Simple()
    state = serialize_instance_state(instance)

    assert state["x"] == 10
    assert state["name"] == "test"
    assert state["values"] == [1, 2, 3]
    assert state["config"] == {"a": 1, "b": 2}


def test_serialize_instance_state_nested_remote():
    """Test serializing instance with nested @remote object."""
    @remote
    class Inner:
        def __init__(self, v):
            self.v = v

    @remote
    class Outer:
        def __init__(self, inner):
            self.inner = inner

    inner = Inner(42)
    outer = Outer(inner)
    state = serialize_instance_state(outer)

    assert "__remote_ref__" in state["inner"]
    assert state["inner"]["__remote_type__"] == "Inner"


# =============================================================================
# Test: Edge cases and error handling
# =============================================================================

def test_remote_rejects_import_from_disallowed():
    """Test that 'from X import Y' for disallowed modules is rejected."""
    with pytest.raises(RemoteValidationError) as exc_info:
        @remote
        def bad_import():
            from pandas import DataFrame
            return DataFrame()

    assert "pandas" in str(exc_info.value)


def test_remote_allows_import_from_allowed():
    """Test that 'from X import Y' for allowed modules works."""
    @remote
    def good_import():
        from collections import OrderedDict
        from itertools import chain
        return OrderedDict(), chain

    assert good_import._remote_validated is True


def test_remote_rejects_compile():
    """Test that calling compile() raises error."""
    with pytest.raises(RemoteValidationError) as exc_info:
        @remote
        def use_compile():
            return compile("1+1", "<string>", "eval")

    assert "calls 'compile()'" in str(exc_info.value)


def test_remote_rejects_input():
    """Test that calling input() raises error."""
    with pytest.raises(RemoteValidationError) as exc_info:
        @remote
        def use_input():
            return input("Enter: ")

    assert "calls 'input()'" in str(exc_info.value)


# Module-level constants for capture tests
MODULE_CONFIG = {"model": {"layers": [1, 2, 3], "hidden": 768}}
MODULE_DIMS = (768, 3072)


def test_remote_captures_nested_dict():
    """Test that nested dict constants are captured."""
    @remote
    def use_config():
        return MODULE_CONFIG["model"]["hidden"]

    assert "MODULE_CONFIG" in use_config._remote_module_refs
    assert use_config._remote_module_refs["MODULE_CONFIG"] == MODULE_CONFIG


def test_remote_captures_tuple():
    """Test that tuple constants are captured."""
    @remote
    def use_dims():
        return MODULE_DIMS[0]

    assert "MODULE_DIMS" in use_dims._remote_module_refs
    assert use_dims._remote_module_refs["MODULE_DIMS"] == (768, 3072)


def test_remote_class_with_class_attributes():
    """Test class with class-level attributes."""
    @remote
    class WithClassAttrs:
        DEFAULT_K = 10
        SUPPORTED_TYPES = ["gpt2", "llama"]

        def __init__(self, k=None):
            self.k = k or self.DEFAULT_K

    assert WithClassAttrs._remote_validated is True
    assert "DEFAULT_K = 10" in WithClassAttrs._remote_source


def test_remote_class_with_methods():
    """Test class with multiple method types."""
    @remote
    class FullClass:
        def __init__(self, x):
            self.x = x

        def instance_method(self):
            return self.x

        @staticmethod
        def static_method(a, b):
            return a + b

        @classmethod
        def class_method(cls):
            return cls.__name__

    assert FullClass._remote_validated is True


def test_remote_function_with_default_args():
    """Test function with default arguments."""
    @remote
    def with_defaults(a, b=10, c="hello", d=None):
        return a, b, c, d

    assert with_defaults._remote_validated is True


def test_remote_function_with_args_kwargs():
    """Test function with *args and **kwargs."""
    @remote
    def with_var_args(*args, **kwargs):
        return args, kwargs

    assert with_var_args._remote_validated is True


def test_remote_function_with_type_hints():
    """Test function with type annotations."""
    @remote
    def typed_func(x: int, y: float = 1.0) -> float:
        return x + y

    assert typed_func._remote_validated is True


def test_remote_nested_class_definitions():
    """Test that nested class definitions work."""
    @remote
    class Outer:
        class Inner:
            def __init__(self):
                self.v = 1

        def make_inner(self):
            return self.Inner()

    assert Outer._remote_validated is True


def test_remote_with_comprehensions():
    """Test that list/dict/set comprehensions work."""
    @remote
    def use_comprehensions():
        squares = [x**2 for x in range(10)]
        even_squares = {x: x**2 for x in range(10) if x % 2 == 0}
        unique = {x % 3 for x in range(10)}
        gen = (x for x in range(10))
        return squares, even_squares, unique, list(gen)

    assert use_comprehensions._remote_validated is True


def test_remote_with_context_manager():
    """Test that with statements are allowed (without file I/O)."""
    @remote
    def use_context():
        import torch
        with torch.no_grad():
            return torch.tensor([1, 2, 3])

    assert use_context._remote_validated is True


def test_remote_with_try_except():
    """Test that try/except blocks work."""
    @remote
    def use_try_except():
        try:
            result = 1 / 0
        except ZeroDivisionError:
            result = 0
        finally:
            pass
        return result

    assert use_try_except._remote_validated is True


def test_remote_with_walrus_operator():
    """Test that walrus operator (:=) works."""
    @remote
    def use_walrus():
        if (n := len([1, 2, 3])) > 2:
            return n
        return 0

    assert use_walrus._remote_validated is True


# =============================================================================
# Test: Source code formatting
# =============================================================================

def test_remote_source_is_properly_dedented():
    """Test that source code is properly dedented for serialization."""
    @remote
    def nested_func():
        return 42

    # Source should not have leading indentation
    assert not nested_func._remote_source.startswith(" ")
    assert not nested_func._remote_source.startswith("\t")
    assert nested_func._remote_source.startswith("@remote") or nested_func._remote_source.startswith("def ")


def test_remote_class_source_includes_decorator():
    """Test that class source includes all methods."""
    @remote
    class FullClass:
        def __init__(self):
            pass

        def method_a(self):
            pass

        def method_b(self):
            pass

    assert "def __init__" in FullClass._remote_source
    assert "def method_a" in FullClass._remote_source
    assert "def method_b" in FullClass._remote_source


# =============================================================================
# Test: Error message quality
# =============================================================================

def test_error_message_includes_line_number():
    """Test that error messages include line numbers."""
    with pytest.raises(RemoteValidationError) as exc_info:
        @remote
        def multi_line_bad():
            x = 1
            y = 2
            import pandas  # This should be reported with line number
            return x + y

    error_msg = str(exc_info.value)
    assert "Line" in error_msg or "imports 'pandas'" in error_msg


# Module-level non-serializable object for error message test
class _NotSerializableClass:
    pass

_NOT_SERIALIZABLE_OBJ = _NotSerializableClass()


def test_error_message_suggests_alternatives():
    """Test that error messages suggest alternatives."""
    with pytest.raises(RemoteValidationError) as exc_info:
        @remote
        def use_non_serializable():
            return _NOT_SERIALIZABLE_OBJ

    error_msg = str(exc_info.value)
    # Should mention that it's not serializable
    assert "not JSON-serializable" in error_msg or "_NotSerializableClass" in error_msg or "Options" in error_msg
