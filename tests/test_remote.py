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


def test_remote_accepts_slots():
    """Test that classes with __slots__ are accepted (supported via special serialization)."""
    @remote
    class SlottedClass:
        __slots__ = ['x', 'y']

        def __init__(self, x, y):
            self.x = x
            self.y = y

    # Should be successfully decorated
    assert hasattr(SlottedClass, '_remote_validated')
    assert SlottedClass._remote_validated is True
    assert hasattr(SlottedClass, '_remote_source')
    assert '__slots__' in SlottedClass._remote_source


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
    # Source is now a dict with code, file, line
    source_data = result["normalize"]["source"]
    assert isinstance(source_data, dict)
    assert "def normalize" in source_data["code"]


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
    # Source is now a dict with code, file, line
    source_data = result["Analyzer"]["source"]
    assert isinstance(source_data, dict)
    assert "class Analyzer" in source_data["code"]


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
    assert not nested_func._remote_source.startswith(" "), "Source should not start with space"
    assert not nested_func._remote_source.startswith("\t"), "Source should not start with tab"

    # Source should start with a valid Python declaration (decorator or def)
    first_line = nested_func._remote_source.split('\n')[0]
    assert first_line.startswith('@') or first_line.startswith('def '), (
        f"First line should start with '@' or 'def ', got: {first_line!r}"
    )


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
    # Error should mention the pandas import
    assert "pandas" in error_msg, f"Error should mention pandas: {error_msg}"
    # Error should include line number information
    assert "Line" in error_msg, f"Error should include line number: {error_msg}"


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
    # Should mention the class name
    assert "_NotSerializableClass" in error_msg, f"Error should mention the class: {error_msg}"
    # Should indicate it's not serializable
    assert "not JSON-serializable" in error_msg, f"Error should say not JSON-serializable: {error_msg}"
    # Should suggest alternatives
    assert "Options" in error_msg, f"Error should suggest options: {error_msg}"


# =============================================================================
# Test: Allowed attribute chains (os.path.join, pathlib.Path)
# =============================================================================

def test_remote_allows_os_path_join():
    """Test that os.path.join is allowed."""
    @remote
    def use_os_path():
        import os
        return os.path.join("foo", "bar", "baz.txt")

    assert use_os_path._remote_validated is True


def test_remote_allows_os_path_dirname():
    """Test that os.path.dirname is allowed."""
    @remote
    def use_os_dirname():
        import os
        return os.path.dirname("/foo/bar/baz.txt")

    assert use_os_dirname._remote_validated is True


def test_remote_allows_pathlib_path_constructor():
    """Test that pathlib.Path constructor is allowed."""
    @remote
    def use_pathlib():
        from pathlib import Path
        p = Path("foo") / "bar" / "baz.txt"
        return str(p)

    assert use_pathlib._remote_validated is True


def test_remote_allows_pathlib_path_methods():
    """Test that safe pathlib.Path methods are allowed."""
    @remote
    def use_pathlib_methods():
        from pathlib import Path
        p = Path("/foo/bar/baz.txt")
        return p.stem, p.suffix, p.parent, p.name

    assert use_pathlib_methods._remote_validated is True


def test_remote_rejects_os_system():
    """Test that os.system is blocked."""
    with pytest.raises(RemoteValidationError) as exc_info:
        @remote
        def use_os_system():
            import os
            return os.system("ls")

    assert "os.system" in str(exc_info.value)


def test_remote_rejects_pathlib_read_text():
    """Test that pathlib.Path.read_text via full path is blocked."""
    # Note: Using `from pathlib import Path; Path().read_text()` currently bypasses
    # the pattern check because the AST chain is just ('read_text',) not ('pathlib', 'Path', 'read_text').
    # This test verifies the full pathlib.Path.read_text() pattern is blocked.
    with pytest.raises(RemoteValidationError) as exc_info:
        @remote
        def use_pathlib_read():
            import pathlib
            return pathlib.Path("file.txt").read_text()

    assert "read_text" in str(exc_info.value)


def test_remote_rejects_subprocess():
    """Test that subprocess is blocked."""
    with pytest.raises(RemoteValidationError) as exc_info:
        @remote
        def use_subprocess():
            import subprocess
            return subprocess.run(["ls"])

    assert "subprocess" in str(exc_info.value)


def test_remote_allows_random():
    """Test that random module is allowed."""
    @remote
    def use_random():
        import random
        return random.randint(1, 100)

    assert use_random._remote_validated is True


# =============================================================================
# Test: Closure variable support
# =============================================================================

def test_remote_captures_closure_primitives():
    """Test that primitive closure variables are captured."""
    def make_func():
        captured_value = 42
        captured_name = "test"

        @remote
        def inner():
            return captured_value, captured_name

        return inner

    func = make_func()
    assert func._remote_validated is True

    # Closure variables should be in _remote_closure_vars
    closure_vars = getattr(func, '_remote_closure_vars', {})
    assert 'captured_value' in closure_vars, f"captured_value should be in closure_vars: {closure_vars}"
    assert 'captured_name' in closure_vars, f"captured_name should be in closure_vars: {closure_vars}"
    assert closure_vars['captured_value'] == 42
    assert closure_vars['captured_name'] == "test"


def test_remote_captures_closure_list():
    """Test that list closure variables are captured."""
    def make_func():
        items = [1, 2, 3, 4, 5]

        @remote
        def inner():
            return sum(items)

        return inner

    func = make_func()
    assert func._remote_validated is True

    # List should be captured in closure_vars
    closure_vars = getattr(func, '_remote_closure_vars', {})
    assert 'items' in closure_vars, f"items should be in closure_vars: {closure_vars}"
    assert closure_vars['items'] == [1, 2, 3, 4, 5]


def test_remote_rejects_non_serializable_closure():
    """Test that non-serializable closure variables are rejected."""
    class NonSerializable:
        pass

    obj = NonSerializable()

    with pytest.raises(RemoteValidationError) as exc_info:
        def make_func():
            @remote
            def inner():
                return obj
            return inner
        make_func()

    error_msg = str(exc_info.value)
    # Error should mention it's a closure variable
    assert "closure variable" in error_msg.lower(), f"Error should mention closure variable: {error_msg}"
    # Error should mention it's not serializable
    assert "not JSON-serializable" in error_msg, f"Error should say not JSON-serializable: {error_msg}"


# =============================================================================
# Test: Builtin override detection
# =============================================================================

def test_remote_detects_overridden_builtin_closure():
    """Test that overridden builtins in closure scope are detected and captured.

    Python allows shadowing builtins with local variables. The @remote decorator
    must detect when a builtin name refers to a different value and capture it.
    """
    # Override 'len' with a constant in local scope
    len = 42  # Shadow the builtin

    def make_func():
        @remote
        def inner(x):
            # This 'len' refers to the local override, not the builtin
            return len

        return inner

    func = make_func()
    assert func._remote_validated is True

    # The override should be captured
    closure_vars = getattr(func, '_remote_closure_vars', {})
    assert 'len' in closure_vars
    assert closure_vars['len'] == 42


def test_remote_skips_unmodified_builtin():
    """Test that builtins are skipped when NOT overridden.

    If a name refers to the actual builtin value, it should not be captured
    since builtins are available by default during execution.
    """
    @remote
    def uses_builtins(x):
        # These are actual builtins, not overridden
        return len(x) + sum(x)

    # No builtins should be captured (they're available by default)
    module_refs = getattr(uses_builtins, '_remote_module_refs', {})
    closure_vars = getattr(uses_builtins, '_remote_closure_vars', {})

    assert 'len' not in module_refs
    assert 'len' not in closure_vars
    assert 'sum' not in module_refs
    assert 'sum' not in closure_vars


# =============================================================================
# Test: File/line metadata
# =============================================================================

from nnsight.intervention.serialization_source import extract_source_metadata


def test_source_metadata_extraction():
    """Test that source metadata includes file and line info."""
    # We can't easily test this without a real tracer, but we can test the format
    # This would require mocking a tracer object
    pass  # Covered by integration tests


def test_extract_remote_object_includes_file_line():
    """Test that remote object extraction includes file/line metadata."""
    @remote
    class TestClass:
        def __init__(self):
            self.x = 1

    result = {}
    extract_remote_object("TestClass", TestClass, result)

    # Should have source as dict with code, file, line
    assert "TestClass" in result
    source_data = result["TestClass"]["source"]
    assert isinstance(source_data, dict)
    assert "code" in source_data
    assert "file" in source_data
    assert "line" in source_data
    assert "class TestClass" in source_data["code"]


# =============================================================================
# Test: Cycle detection
# =============================================================================

def test_serialize_handles_self_reference():
    """Test that self-references are handled via reference (not infinite recursion)."""
    @remote
    class Node:
        def __init__(self, value):
            self.value = value
            self.next = None

    # Create self-reference (direct cycle)
    node = Node(1)
    node.self_ref = node  # Self-reference

    # With reference-based serialization, this should work (not infinite loop)
    state = serialize_instance_state(node)

    # Non-nn.Module @remote objects use __remote_ref__ for references
    assert "__remote_ref__" in state.get("self_ref", {}), "Self-reference should use __remote_ref__"
    assert state["value"] == 1


def test_serialize_deduplicates_shared_tensors():
    """Test that shared tensor references are deduplicated and restored correctly."""
    import json
    from nnsight.intervention.serialization_source import reconstruct_state

    @remote
    class SharedTensorHolder:
        def __init__(self):
            pass

    # Create object with shared tensor reference
    obj = SharedTensorHolder()
    shared_tensor = torch.randn(10)
    obj.a = shared_tensor
    obj.b = shared_tensor  # Same reference

    # Verify original has identity
    assert obj.a is obj.b, "Original should have shared reference"

    # Serialize
    state = serialize_instance_state(obj)
    json_str = json.dumps(state)

    # One should be full tensor, other should be __ref__
    tensor_count = json_str.count('"__tensor__"')
    ref_count = json_str.count('"__ref__"')
    assert tensor_count == 1, f"Should have exactly 1 tensor serialization, got {tensor_count}"
    assert ref_count == 1, f"Should have exactly 1 reference, got {ref_count}"

    # Deserialize and verify identity is preserved
    state_back = json.loads(json_str)
    restored = object.__new__(SharedTensorHolder)
    restored.__dict__ = reconstruct_state(state_back, {}, None, {})

    assert restored.a is restored.b, "Restored should preserve shared reference"
    assert torch.allclose(obj.a, restored.a), "Tensor values should match"


def test_serialize_handles_remote_references():
    """Test that indirect remote object references work without cycle detection issues."""
    # This test verifies that referencing other @remote objects via __remote_ref__
    # doesn't cause cycle issues - they're stored as references, not fully serialized
    @remote
    class Node:
        def __init__(self, value):
            self.value = value
            self.next = None

    node1 = Node(1)
    node2 = Node(2)
    node1.next = node2
    node2.next = node1  # Indirect cycle through references

    # This should work - node2 is stored as __remote_ref__, not recursively serialized
    state = serialize_instance_state(node1)
    assert state["value"] == 1
    assert "__remote_ref__" in state["next"]


def test_serialize_allows_shared_reference():
    """Test that shared (non-circular) references work."""
    @remote
    class Container:
        def __init__(self, value):
            self.value = value

    @remote
    class Parent:
        def __init__(self, a, b):
            self.a = a
            self.b = b

    shared = Container(42)
    parent = Parent(shared, shared)  # Same object referenced twice

    # This should NOT raise (no circular reference, just shared)
    state = serialize_instance_state(parent)
    assert "__remote_ref__" in state["a"]
    assert "__remote_ref__" in state["b"]


# =============================================================================
# Test: nn.Module and Dataset training flow (integration)
# =============================================================================

from nnsight.intervention.serialization_source import (
    reconstruct_state,
)


def test_nn_module_subclass_roundtrip():
    """Test that @remote nn.Module subclasses serialize and deserialize correctly."""
    import torch.nn as nn

    @remote
    class ProbeClassifier(nn.Module):
        """Linear probe with optional bias and temperature."""

        def __init__(self, hidden_size, num_classes, use_bias=True, temperature=1.0):
            super().__init__()
            self.linear = nn.Linear(hidden_size, num_classes, bias=use_bias)
            self.temperature = temperature
            self.trained = False

        def forward(self, hidden_states):
            logits = self.linear(hidden_states) / self.temperature
            return logits

        def predict(self, hidden_states):
            return self.forward(hidden_states).argmax(dim=-1)

        def mark_trained(self):
            self.trained = True

    # Create classifier
    classifier = ProbeClassifier(768, 3, temperature=0.5)
    assert classifier.trained is False

    # Serialize
    state = serialize_instance_state(classifier)
    json_str = json.dumps(state)

    # Deserialize
    restored = object.__new__(ProbeClassifier)
    restored.__dict__ = reconstruct_state(json.loads(json_str), {}, None, {})

    # Verify state
    assert restored.temperature == 0.5
    assert restored.trained is False

    # Verify forward works
    x = torch.randn(2, 768)
    y = restored(x)
    assert y.shape == (2, 3)


def test_dataset_subclass_roundtrip():
    """Test that @remote dataset-like classes serialize and deserialize correctly."""

    @remote
    class ClassifiedStringsDataset:
        """Custom dataset holding classified strings in memory."""

        def __init__(self, strings, labels):
            self.strings = strings
            self.labels = torch.tensor(labels)

        def __len__(self):
            return len(self.strings)

        def __getitem__(self, idx):
            return self.strings[idx], self.labels[idx]

        def get_all_labels(self):
            return self.labels

    # Create dataset
    strings = [f"Example sentence {i}" for i in range(100)]
    labels = [i % 3 for i in range(100)]
    dataset = ClassifiedStringsDataset(strings, labels)

    # Serialize
    state = serialize_instance_state(dataset)
    json_str = json.dumps(state)

    # Deserialize
    restored = object.__new__(ClassifiedStringsDataset)
    restored.__dict__ = reconstruct_state(json.loads(json_str), {}, None, {})

    # Verify state
    assert len(restored) == 100
    assert restored.strings[0] == "Example sentence 0"
    assert torch.equal(restored.labels, dataset.labels)


def test_training_flow_simulation():
    """Test full training flow: serialize, train on 'server', return trained model."""
    import torch.nn as nn

    @remote
    class ProbeClassifier(nn.Module):
        def __init__(self, hidden_size, num_classes):
            super().__init__()
            self.linear = nn.Linear(hidden_size, num_classes)
            self.trained = False

        def forward(self, x):
            return self.linear(x)

        def mark_trained(self):
            self.trained = True

    # 1. CLIENT: Create classifier
    classifier = ProbeClassifier(64, 3)
    original_weight = classifier.linear.weight[0, 0].item()
    assert classifier.trained is False

    # 2. CLIENT: Serialize
    state = serialize_instance_state(classifier)
    wire_data = json.dumps(state)

    # 3. SERVER: Reconstruct
    server_classifier = object.__new__(ProbeClassifier)
    server_classifier.__dict__ = reconstruct_state(
        json.loads(wire_data), {}, None, {}
    )

    # 4. SERVER: Train (simulate)
    fake_data = torch.randn(100, 64)
    fake_labels = torch.randint(0, 3, (100,))
    optimizer = torch.optim.Adam(server_classifier.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    for _ in range(10):
        optimizer.zero_grad()
        logits = server_classifier(fake_data)
        loss = criterion(logits, fake_labels)
        loss.backward()
        optimizer.step()

    server_classifier.mark_trained()

    # 5. SERVER: Serialize trained model
    trained_state = serialize_instance_state(server_classifier)
    trained_wire = json.dumps(trained_state)

    # 6. CLIENT: Receive trained model
    client_classifier = object.__new__(ProbeClassifier)
    client_classifier.__dict__ = reconstruct_state(
        json.loads(trained_wire), {}, None, {}
    )

    # 7. Verify training happened
    assert client_classifier.trained is True
    trained_weight = client_classifier.linear.weight[0, 0].item()
    assert trained_weight != original_weight, "Weights should have changed after training"

    # Verify weights match server
    assert torch.allclose(
        server_classifier.linear.weight,
        client_classifier.linear.weight
    ), "Client and server weights should match"


# =============================================================================
# Test: Improved callable reference detection
# =============================================================================

from nnsight.intervention.serialization_source import get_callable_reference


def test_get_callable_reference_torch_function():
    """Test callable reference for torch functions."""
    ref = get_callable_reference(torch.relu)
    assert ref is not None
    assert "torch" in ref
    assert "relu" in ref


def test_get_callable_reference_numpy_function():
    """Test callable reference for numpy functions."""
    ref = get_callable_reference(np.mean)
    assert ref is not None
    assert "numpy" in ref
    assert "mean" in ref


def test_get_callable_reference_rejects_custom():
    """Test that custom functions return None."""
    def my_func():
        pass

    ref = get_callable_reference(my_func)
    assert ref is None


def test_get_callable_reference_math():
    """Test callable reference for math functions."""
    import math
    ref = get_callable_reference(math.sqrt)
    assert ref is not None
    assert "math" in ref
    assert "sqrt" in ref


def test_serialize_instance_with_callable():
    """Test serializing instance with callable attribute."""
    @remote
    class WithCallable:
        def __init__(self):
            self.activation = torch.relu

    instance = WithCallable()
    state = serialize_instance_state(instance)

    assert "__callable_ref__" in state["activation"]
    assert "relu" in state["activation"]["__callable_ref__"]


# =============================================================================
# Test: Deserialization with new modules
# =============================================================================

from nnsight.intervention.serialization_source import deserialize_source_based


def test_deserialize_includes_os_pathlib_random():
    """Test that deserialized namespace includes os, pathlib, random."""
    payload = json.dumps({
        "version": "2.1",
        "source": {"code": "", "file": "test.py", "line": 1},
        "variables": {},
        "remote_objects": {},
        "model_refs": [],
    }).encode('utf-8')

    # Mock model
    class MockModel:
        pass

    namespace = deserialize_source_based(payload, MockModel())

    assert 'os' in namespace
    assert 'pathlib' in namespace
    assert 'random' in namespace
    assert 'torch' in namespace
    assert 'numpy' in namespace


def test_deserialize_handles_legacy_format():
    """Test that deserialization handles legacy v2.0 format."""
    payload = json.dumps({
        "version": "2.0",
        "source": "pass",  # Legacy: just a string
        "variables": {"x": 42},
        "remote_objects": {},
        "model_refs": [],
    }).encode('utf-8')

    class MockModel:
        pass

    namespace = deserialize_source_based(payload, MockModel())

    assert namespace['x'] == 42
    assert namespace['__nnsight_source_code__'] == "pass"


def test_deserialize_includes_closure_vars():
    """Test that deserialization includes closure variables."""
    payload = json.dumps({
        "version": "2.1",
        "source": {"code": "", "file": "test.py", "line": 1},
        "variables": {},
        "remote_objects": {
            "my_func": {
                "source": {"code": "def my_func(): return closure_var", "file": "test.py", "line": 1},
                "module_refs": {},
                "closure_vars": {"closure_var": 99},
                "type": "function",
                "instances": {},
                "library": None,
                "version": None,
            }
        },
        "model_refs": [],
    }).encode('utf-8')

    class MockModel:
        pass

    namespace = deserialize_source_based(payload, MockModel())

    # The closure_var should be in namespace
    assert 'closure_var' in namespace
    assert namespace['closure_var'] == 99


# =============================================================================
# Test: Lambda source extraction
# =============================================================================

from nnsight.remote import (
    extract_lambda_source, LambdaExtractionError, is_lambda, validate_lambda_for_remote
)


def test_is_lambda_detection():
    """Test that is_lambda correctly identifies lambdas."""
    assert is_lambda(lambda x: x) is True
    assert is_lambda(lambda: 42) is True

    def named_func():
        pass
    assert is_lambda(named_func) is False
    assert is_lambda(42) is False
    assert is_lambda("not a function") is False


def test_extract_lambda_single():
    """Test extracting a single lambda on a line."""
    simple = lambda x: x * 2
    source = extract_lambda_source(simple)
    assert source == "lambda x: x * 2"


def test_extract_lambda_with_expression():
    """Test extracting lambda with complex expression."""
    conditional = lambda x: "yes" if x > 0 else "no"
    source = extract_lambda_source(conditional)
    assert "lambda x:" in source
    assert "if x > 0" in source


def test_extract_multiple_lambdas_same_line():
    """Test extracting the correct lambda when multiple are on same line."""
    f1, f2 = lambda x: x + 1, lambda x: x - 1
    source1 = extract_lambda_source(f1)
    source2 = extract_lambda_source(f2)

    assert source1 == "lambda x: x + 1"
    assert source2 == "lambda x: x - 1"
    assert source1 != source2


def test_extract_lambdas_in_dict():
    """Test extracting lambdas from dict values."""
    ops = {'add': lambda a, b: a + b, 'mul': lambda a, b: a * b}

    add_source = extract_lambda_source(ops['add'])
    mul_source = extract_lambda_source(ops['mul'])

    assert "a + b" in add_source
    assert "a * b" in mul_source


def test_extract_lambda_with_default_args():
    """Test extracting lambda with default arguments."""
    with_default = lambda x, y=10: x + y
    source = extract_lambda_source(with_default)

    # Normalize whitespace for comparison
    normalized = source.replace(" ", "")
    assert "y=10" in normalized, f"Default arg y=10 should be in source: {source}"
    assert "x+y" in normalized, f"Body x+y should be in source: {source}"


def test_validate_lambda_valid():
    """Test validation passes for valid lambda."""
    valid = lambda x: x * 2 + 1
    source, errors = validate_lambda_for_remote(valid)
    assert source == "lambda x: x * 2 + 1"
    assert errors == []


def test_validate_lambda_with_torch():
    """Test validation allows torch usage in lambda."""
    torch_lambda = lambda x: torch.relu(x)
    source, errors = validate_lambda_for_remote(torch_lambda)
    assert "torch.relu" in source
    assert errors == []


def test_lambda_rejects_disallowed_call():
    """Test that lambda with disallowed calls is rejected during validation."""
    # Lambda that calls open() should be rejected
    bad_lambda = lambda path: open(path).read()

    source, errors = validate_lambda_for_remote(bad_lambda)

    # Should have validation errors
    assert len(errors) > 0, f"Lambda with open() should be rejected, got errors: {errors}"
    assert any("open" in err for err in errors), f"Error should mention 'open': {errors}"


def test_deserialize_lambda():
    """Test that lambdas are correctly deserialized."""
    payload = json.dumps({
        "version": "2.1",
        "source": {"code": "", "file": "test.py", "line": 1},
        "variables": {},
        "remote_objects": {
            "__lambda_123": {
                "source": {"code": "lambda x: x * 2", "file": "test.py", "line": 1},
                "module_refs": {},
                "closure_vars": {},
                "type": "lambda",
                "var_name": "my_lambda",
                "library": None,
                "version": None,
            }
        },
        "model_refs": [],
    }).encode('utf-8')

    class MockModel:
        pass

    namespace = deserialize_source_based(payload, MockModel())

    # The lambda should be in namespace under its var_name
    assert 'my_lambda' in namespace
    assert callable(namespace['my_lambda'])
    assert namespace['my_lambda'](5) == 10


def test_deserialize_lambda_with_closure():
    """Test that lambdas with closure vars are correctly deserialized."""
    payload = json.dumps({
        "version": "2.1",
        "source": {"code": "", "file": "test.py", "line": 1},
        "variables": {},
        "remote_objects": {
            "__lambda_456": {
                "source": {"code": "lambda x: x * multiplier", "file": "test.py", "line": 1},
                "module_refs": {},
                "closure_vars": {"multiplier": 5},
                "type": "lambda",
                "var_name": "scaled",
                "library": None,
                "version": None,
            }
        },
        "model_refs": [],
    }).encode('utf-8')

    class MockModel:
        pass

    namespace = deserialize_source_based(payload, MockModel())

    assert 'scaled' in namespace
    assert 'multiplier' in namespace
    assert namespace['multiplier'] == 5
    assert namespace['scaled'](10) == 50


# Test error messages for unsupported lambda patterns

def test_lambda_extraction_error_non_lambda():
    """Test that non-lambdas raise appropriate error."""
    def named():
        pass

    with pytest.raises(LambdaExtractionError) as exc_info:
        extract_lambda_source(named)

    assert "Expected a lambda" in str(exc_info.value)


# =============================================================================
# Test: Lambda closure variable handling
# =============================================================================

from nnsight.intervention.serialization_source import extract_lambda_object


def test_lambda_closure_json_serializable():
    """Test that JSON-serializable closure variables are captured."""
    def make_scaler(factor):
        return lambda x: x * factor

    scale_by_5 = make_scaler(5)

    # Verify the closure is captured
    assert scale_by_5.__closure__ is not None
    assert scale_by_5.__code__.co_freevars == ('factor',)
    assert scale_by_5.__closure__[0].cell_contents == 5

    # Serialize and check the payload
    result = {}
    extract_lambda_object("scale_by_5", scale_by_5, result)

    # Find the lambda entry (key is __lambda_<id>)
    lambda_key = list(result.keys())[0]
    lambda_data = result[lambda_key]

    assert lambda_data["type"] == "lambda"
    assert lambda_data["var_name"] == "scale_by_5"
    assert lambda_data["closure_vars"] == {"factor": 5}
    assert "lambda x: x * factor" in lambda_data["source"]["code"]


def test_lambda_closure_multiple_values():
    """Test lambda capturing multiple closure variables."""
    def make_affine(scale, offset):
        return lambda x: x * scale + offset

    affine = make_affine(2, 10)

    result = {}
    extract_lambda_object("affine", affine, result)

    lambda_key = list(result.keys())[0]
    lambda_data = result[lambda_key]

    assert lambda_data["closure_vars"] == {"scale": 2, "offset": 10}


def test_lambda_closure_list_and_dict():
    """Test lambda capturing list and dict closure variables."""
    def make_processor(config, weights):
        return lambda x: sum(w * x for w in weights) if config["enabled"] else 0

    processor = make_processor({"enabled": True}, [1, 2, 3])

    result = {}
    extract_lambda_object("processor", processor, result)

    lambda_key = list(result.keys())[0]
    lambda_data = result[lambda_key]

    assert lambda_data["closure_vars"]["config"] == {"enabled": True}
    assert lambda_data["closure_vars"]["weights"] == [1, 2, 3]


def test_lambda_closure_torch_module_skipped():
    """Test that torch module in closure is skipped (available on server)."""
    import torch

    # torch is captured in closure but should be skipped
    normalize = lambda x: torch.nn.functional.normalize(x, dim=-1)

    result = {}
    extract_lambda_object("normalize", normalize, result)

    lambda_key = list(result.keys())[0]
    lambda_data = result[lambda_key]

    # torch should NOT be in closure_vars (it's available on server)
    assert "torch" not in lambda_data["closure_vars"]


def test_lambda_closure_numpy_function_skipped():
    """Test that numpy functions in closure are skipped."""
    mean_fn = np.mean

    compute = lambda x: mean_fn(x)

    result = {}
    extract_lambda_object("compute", compute, result)

    lambda_key = list(result.keys())[0]
    lambda_data = result[lambda_key]

    # mean_fn should NOT be in closure_vars (numpy is available on server)
    assert "mean_fn" not in lambda_data["closure_vars"]


def test_lambda_closure_tensor_raises_error():
    """Test that tensor in closure raises clear error."""
    import torch

    def make_biased(bias_tensor):
        return lambda x: x + bias_tensor

    bias = torch.tensor([1.0, 2.0, 3.0])
    biased_func = make_biased(bias)

    result = {}
    with pytest.raises(SourceSerializationError) as exc_info:
        extract_lambda_object("biased_func", biased_func, result)

    error_msg = str(exc_info.value)
    assert "bias_tensor" in error_msg
    assert "Tensor" in error_msg
    assert "cannot be serialized" in error_msg


def test_lambda_closure_nested_lambda_raises_error():
    """Test that nested lambda in closure raises clear error."""
    outer = lambda x: x + 1

    def make_combined(transform):
        return lambda x: transform(x) * 2

    combined = make_combined(outer)

    result = {}
    with pytest.raises(SourceSerializationError) as exc_info:
        extract_lambda_object("combined", combined, result)

    error_msg = str(exc_info.value)
    assert "captures another lambda" in error_msg
    assert "Nested lambdas are not supported" in error_msg


def test_lambda_closure_custom_object_raises_error():
    """Test that custom non-serializable objects raise clear error."""
    class CustomProcessor:
        def process(self, x):
            return x * 2

    processor = CustomProcessor()

    def make_with_processor(proc):
        return lambda x: proc.process(x)

    func = make_with_processor(processor)

    result = {}
    with pytest.raises(SourceSerializationError) as exc_info:
        extract_lambda_object("func", func, result)

    error_msg = str(exc_info.value)
    assert "proc" in error_msg
    assert "CustomProcessor" in error_msg
    assert "cannot be serialized" in error_msg


def test_lambda_closure_remote_object_skipped():
    """Test that @nnsight.remote objects in closure are skipped (serialized separately)."""
    @remote
    class Helper:
        def process(self, x):
            return x * 2

    helper = Helper()

    # Lambda capturing a @remote instance
    def make_processor(h):
        return lambda x: h.process(x)

    processor = make_processor(helper)

    result = {}
    extract_lambda_object("processor", processor, result)

    lambda_key = list(result.keys())[0]
    lambda_data = result[lambda_key]

    # helper should NOT be in closure_vars (it's a @remote object, serialized separately)
    assert "h" not in lambda_data["closure_vars"]


def test_lambda_full_roundtrip_with_closure():
    """Test full serialize/deserialize roundtrip with closure variables."""
    # Create a lambda with closure
    def make_scaler(factor, offset):
        return lambda x: x * factor + offset

    my_scaler = make_scaler(3, 10)

    # Serialize
    result = {}
    extract_lambda_object("my_scaler", my_scaler, result)

    lambda_key = list(result.keys())[0]
    lambda_data = result[lambda_key]

    # Build the payload
    payload = json.dumps({
        "version": "2.1",
        "source": {"code": "", "file": "test.py", "line": 1},
        "variables": {},
        "remote_objects": {
            lambda_key: lambda_data
        },
        "model_refs": [],
    }).encode('utf-8')

    # Deserialize
    class MockModel:
        pass

    namespace = deserialize_source_based(payload, MockModel())

    # Verify the lambda works correctly
    assert "my_scaler" in namespace
    assert callable(namespace["my_scaler"])
    assert namespace["my_scaler"](5) == 25  # 5 * 3 + 10 = 25
    assert namespace["my_scaler"](0) == 10  # 0 * 3 + 10 = 10


# =============================================================================
# Test: Tensor Serialization
# =============================================================================

from nnsight.intervention.serialization_source import (
    is_tensor,
    serialize_tensor,
    deserialize_tensor,
    extract_all,
)


def test_is_tensor_torch():
    """Test is_tensor detects torch.Tensor."""
    t = torch.randn(10, 20)
    assert is_tensor(t) is True


def test_is_tensor_numpy():
    """Test is_tensor detects numpy.ndarray."""
    a = np.random.randn(10, 20)
    assert is_tensor(a) is True


def test_is_tensor_other():
    """Test is_tensor returns False for non-tensors."""
    assert is_tensor([1, 2, 3]) is False
    assert is_tensor(42) is False
    assert is_tensor("hello") is False
    assert is_tensor({"a": 1}) is False


def test_serialize_tensor_basic():
    """Test basic tensor serialization."""
    t = torch.tensor([1.0, 2.0, 3.0])
    result = serialize_tensor(t)

    assert "__tensor__" in result
    assert result["dtype"] == "float32"
    assert result["shape"] == [3]


def test_serialize_tensor_2d():
    """Test 2D tensor serialization."""
    t = torch.randn(100, 50)
    result = serialize_tensor(t)

    assert result["shape"] == [100, 50]
    assert result["dtype"] == "float32"


def test_serialize_tensor_numpy():
    """Test numpy array serialization."""
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    result = serialize_tensor(a)

    assert result["shape"] == [2, 2]
    assert result["dtype"] == "int32"


def test_deserialize_tensor_torch():
    """Test tensor deserialization to torch."""
    original = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    serialized = serialize_tensor(original)
    restored = deserialize_tensor(serialized, as_torch=True)

    assert isinstance(restored, torch.Tensor)
    assert torch.allclose(original, restored)


def test_deserialize_tensor_numpy():
    """Test tensor deserialization to numpy."""
    original = np.array([1.0, 2.0, 3.0])
    serialized = serialize_tensor(original)
    restored = deserialize_tensor(serialized, as_torch=False)

    assert isinstance(restored, np.ndarray)
    assert np.allclose(original, restored)


def test_tensor_roundtrip_random():
    """Test roundtrip for random float tensor."""
    original = torch.randn(100, 50)
    serialized = serialize_tensor(original)
    restored = deserialize_tensor(serialized)

    assert torch.allclose(original, restored)


def test_tensor_roundtrip_preserves_dtype():
    """Test that dtype is preserved through roundtrip."""
    for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
        original = torch.tensor([1, 2, 3], dtype=dtype)
        serialized = serialize_tensor(original)
        restored = deserialize_tensor(serialized)
        assert restored.dtype == dtype


def test_tensor_in_extract_all():
    """Test that tensors are handled by extract_all."""
    my_tensor = torch.tensor([1.0, 2.0, 3.0])
    my_int = 42

    variables, remote_objects, model_refs = extract_all({
        "my_tensor": my_tensor,
        "my_int": my_int,
    })

    # Tensor should be serialized in variables
    assert "my_tensor" in variables
    assert "__tensor__" in variables["my_tensor"]

    # Int should be plain
    assert variables["my_int"] == 42


def test_tensor_in_instance_state():
    """Test that tensors in @remote instance state are serialized."""
    from nnsight.intervention.serialization_source import serialize_instance_state

    @remote
    class TensorHolder:
        def __init__(self, data):
            self.data = data
            self.scale = 2.0

    holder = TensorHolder(torch.tensor([1.0, 2.0]))
    state = serialize_instance_state(holder)

    # data should be a tensor dict
    assert "__tensor__" in state["data"]
    # scale should be plain
    assert state["scale"] == 2.0


def test_tensor_full_roundtrip():
    """Test full serialize/deserialize roundtrip with tensor variable."""
    original = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    # Serialize
    variables, remote_objects, model_refs = extract_all({
        "my_vec": original,
        "factor": 2,
    })

    # Build payload
    payload = json.dumps({
        "version": "2.1",
        "source": {"code": "", "file": "test.py", "line": 1},
        "variables": variables,
        "remote_objects": remote_objects,
        "model_refs": model_refs,
    }).encode('utf-8')

    # Deserialize
    class MockModel:
        pass

    namespace = deserialize_source_based(payload, MockModel())

    # Verify tensor is restored correctly
    assert "my_vec" in namespace
    assert isinstance(namespace["my_vec"], torch.Tensor)
    assert torch.allclose(namespace["my_vec"], original)
    assert namespace["factor"] == 2


def test_tensor_sparse_pytorch():
    """Test that PyTorch sparse tensors are preserved."""
    # Create a sparse COO tensor
    indices = torch.tensor([[0, 1, 2], [0, 1, 2]])
    values = torch.tensor([1.0, 2.0, 3.0])
    sparse = torch.sparse_coo_tensor(indices, values, (3, 3))

    assert sparse.is_sparse

    # Serialize - should preserve sparse format
    serialized = serialize_tensor(sparse)

    # Check sparse info is in the payload
    assert "sparse" in serialized
    assert serialized["sparse"]["dense_shape"] == [3, 3]

    # Deserialize - should restore as sparse tensor
    restored = deserialize_tensor(serialized)

    assert restored.is_sparse
    assert restored.shape == sparse.shape

    # Check values match exactly
    assert torch.equal(restored.to_dense(), sparse.to_dense())


def test_tensor_bfloat16():
    """Test that bfloat16 tensors are preserved exactly."""
    bf16 = torch.randn(10, 10, dtype=torch.bfloat16)

    # Serialize - should store as int16 with torch_dtype marker
    serialized = serialize_tensor(bf16)
    assert serialized["dtype"] == "int16"  # Stored as int16 (same bit pattern)
    assert serialized["torch_dtype"] == "bfloat16"

    # Deserialize
    restored = deserialize_tensor(serialized)

    # Should restore as bfloat16, values exactly equal
    assert restored.dtype == torch.bfloat16
    assert torch.equal(bf16, restored)


def test_tensor_nested_raises_error():
    """Test that nested tensors raise a clear error."""
    from nnsight.intervention.serialization_source import SourceSerializationError

    nested = torch.nested.nested_tensor([torch.randn(3), torch.randn(5)])

    with pytest.raises(SourceSerializationError, match="Nested tensors"):
        serialize_tensor(nested)


def test_tensor_quantized():
    """Test that quantized tensors are preserved exactly."""
    # Create a quantized tensor
    float_tensor = torch.randn(10, 10)
    quant_tensor = torch.quantize_per_tensor(
        float_tensor, scale=0.1, zero_point=10, dtype=torch.qint8
    )

    assert quant_tensor.is_quantized

    # Serialize - should preserve quantization info
    serialized = serialize_tensor(quant_tensor)

    # Check quantization info is in the payload
    assert "quantization" in serialized
    assert serialized["quantization"]["scale"] == 0.1
    assert serialized["quantization"]["zero_point"] == 10
    assert serialized["quantization"]["qtype"] == "qint8"

    # Deserialize - should restore as quantized tensor
    restored = deserialize_tensor(serialized)

    assert restored.is_quantized
    assert restored.q_scale() == quant_tensor.q_scale()
    assert restored.q_zero_point() == quant_tensor.q_zero_point()

    # Check underlying int values are exactly equal
    assert torch.equal(restored.int_repr(), quant_tensor.int_repr())


# =============================================================================
# Test: Restricted Execution
# =============================================================================

# Skip these tests if RestrictedPython is not installed
try:
    from RestrictedPython import compile_restricted
    HAS_RESTRICTED_PYTHON = True
except ImportError:
    HAS_RESTRICTED_PYTHON = False

from nnsight.intervention.restricted_execution import (
    SecurityAuditError,
    create_guarded_getattr,
    create_guarded_getitem,
    create_guarded_import,
    create_restricted_globals,
    execute_restricted,
    is_restricted_python_available,
    SUSPICIOUS_ATTRS,
    BLOCKED_MODULES,
    DEFAULT_ALLOWED_MODULES,
)


@pytest.mark.skipif(not HAS_RESTRICTED_PYTHON, reason="RestrictedPython not installed")
def test_restricted_execution_normal_code():
    """Test that normal code executes correctly in restricted mode."""
    code = """
result = x * 2 + y
squares = [i**2 for i in range(5)]
"""
    globals_dict = {'x': 10, 'y': 5}
    namespace = execute_restricted(
        code=code,
        globals_dict=globals_dict,
        user_id="test_user",
        job_id="test_job",
    )

    assert namespace['result'] == 25
    assert namespace['squares'] == [0, 1, 4, 9, 16]


@pytest.mark.skipif(not HAS_RESTRICTED_PYTHON, reason="RestrictedPython not installed")
def test_restricted_execution_with_torch():
    """Test that torch operations work in restricted mode."""
    code = """
import torch
t = torch.tensor([1, 2, 3])
result = t.sum().item()
"""
    namespace = execute_restricted(
        code=code,
        globals_dict={},
        user_id="test_user",
        job_id="test_job",
    )

    assert namespace['result'] == 6


@pytest.mark.skipif(not HAS_RESTRICTED_PYTHON, reason="RestrictedPython not installed")
def test_restricted_execution_blocks_suspicious_getattr():
    """Test that suspicious getattr access is blocked and raises SecurityAuditError."""
    import os
    # Pass os as a pre-imported module so we can test getattr blocking
    code = """
x = getattr(os, 'system')
"""
    with pytest.raises(SecurityAuditError) as exc_info:
        execute_restricted(
            code=code,
            globals_dict={'os': os},
            user_id="test_user",
            job_id="test_job",
        )

    assert "system" in str(exc_info.value)
    assert "not allowed" in str(exc_info.value).lower()


@pytest.mark.skipif(not HAS_RESTRICTED_PYTHON, reason="RestrictedPython not installed")
def test_restricted_execution_blocks_direct_suspicious_attr():
    """Test that direct suspicious attribute access is blocked."""
    import os
    code = """
x = os.system
"""
    with pytest.raises(SecurityAuditError) as exc_info:
        execute_restricted(
            code=code,
            globals_dict={'os': os},
            user_id="test_user",
            job_id="test_job",
        )

    assert "system" in str(exc_info.value)


@pytest.mark.skipif(not HAS_RESTRICTED_PYTHON, reason="RestrictedPython not installed")
def test_restricted_execution_blocks_frame_access():
    """Test that frame access is blocked at compile time."""
    code = """
import sys
frame = sys._getframe()
x = frame.f_back
"""
    # RestrictedPython blocks _getframe and f_back at compile time
    with pytest.raises(SyntaxError):
        execute_restricted(
            code=code,
            globals_dict={},
            user_id="test_user",
            job_id="test_job",
            allowed_modules={'sys'} | DEFAULT_ALLOWED_MODULES,
        )


@pytest.mark.skipif(not HAS_RESTRICTED_PYTHON, reason="RestrictedPython not installed")
def test_restricted_execution_blocks_blocked_import():
    """Test that blocked modules cannot be imported."""
    code = """
import subprocess
"""
    with pytest.raises(SecurityAuditError) as exc_info:
        execute_restricted(
            code=code,
            globals_dict={},
            user_id="test_user",
            job_id="test_job",
        )

    assert "subprocess" in str(exc_info.value)


@pytest.mark.skipif(not HAS_RESTRICTED_PYTHON, reason="RestrictedPython not installed")
def test_restricted_execution_blocks_unauthorized_import():
    """Test that unauthorized modules cannot be imported."""
    code = """
import my_custom_module
"""
    with pytest.raises(SecurityAuditError) as exc_info:
        execute_restricted(
            code=code,
            globals_dict={},
            user_id="test_user",
            job_id="test_job",
        )

    assert "my_custom_module" in str(exc_info.value)


@pytest.mark.skipif(not HAS_RESTRICTED_PYTHON, reason="RestrictedPython not installed")
def test_restricted_execution_blocks_dunder_at_compile():
    """Test that dunder access is blocked at compile time."""
    code = """
x = ().__class__.__bases__[0]
"""
    # RestrictedPython blocks this at compile time
    with pytest.raises(SyntaxError):
        execute_restricted(
            code=code,
            globals_dict={},
            user_id="test_user",
            job_id="test_job",
        )


@pytest.mark.skipif(not HAS_RESTRICTED_PYTHON, reason="RestrictedPython not installed")
def test_restricted_execution_allows_safe_imports():
    """Test that safe modules can be imported."""
    code = """
import math
import functools
import itertools
result = math.sqrt(16)
"""
    namespace = execute_restricted(
        code=code,
        globals_dict={},
        user_id="test_user",
        job_id="test_job",
    )

    assert namespace['result'] == 4.0


def test_guarded_getattr_blocks_suspicious():
    """Test that guarded getattr blocks suspicious attributes."""
    guarded = create_guarded_getattr("user", "job")

    class Dummy:
        safe_attr = 42
        system = "dangerous"

    d = Dummy()

    # Safe access works
    assert guarded(d, 'safe_attr') == 42

    # Suspicious access is blocked
    with pytest.raises(SecurityAuditError):
        guarded(d, 'system')


def test_guarded_getitem_blocks_suspicious():
    """Test that guarded getitem blocks suspicious keys."""
    guarded = create_guarded_getitem("user", "job")

    d = {'safe_key': 42, 'system': 'dangerous'}

    # Safe access works
    assert guarded(d, 'safe_key') == 42

    # Suspicious key access is blocked
    with pytest.raises(SecurityAuditError):
        guarded(d, 'system')


def test_guarded_import_blocks_modules():
    """Test that guarded import blocks unauthorized modules."""
    guarded = create_guarded_import("user", "job", {'math', 'torch'})

    # Allowed module works
    m = guarded('math')
    assert hasattr(m, 'sqrt')

    # Blocked module raises
    with pytest.raises(SecurityAuditError):
        guarded('subprocess')

    # Unauthorized module raises
    with pytest.raises(SecurityAuditError):
        guarded('random')  # Not in allowed set


def test_create_restricted_globals_has_guards():
    """Test that restricted globals include all necessary guards."""
    globals_dict = create_restricted_globals(
        user_id="user",
        job_id="job",
        base_globals={'x': 42},
    )

    # Has guards
    assert '_getattr_' in globals_dict
    assert '_getitem_' in globals_dict
    assert '__import__' in globals_dict
    assert '_getiter_' in globals_dict
    assert '_write_' in globals_dict

    # Has builtins
    assert '__builtins__' in globals_dict
    assert 'len' in globals_dict['__builtins__']

    # Includes base globals
    assert globals_dict['x'] == 42


def test_suspicious_attrs_comprehensive():
    """Test that SUSPICIOUS_ATTRS contains expected dangerous attributes."""
    # Process execution
    assert 'system' in SUSPICIOUS_ATTRS
    assert 'popen' in SUSPICIOUS_ATTRS
    assert 'fork' in SUSPICIOUS_ATTRS

    # Frame inspection
    assert 'f_back' in SUSPICIOUS_ATTRS
    assert 'f_locals' in SUSPICIOUS_ATTRS
    assert 'f_globals' in SUSPICIOUS_ATTRS

    # Code manipulation
    assert '__code__' in SUSPICIOUS_ATTRS
    assert '__globals__' in SUSPICIOUS_ATTRS
    assert '__builtins__' in SUSPICIOUS_ATTRS


def test_blocked_modules_comprehensive():
    """Test that BLOCKED_MODULES contains expected dangerous modules."""
    assert 'subprocess' in BLOCKED_MODULES
    assert 'socket' in BLOCKED_MODULES
    assert 'ctypes' in BLOCKED_MODULES
    assert 'pickle' in BLOCKED_MODULES
    assert 'multiprocessing' in BLOCKED_MODULES


def test_is_restricted_python_available():
    """Test that availability check works."""
    result = is_restricted_python_available()
    assert result == HAS_RESTRICTED_PYTHON


@pytest.mark.skipif(not HAS_RESTRICTED_PYTHON, reason="RestrictedPython not installed")
def test_deserialize_source_based_with_restricted():
    """Test deserialization with restricted execution enabled."""
    payload = json.dumps({
        "version": "2.1",
        "source": {"code": "", "file": "test.py", "line": 1},
        "variables": {"x": 42},
        "remote_objects": {
            "my_func": {
                "source": {"code": "def my_func(a): return a * 2", "file": "test.py", "line": 1},
                "module_refs": {},
                "closure_vars": {},
                "type": "function",
                "instances": {},
                "library": None,
                "version": None,
            }
        },
        "model_refs": [],
    }).encode('utf-8')

    class MockModel:
        pass

    namespace = deserialize_source_based(
        payload,
        MockModel(),
        user_id="test_user",
        job_id="test_job",
        use_restricted=True,
    )

    assert namespace['x'] == 42
    assert 'my_func' in namespace
    assert namespace['my_func'](10) == 20


@pytest.mark.skipif(not HAS_RESTRICTED_PYTHON, reason="RestrictedPython not installed")
def test_deserialize_source_based_restricted_blocks_dangerous():
    """Test that restricted deserialization blocks dangerous code at compile time."""
    payload = json.dumps({
        "version": "2.1",
        "source": {"code": "", "file": "test.py", "line": 1},
        "variables": {},
        "remote_objects": {
            "bad_func": {
                "source": {"code": "def bad_func():\n    import subprocess\n    return subprocess.run(['ls'])", "file": "test.py", "line": 1},
                "module_refs": {},
                "closure_vars": {},
                "type": "function",
                "instances": {},
                "library": None,
                "version": None,
            }
        },
        "model_refs": [],
    }).encode('utf-8')

    class MockModel:
        pass

    # Should fail at deserialization time due to static import analysis
    with pytest.raises(SecurityAuditError) as exc_info:
        deserialize_source_based(
            payload,
            MockModel(),
            user_id="test_user",
            job_id="test_job",
            use_restricted=True,
        )

    assert "subprocess" in str(exc_info.value)
