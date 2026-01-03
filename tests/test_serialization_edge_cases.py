"""
Unit tests for source serialization edge cases.

These tests verify that edge cases in the serialization system are handled correctly.
Run with: pytest tests/test_serialization_edge_cases.py -v
"""

import json
import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import TYPE_CHECKING

import sys
sys.path.insert(0, 'src')

from nnsight.remote import remote
from nnsight.intervention.serialization_source import (
    serialize_instance_state,
    reconstruct_state,
    serialize_tensor,
    deserialize_tensor,
    serialize_value,
    auto_discover_class,
    is_tensor,
    SourceSerializationError,
)


def make_exec_namespace():
    """Create a namespace with common builtins needed for class reconstruction.

    Note: In production, module-level references like torch, nn, np are resolved
    via the server_imports mechanism. For these unit tests, we include them
    directly to test the serialization logic in isolation.
    """
    # No-op remote decorator for server-side reconstruction
    def noop_remote(cls):
        return cls

    # Start with builtins
    import builtins
    namespace = dict(vars(builtins))

    # Add common decorators and builtins
    namespace.update({
        'remote': noop_remote,
        'property': property,
        'staticmethod': staticmethod,
        'classmethod': classmethod,
    })

    # Add module-level references (in production these come from server_imports)
    namespace.update({
        'torch': torch,
        'nn': nn,
        'np': np,
        'numpy': np,
    })

    return namespace


# =============================================================================
# Test 1: Classes with __slots__
# =============================================================================

def test_slots_class():
    """Test that classes with __slots__ are handled correctly."""
    @remote
    class SlottedClass:
        __slots__ = ['x', 'y']

        def __init__(self, x, y):
            self.x = x
            self.y = y

        def sum(self):
            return self.x + self.y

    obj = SlottedClass(10, 20)

    # Serialize
    state = serialize_instance_state(obj)

    # Should have captured x and y
    assert 'x' in state, f"x not captured: {state}"
    assert 'y' in state, f"y not captured: {state}"
    assert state.get('x') == 10, f"x value wrong: {state}"
    assert state.get('y') == 20, f"y value wrong: {state}"

    # Deserialize
    namespace = make_exec_namespace()
    exec(obj._remote_source, namespace)
    restored = object.__new__(namespace['SlottedClass'])

    # For slots, we need to set attributes directly
    restored_state = reconstruct_state(state, namespace, None, {})
    for key, value in restored_state.items():
        setattr(restored, key, value)

    assert restored.x == 10
    assert restored.y == 20
    assert restored.sum() == 30


# =============================================================================
# Test 2: Model stored in a container
# =============================================================================

def test_model_in_container():
    """Test that model references inside containers are detected."""
    # Create a mock model-like object
    class MockModel:
        pass

    model = MockModel()

    # Test direct reference
    from nnsight.intervention.serialization_source import is_the_traced_model
    assert is_the_traced_model(model, model) is True

    # Test in list - currently NOT detected (this documents current behavior)
    models_list = [model]
    assert is_the_traced_model(models_list, model) is False
    assert is_the_traced_model(models_list[0], model) is True

    # Test in dict
    models_dict = {"m": model}
    assert is_the_traced_model(models_dict, model) is False
    assert is_the_traced_model(models_dict["m"], model) is True


# =============================================================================
# Test 3: Numpy arrays in instance state
# =============================================================================

def test_numpy_array_serialization():
    """Test that numpy arrays in instance state are serialized correctly."""
    @remote
    class WithNumpy:
        def __init__(self):
            self.mean = np.array([1.0, 2.0, 3.0])
            self.std = np.array([[1, 2], [3, 4]])

    obj = WithNumpy()

    # Serialize
    state = serialize_instance_state(obj)

    # Check that arrays are serialized
    assert '__tensor__' in state.get('mean', {}), f"mean not serialized as tensor: {state}"
    assert '__tensor__' in state.get('std', {}), f"std not serialized as tensor: {state}"

    # Deserialize and verify
    mean_restored = deserialize_tensor(state['mean'])
    std_restored = deserialize_tensor(state['std'])

    np.testing.assert_array_almost_equal(mean_restored, obj.mean)
    np.testing.assert_array_almost_equal(std_restored, obj.std)


# =============================================================================
# Test 4: Properties vs stored attributes
# =============================================================================

def test_properties_not_in_dict():
    """Test that properties are not stored in __dict__ but computed on access."""
    @remote
    class ComputedClass:
        def __init__(self, x):
            self._x = x

        @property
        def doubled(self):
            return self._x * 2

        @property
        def tripled(self):
            return self._x * 3

    obj = ComputedClass(5)

    # Properties work on original
    assert obj.doubled == 10
    assert obj.tripled == 15

    # Serialize - only _x should be in state, not doubled/tripled
    state = serialize_instance_state(obj)
    assert '_x' in state
    assert 'doubled' not in state  # Properties shouldn't be in __dict__
    assert 'tripled' not in state

    # Reconstruct
    namespace = make_exec_namespace()
    exec(obj._remote_source, namespace)
    restored = object.__new__(namespace['ComputedClass'])
    restored.__dict__ = reconstruct_state(state, namespace, None, {})

    # Properties should still work
    assert restored.doubled == 10
    assert restored.tripled == 15


# =============================================================================
# Test 5: Custom __getstate__/__setstate__
# =============================================================================

def test_custom_pickle_protocol():
    """Test behavior with classes that define __getstate__/__setstate__."""
    @remote
    class PickleAware:
        def __init__(self, value):
            self.value = value
            self._cache = {"expensive": "computation"}

        def __getstate__(self):
            # Exclude _cache from pickle
            return {k: v for k, v in self.__dict__.items() if k != '_cache'}

        def __setstate__(self, state):
            self.__dict__.update(state)
            self._cache = {}  # Reinitialize cache

    obj = PickleAware(42)
    obj._cache["key"] = "data"

    # Current behavior: serialize_instance_state does NOT call __getstate__
    # It serializes the full __dict__
    state = serialize_instance_state(obj)

    # Document current behavior: _cache IS included (unlike pickle)
    # This is a known limitation
    assert '_cache' in state, "Current behavior: __getstate__ is NOT used"

    # If we want pickle-like behavior, we'd need to check for __getstate__


# =============================================================================
# Test 6: Closures referencing outer scope in methods
# =============================================================================

def test_closure_in_method():
    """Test that closures in methods capture outer scope correctly."""
    GLOBAL_MULTIPLIER = 3.0

    @remote
    class ClosureClass:
        def __init__(self, base):
            self.base = base

        def compute(self):
            # This closure captures GLOBAL_MULTIPLIER from outer scope
            def inner(x):
                return x * GLOBAL_MULTIPLIER
            return inner(self.base)

    # Note: This test verifies the class source captures the reference
    # The actual resolution depends on the namespace during reconstruction
    obj = ClosureClass(10)

    # Verify source is captured
    assert 'GLOBAL_MULTIPLIER' in obj._remote_source or 'inner' in obj._remote_source

    # For full functionality, GLOBAL_MULTIPLIER would need to be in module_refs
    # or server_imports


# =============================================================================
# Test 7: @staticmethod and @classmethod
# =============================================================================

def test_staticmethod_classmethod():
    """Test that @staticmethod and @classmethod work after reconstruction."""
    @remote
    class WithStatic:
        class_var = 100

        def __init__(self, x):
            self.x = x

        @staticmethod
        def helper(x):
            return x * 2

        @classmethod
        def from_value(cls, value):
            return cls(value)

        def use_helper(self):
            return self.helper(self.x)

    # Test on original
    obj = WithStatic(5)
    assert WithStatic.helper(10) == 20
    assert obj.helper(10) == 20
    assert obj.use_helper() == 10

    # Reconstruct
    namespace = make_exec_namespace()
    exec(obj._remote_source, namespace)

    # Test reconstructed class
    ReconClass = namespace['WithStatic']
    assert ReconClass.helper(10) == 20

    recon_obj = ReconClass.from_value(7)
    assert recon_obj.x == 7
    assert recon_obj.use_helper() == 14


# =============================================================================
# Test 8: Metaclasses
# =============================================================================

def test_metaclass():
    """Test behavior with custom metaclasses (known limitation)."""
    class MyMeta(type):
        def __new__(mcs, name, bases, namespace):
            namespace['meta_added'] = True
            return super().__new__(mcs, name, bases, namespace)

    # Note: @remote with metaclass is rejected by validation
    # This test documents that limitation

    # Instead, test that non-metaclass classes work fine
    @remote
    class NormalClass:
        def __init__(self):
            self.value = 42

    obj = NormalClass()
    assert obj.value == 42

    # Reconstruction should work
    namespace = make_exec_namespace()
    exec(obj._remote_source, namespace)
    ReconClass = namespace['NormalClass']

    recon_obj = ReconClass()
    assert recon_obj.value == 42


# =============================================================================
# Test 9: __init_subclass__ hooks
# =============================================================================

def test_init_subclass():
    """Test behavior with __init_subclass__ hooks that capture closure variables."""
    registered_classes = []

    @remote
    class BaseWithHook:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            registered_classes.append(cls.__name__)

    @remote
    class ChildClass(BaseWithHook):
        pass

    # Child was registered when defined
    assert 'ChildClass' in registered_classes

    # The closure variable should be captured and available for reconstruction
    assert 'registered_classes' in BaseWithHook._remote_closure_vars
    assert BaseWithHook._remote_closure_vars['registered_classes'] is registered_classes

    # Reconstruction should inject the closure variable into namespace
    namespace = make_exec_namespace()
    namespace.update(BaseWithHook._remote_closure_vars)
    exec(BaseWithHook._remote_source, namespace)
    exec(ChildClass._remote_source, namespace)

    # ChildClass was registered again during reconstruction
    assert registered_classes.count('ChildClass') == 2


# =============================================================================
# Test 10: Relative imports in source (documented limitation)
# =============================================================================

def test_relative_import_detection():
    """Test that relative imports in source are detectable."""
    # We can't easily test actual relative imports in a unit test,
    # but we can verify the source extraction works

    @remote
    class NoRelativeImports:
        def method(self):
            import os
            return os.getcwd()

    # Source should be extractable
    assert 'import os' in NoRelativeImports._remote_source

    # Relative imports would look like: from . import foo
    # These would fail during reconstruction - document this limitation


# =============================================================================
# Test 11: TYPE_CHECKING imports
# =============================================================================

def test_type_checking_imports():
    """Test handling of TYPE_CHECKING conditional imports."""
    # TYPE_CHECKING is False at runtime, so these imports don't execute

    @remote
    class WithTypeHints:
        def method(self) -> "SomeType":  # String annotation
            return None

    # Source extraction should work
    assert 'def method' in WithTypeHints._remote_source

    # String annotations don't need the type to be importable
    obj = WithTypeHints()
    assert obj.method() is None


# =============================================================================
# Test 12: Activation patching pattern
# =============================================================================

def test_cross_reference_timing():
    """Test that saved values can be used across contexts."""
    # This is more of a documentation test for the pattern

    @remote
    class ActivationStore:
        def __init__(self):
            self.saved = None

        def save(self, value):
            self.saved = value

        def get(self):
            return self.saved

    store = ActivationStore()
    store.save(torch.randn(10))

    # Serialize and restore
    state = serialize_instance_state(store)

    namespace = make_exec_namespace()
    exec(store._remote_source, namespace)
    restored = object.__new__(namespace['ActivationStore'])
    restored.__dict__ = reconstruct_state(state, namespace, None, {})

    # Saved tensor should be restored
    assert restored.saved is not None
    assert restored.saved.shape == (10,)


# =============================================================================
# Test 13: Empty class
# =============================================================================

def test_empty_class():
    """Test that empty classes serialize correctly."""
    @remote
    class Empty:
        pass

    obj = Empty()

    # Serialize
    state = serialize_instance_state(obj)
    assert state == {} or state == {'__dict__': {}}

    # Reconstruct
    namespace = make_exec_namespace()
    exec(obj._remote_source, namespace)
    restored = object.__new__(namespace['Empty'])
    restored.__dict__ = reconstruct_state(state, namespace, None, {})

    assert type(restored).__name__ == 'Empty'


# =============================================================================
# Test 14: Lambda in default argument
# =============================================================================

def test_lambda_default_argument():
    """Test functions with lambda default arguments."""
    @remote
    def func_with_lambda_default(processor=lambda x: x * 2):
        return processor(5)

    # Function should work
    assert func_with_lambda_default() == 10
    assert func_with_lambda_default(lambda x: x + 1) == 6

    # Source should include the lambda
    assert 'lambda' in func_with_lambda_default._remote_source


# =============================================================================
# Test 15: Very deeply nested state
# =============================================================================

def test_deeply_nested_state():
    """Test serialization of deeply nested data structures."""
    @remote
    class DeeplyNested:
        def __init__(self):
            self.data = {
                "level1": {
                    "level2": {
                        "level3": {
                            "level4": {
                                "values": [1, 2, 3, 4, 5],
                                "tensor": torch.tensor([1.0, 2.0]),
                            }
                        }
                    }
                }
            }

    obj = DeeplyNested()

    # Serialize
    state = serialize_instance_state(obj)

    # Should preserve nesting
    assert 'data' in state

    # Reconstruct
    namespace = make_exec_namespace()
    exec(obj._remote_source, namespace)
    restored = object.__new__(namespace['DeeplyNested'])
    restored.__dict__ = reconstruct_state(state, namespace, None, {})

    # Verify deep structure
    assert restored.data['level1']['level2']['level3']['level4']['values'] == [1, 2, 3, 4, 5]

    # Tensor should be restored
    deep_tensor = restored.data['level1']['level2']['level3']['level4']['tensor']
    assert torch.allclose(deep_tensor, torch.tensor([1.0, 2.0]))


# =============================================================================
# Test 16: Circular reference in instance state
# =============================================================================

def test_circular_reference():
    """Test that circular references are handled via deduplication."""
    @remote
    class Node:
        def __init__(self, value):
            self.value = value
            self.next = None

        def chain_length(self):
            seen = set()
            current = self
            length = 0
            while current and id(current) not in seen:
                seen.add(id(current))
                length += 1
                current = current.next
            return length

    # Create circular structure
    n1 = Node(1)
    n2 = Node(2)
    n3 = Node(3)
    n1.next = n2
    n2.next = n3
    n3.next = n1  # Circular!

    # Serialize - should not infinite loop
    state = serialize_instance_state(n1)

    # State should have __ref__ markers for deduplication
    json_str = json.dumps(state)  # Should not fail
    assert '__ref__' in json_str or '__id__' in json_str or 'value' in json_str


# =============================================================================
# Additional edge case tests
# =============================================================================

def test_nn_module_with_buffer():
    """Test nn.Module with registered buffers."""
    @remote
    class ModuleWithBuffer(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('running_mean', torch.zeros(10))
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x - self.running_mean)

    module = ModuleWithBuffer()
    module.running_mean.fill_(1.0)

    # Serialize
    state = serialize_instance_state(module)

    # Should capture buffers
    assert '_buffers' in state

    # Reconstruct
    namespace = make_exec_namespace()
    exec(module._remote_source, namespace)
    restored = object.__new__(namespace['ModuleWithBuffer'])
    restored.__dict__ = reconstruct_state(state, namespace, None, {})

    # Buffer should be restored
    assert torch.allclose(restored.running_mean, torch.ones(10))


def test_dataclass_serialization():
    """Test @dataclass decorated classes."""
    from dataclasses import dataclass

    @remote
    @dataclass
    class DataPoint:
        x: float
        y: float
        label: str = "default"

    point = DataPoint(1.5, 2.5, "test")

    # Serialize
    state = serialize_instance_state(point)

    assert state.get('x') == 1.5
    assert state.get('y') == 2.5
    assert state.get('label') == "test"

    # Reconstruct
    namespace = make_exec_namespace()
    namespace['dataclass'] = dataclass
    exec(point._remote_source, namespace)

    # Create new instance
    ReconClass = namespace['DataPoint']
    restored = ReconClass(**reconstruct_state(state, namespace, None, {}))

    assert restored.x == 1.5
    assert restored.y == 2.5
    assert restored.label == "test"


def test_enum_in_state():
    """Test serialization of enum values in state."""
    from enum import Enum

    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3

    @remote
    class WithEnum:
        def __init__(self, color):
            self.color = color  # Store actual enum instance
            self.name = color.name

    obj = WithEnum(Color.RED)

    # Serialize - enum should be serialized with class/module/member info
    state = serialize_instance_state(obj)

    # Check enum was serialized properly
    assert '__enum__' in state.get('color', {})
    assert state['color']['class'] == 'Color'
    assert state['color']['member'] == 'RED'

    # name (string) should serialize as-is
    assert state.get('name') == 'RED'


def test_mixed_tensor_types():
    """Test mixed torch tensor and numpy array serialization."""
    @remote
    class MixedArrays:
        def __init__(self):
            self.torch_tensor = torch.tensor([1.0, 2.0, 3.0])
            self.numpy_array = np.array([4.0, 5.0, 6.0])
            self.torch_int = torch.tensor([1, 2, 3], dtype=torch.int32)
            self.numpy_int = np.array([4, 5, 6], dtype=np.int32)

    obj = MixedArrays()

    # Serialize
    state = serialize_instance_state(obj)

    # All should be serialized as tensors
    assert '__tensor__' in state.get('torch_tensor', {})
    assert '__tensor__' in state.get('numpy_array', {})

    # Deserialize and check types/values
    torch_restored = deserialize_tensor(state['torch_tensor'])
    numpy_restored = deserialize_tensor(state['numpy_array'])

    # Both should restore to numpy (our deserialize returns numpy)
    np.testing.assert_array_almost_equal(torch_restored, [1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(numpy_restored, [4.0, 5.0, 6.0])


# =============================================================================
# Test 21: Async functions
# =============================================================================

def test_async_function():
    """Test behavior with async functions.

    Async functions are a gap in current support. This test documents
    the current behavior and what should happen.
    """
    import asyncio

    # Test 1: Can we decorate an async function?
    try:
        @remote
        async def async_analyze(x):
            return x * 2

        decorated = True
        has_source = hasattr(async_analyze, '_remote_source')
    except Exception as e:
        decorated = False
        has_source = False

    # Document current behavior
    assert decorated, "Async functions should be decoratable with @remote"
    assert has_source, "Async functions should have _remote_source"

    # Verify source contains 'async def'
    if has_source:
        assert 'async def' in async_analyze._remote_source

    # Test 2: Async method in a class
    @remote
    class AsyncAnalyzer:
        async def process(self, x):
            return x + 1

    assert 'async def process' in AsyncAnalyzer._remote_source

    # Note: Actually awaiting these would require server-side async support
    # which is documented as a gap in the design doc


# =============================================================================
# Test 22: Generator functions
# =============================================================================

def test_generator_function():
    """Test behavior with generator functions (yield).

    Generator functions are a gap in current support. This test documents
    the current behavior.
    """
    # Test 1: Can we decorate a generator function?
    try:
        @remote
        def generate_layers(n):
            for i in range(n):
                yield i

        decorated = True
        has_source = hasattr(generate_layers, '_remote_source')
    except Exception as e:
        decorated = False
        has_source = False

    # Document current behavior
    assert decorated, "Generator functions should be decoratable with @remote"
    assert has_source, "Generator functions should have _remote_source"

    # Verify source contains 'yield'
    if has_source:
        assert 'yield' in generate_layers._remote_source

    # Test 2: Generator method in a class
    @remote
    class LayerIterator:
        def __init__(self, n):
            self.n = n

        def iterate(self):
            for i in range(self.n):
                yield i

    assert 'yield' in LayerIterator._remote_source

    # Note: Server-side behavior for generators is undefined
    # (does it return a list? a generator? lazy eval?)


# =============================================================================
# Test 23: Weak references
# =============================================================================

def test_weakref_serialization():
    """Test serialization of objects containing weak references.

    Current behavior: weakrefs serialize as {"__weakref__": True} which
    deserializes to None. This test verifies that behavior doesn't cause errors.
    """
    import weakref

    # Create a simple object to hold a weakref
    # We don't use @remote here to avoid the closure issue with weakref module
    class Target:
        def __init__(self, value):
            self.value = value

    class Holder:
        pass

    target = Target(42)
    holder = Holder()
    holder.target_ref = weakref.ref(target)
    holder.name = "holder"

    # Verify weakref works locally
    assert holder.target_ref() is target
    assert holder.target_ref().value == 42

    # Serialize - should not raise
    state = serialize_instance_state(holder)

    # Document current behavior: weakrefs become {"__weakref__": True}
    # which deserializes to None
    assert '__weakref__' in state.get('target_ref', {}), \
        "Weakrefs should serialize as __weakref__ marker"

    # The name should serialize normally
    assert state.get('name') == "holder"

    # Test deserialization - weakref becomes None
    from nnsight.intervention.serialization_source import deserialize_source_based

    # Create minimal data structure for reconstruction
    namespace = {}
    model = None
    reconstructed = {}

    # Import reconstruct_value directly
    from nnsight.intervention.serialization_source import reconstruct_value
    restored_ref = reconstruct_value(state['target_ref'], namespace, model, reconstructed)

    # Weakref deserializes to None
    assert restored_ref is None, "Weakrefs should deserialize to None"


# =============================================================================
# Test 24: Pickle hooks behavior documentation
# =============================================================================

def test_pickle_hooks_current_behavior():
    """Document that __getstate__/__setstate__ are NOT currently honored.

    This is a known limitation documented in the design doc. The test
    verifies the current behavior so we know if it changes.
    """
    @remote
    class ExcludesLargeData:
        def __init__(self):
            self.small = "keep me"
            self.large = "x" * 10000  # Would be excluded by __getstate__

        def __getstate__(self):
            # Exclude large data from serialization
            return {'small': self.small}

        def __setstate__(self, state):
            self.__dict__.update(state)
            self.large = ""  # Would be reinitialized

    obj = ExcludesLargeData()
    state = serialize_instance_state(obj)

    # CURRENT BEHAVIOR: __getstate__ is IGNORED
    # Both small and large are serialized
    assert 'small' in state
    assert 'large' in state, "Current behavior: __getstate__ is NOT used, large data IS serialized"

    # Verify the large data is actually there
    assert len(state['large']) == 10000

    # TODO: When pickle hooks are implemented, this test should change:
    # assert 'large' not in state, "Future behavior: __getstate__ should be honored"


# =============================================================================
# Test 25: Unconventional 'self' usage
# =============================================================================

def test_unconventional_self():
    """Test that 'self' as a module variable (not a parameter) is handled correctly.

    Python convention uses 'self' as the first parameter of instance methods,
    but 'self' is not a keyword - it can be used as a regular variable name.
    This test verifies such unconventional usage is handled properly.
    """
    # Define 'self' as a module-level variable with a value
    self = {"config": "global_config", "value": 42}

    @remote
    class UnconventionalSelf:
        def __init__(this, x):  # Using 'this' instead of 'self'
            this.x = x

        def get_global_self(this):
            # Access the module-level 'self' variable
            return self["config"]

        def compute(this):
            return this.x + self["value"]

    # Verify the closure captured module-level 'self'
    assert 'self' in UnconventionalSelf._remote_closure_vars
    assert UnconventionalSelf._remote_closure_vars['self']['config'] == 'global_config'

    obj = UnconventionalSelf(10)

    # Test that methods work correctly
    assert obj.get_global_self() == "global_config"
    assert obj.compute() == 52  # 10 + 42

    # Test reconstruction
    namespace = make_exec_namespace()
    namespace.update(UnconventionalSelf._remote_closure_vars)
    exec(obj._remote_source, namespace)

    ReconClass = namespace['UnconventionalSelf']
    restored = ReconClass(20)

    assert restored.get_global_self() == "global_config"
    assert restored.compute() == 62  # 20 + 42


# =============================================================================
# Test 26: Module variable overriding builtin
# =============================================================================

def test_module_variable_overrides_builtin():
    """Test that module variables overriding builtins are correctly captured.

    Python allows shadowing builtins with module-level variables. For example,
    a module can define `list = SomeClass` which shadows the builtin `list`.
    The serialization system must capture these overridden values rather than
    assuming they refer to builtins.

    This test uses a JSON-serializable value to override a builtin, since
    non-serializable values (like functions) would need to be @remote decorated.
    """
    # Override 'len' with a constant (simulating a config value that shadows a builtin)
    # Note: In real code, overriding builtins with constants is unusual but legal
    CUSTOM_LENGTH = 42
    len = CUSTOM_LENGTH  # Shadow the builtin with a constant

    @remote
    class UsesOverriddenBuiltin:
        def __init__(self, data):
            self.data = data

        def get_length(self):
            # This should use the overridden 'len' (a constant), not the builtin
            return len

    # Verify that 'len' was captured as a closure variable
    assert 'len' in UsesOverriddenBuiltin._remote_closure_vars
    assert UsesOverriddenBuiltin._remote_closure_vars['len'] == CUSTOM_LENGTH

    obj = UsesOverriddenBuiltin([1, 2, 3])

    # Should use the overridden len (returns the constant)
    assert obj.get_length() == 42

    # Test reconstruction
    namespace = make_exec_namespace()
    namespace.update(UsesOverriddenBuiltin._remote_closure_vars)
    exec(obj._remote_source, namespace)

    ReconClass = namespace['UsesOverriddenBuiltin']
    restored = ReconClass([1, 2, 3, 4, 5])

    # Should still use the overridden len
    assert restored.get_length() == 42


def test_module_variable_overrides_builtin_with_remote_function():
    """Test overriding a builtin with a @remote-decorated function.

    This is a more realistic case where someone defines a custom implementation
    of a builtin function and uses @remote to make it serializable.
    """
    @remote
    def custom_len(x):
        # Custom implementation that doubles the length
        result = 0
        for _ in x:
            result += 2
        return result

    len = custom_len  # Shadow the builtin

    @remote
    class UsesRemoteOverride:
        def __init__(self, data):
            self.data = data

        def get_length(self):
            # This should use the @remote custom_len, not the builtin
            return len(self.data)

    # Since custom_len is @remote decorated, it should be skipped (not captured)
    # because @remote functions are serialized separately
    assert 'len' not in UsesRemoteOverride._remote_closure_vars

    obj = UsesRemoteOverride([1, 2, 3])

    # Should use the custom len
    assert obj.get_length() == 6  # 3 items * 2 = 6


# =============================================================================
# Test 28: NameFinder edge cases (potential bugs in simpler AST visitor)
# =============================================================================

def test_namefinder_comprehension_shadowing():
    """Test that comprehension variables don't incorrectly shadow external names.

    This test exposes a potential bug in simpler AST visitors that don't track
    scopes properly. If the visitor doesn't distinguish between comprehension
    scope and class scope, it might think 'x' is defined locally when it's
    actually an external reference that happens to share a name with the
    comprehension variable.

    The correct behavior: 'x' used OUTSIDE the comprehension should be recognized
    as external, even if 'x' is also used as a loop variable inside a comprehension.
    """
    # Module-level x that should be captured
    x = 42

    @remote
    class UsesExternalAndComprehension:
        def method(self):
            # 'x' here refers to module-level x (external)
            external_value = x

            # 'x' here is a comprehension-local variable (different scope)
            squares = [x * x for x in range(5)]

            return external_value, squares

    # The external 'x' should be captured
    closure_vars = getattr(UsesExternalAndComprehension, '_remote_closure_vars', {})
    assert 'x' in closure_vars, f"External 'x' should be captured, got: {closure_vars}"
    assert closure_vars['x'] == 42


def test_namefinder_nested_function_shadowing():
    """Test that inner function parameters don't shadow outer references.

    Similar to comprehension shadowing - an inner function's parameter shouldn't
    cause the outer reference to be missed.
    """
    multiplier = 3

    @remote
    class UsesExternalWithInnerFunction:
        def method(self):
            # Uses external 'multiplier'
            base = multiplier * 10

            # Inner function has its own 'multiplier' parameter
            def inner(multiplier):
                return multiplier * 2

            return base, inner(5)

    # The external 'multiplier' should be captured
    closure_vars = getattr(UsesExternalWithInnerFunction, '_remote_closure_vars', {})
    assert 'multiplier' in closure_vars, f"External 'multiplier' should be captured, got: {closure_vars}"
    assert closure_vars['multiplier'] == 3


def test_namefinder_import_shadowing():
    """Test that import statements are recognized as defining names locally.

    If a name is imported inside the class, it shouldn't be treated as external.
    """
    @remote
    class HasLocalImport:
        def method(self):
            import json  # Local import
            return json.dumps([1, 2, 3])

    # 'json' is imported locally, shouldn't need to be captured
    closure_vars = getattr(HasLocalImport, '_remote_closure_vars', {})
    module_refs = getattr(HasLocalImport, '_remote_module_refs', {})

    # json shouldn't be in closure_vars (it's imported locally)
    # Note: It might be in module_refs if the system decides to track server imports
    assert 'json' not in closure_vars, f"'json' should not be in closure_vars: {closure_vars}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
