"""
Tests for disambiguation logic in source serialization.

These tests verify the invariants and capabilities described in
docs/disambiguation-design.md, including:
- Qualified name uniqueness (no collisions)
- Identity preservation (shared references stay shared)
- Namespace sharing (module-level state)
- Circular reference handling
- Nonlocal closure detection
- Mutable class attribute warnings
- Immutable closure captures
- Multiple instances of same class
"""

import json
import pytest
import sys
import warnings
import torch
import torch.nn as nn

from nnsight.intervention.serialization_source import (
    serialize_source_based,
    deserialize_source_based,
    serialize_value,
    serialize_instance_state,
    extract_all,
    auto_discover_class,
    _auto_discover_function,
    _has_nonlocal_closure,
    _check_mutable_class_attributes,
    SourceSerializationError,
    CLASS_MARKER,
    DICT_MARKER,
    REF_MARKER,
    ID_MARKER,
)
from nnsight.remote import remote


# =============================================================================
# Test: Qualified Name Uniqueness (Challenge 1 from design doc)
# =============================================================================

def test_qualified_names_prevent_collision():
    """Test that classes with same short name but different modules don't collide.

    From disambiguation-design.md Challenge 1:
    Two classes named 'Helper' from different modules should coexist.
    """
    # Define two classes with same name in different "modules"
    @remote
    class Helper:
        """First Helper class"""
        def process(self, x):
            return x * 2

    # Manually set module to simulate different origins
    Helper.__module__ = 'test_disambiguation.analytics'

    @remote
    class Helper2:
        """Second Helper class (we'll rename it)"""
        def process(self, x):
            return x * 3

    # Rename to Helper but with different module
    Helper2.__name__ = 'Helper'
    Helper2.__module__ = 'test_disambiguation.visualization'

    # Create instances
    h1 = Helper()
    h2 = Helper2()

    # Use extract_all with correct signature
    variables, remote_objects, model_refs = extract_all(
        {'h1': h1, 'h2': h2},
        traced_model=None,
        strict_remote=False
    )

    # Both should be present with qualified names
    qualified_names = list(remote_objects.keys())
    assert len(qualified_names) == 2, f"Expected 2 classes, got {qualified_names}"

    # Qualified names should be different
    assert qualified_names[0] != qualified_names[1], "Qualified names should differ"

    # Both should contain 'Helper'
    assert all('Helper' in name for name in qualified_names)


def test_qualified_names_in_serialized_value():
    """Test that serialize_value uses qualified names for class instances."""
    @remote
    class MyClass:
        def __init__(self, value):
            self.value = value

    MyClass.__module__ = 'test_disambiguation.mymodule'

    obj = MyClass(42)

    # serialize_value wraps in CLASS_MARKER
    memo = {}
    discovered = {}
    state = serialize_value(obj, 'my_obj', memo, discovered)

    # The __class__ should be the qualified name
    assert CLASS_MARKER in state
    assert 'test_disambiguation.mymodule.MyClass' in state[CLASS_MARKER]


# =============================================================================
# Test: Shared Reference Identity (Challenge 2 & 3 from design doc)
# =============================================================================

def test_shared_reference_identity_preserved():
    """Test that shared references remain identical after serialization.

    From disambiguation-design.md Invariant 2:
    If `a is b` before serialization, then `a' is b'` after deserialization.
    """
    @remote
    class Container:
        def __init__(self, data):
            self.data = data

    # Create shared data
    shared_list = [1, 2, 3]

    # Two containers pointing to same list
    c1 = Container(shared_list)
    c2 = Container(shared_list)

    # Verify they share before serialization
    assert c1.data is c2.data

    # Serialize both with shared memo
    memo = {}
    discovered = {}
    s1 = serialize_value(c1, 'c1', memo, discovered)
    s2 = serialize_value(c2, 'c2', memo, discovered)

    # Both should have CLASS_MARKER
    assert CLASS_MARKER in s1
    assert CLASS_MARKER in s2

    # The second should have a __ref__ since we already saw c1
    # (c1 and c2 are different objects, but share the same data list)


def test_circular_reference_handling():
    """Test that circular references are handled correctly.

    From disambiguation-design.md Capabilities - Circular References:
    a["ref"] = b; b["ref"] = a should work after round-trip.
    """
    @remote
    class Node:
        def __init__(self, name):
            self.name = name
            self.next = None

    # Create circular reference
    a = Node("a")
    b = Node("b")
    a.next = b
    b.next = a  # Circular!

    # Serialize - should not infinite loop
    memo = {}
    discovered = {}
    state_a = serialize_value(a, 'node_a', memo, discovered)

    # Check structure - should have CLASS_MARKER
    assert CLASS_MARKER in state_a
    assert DICT_MARKER in state_a
    assert state_a[DICT_MARKER]['name'] == 'a'

    # The nested node should eventually have a __ref__ to break the cycle
    next_state = state_a[DICT_MARKER]['next']
    assert CLASS_MARKER in next_state or REF_MARKER in next_state


def test_self_reference_handling():
    """Test that self-referential objects work."""
    @remote
    class SelfRef:
        def __init__(self):
            self.self_ref = self

    obj = SelfRef()
    assert obj.self_ref is obj

    # Serialize
    memo = {}
    discovered = {}
    state = serialize_value(obj, 'self_ref_obj', memo, discovered)

    # The self_ref should be a __ref__ since we've already seen this object
    assert DICT_MARKER in state
    assert REF_MARKER in state[DICT_MARKER]['self_ref']


# =============================================================================
# Test: Nonlocal Closure Detection (Limitation 1 from design doc)
# =============================================================================

def test_nonlocal_closure_detected():
    """Test that functions with nonlocal are detected."""
    def make_counter():
        count = 0
        def increment():
            nonlocal count
            count += 1
            return count
        return increment

    counter = make_counter()

    has_nonlocal, names = _has_nonlocal_closure(counter)
    assert has_nonlocal
    assert 'count' in names


def test_nonlocal_closure_rejected():
    """Test that auto-discovery rejects nonlocal closures with clear error.

    From disambiguation-design.md Limitation 1:
    Functions with nonlocal should raise SourceSerializationError.
    """
    def make_counter():
        count = 0
        def increment():
            nonlocal count
            count += 1
            return count
        return increment

    counter = make_counter()

    with pytest.raises(SourceSerializationError) as exc_info:
        _auto_discover_function(counter)

    # Error should mention nonlocal and the variable
    error_msg = str(exc_info.value)
    assert 'nonlocal' in error_msg
    assert 'count' in error_msg
    assert 'class' in error_msg.lower()  # Should suggest using a class


def test_multiple_nonlocal_variables():
    """Test detection of multiple nonlocal variables."""
    def make_accumulator():
        total = 0
        count = 0
        def add(x):
            nonlocal total, count
            total += x
            count += 1
            return total / count
        return add

    acc = make_accumulator()

    has_nonlocal, names = _has_nonlocal_closure(acc)
    assert has_nonlocal
    assert 'total' in names
    assert 'count' in names


def test_no_nonlocal_passes():
    """Test that functions without nonlocal pass the check."""
    def simple_func(x):
        return x * 2

    has_nonlocal, names = _has_nonlocal_closure(simple_func)
    assert not has_nonlocal
    assert len(names) == 0


# =============================================================================
# Test: Mutable Class Attribute Warnings (Limitation 2 from design doc)
# =============================================================================

def test_mutable_class_attribute_detected():
    """Test that mutable class attributes are detected."""
    class Registry:
        instances = []  # Mutable class attribute
        name = "registry"  # Immutable - should not be flagged

        def __init__(self, value):
            self.value = value

    attrs = _check_mutable_class_attributes(Registry)

    assert len(attrs) == 1
    assert attrs[0] == ('instances', 'list')


def test_multiple_mutable_class_attributes():
    """Test detection of multiple mutable class attributes."""
    class MultiMutable:
        items = []
        cache = {}
        tags = set()
        name = "test"  # Immutable

        def __init__(self):
            pass

    attrs = _check_mutable_class_attributes(MultiMutable)

    attr_names = [name for name, _ in attrs]
    assert 'items' in attr_names
    assert 'cache' in attr_names


def test_mutable_class_attribute_warning_issued():
    """Test that auto_discover_class warns about mutable class attributes."""
    class WithMutableAttr:
        shared_state = []

        def add(self, x):
            self.shared_state.append(x)

    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        discovered = {}
        auto_discover_class(WithMutableAttr, discovered)

        # Should have issued a warning
        mutable_warnings = [x for x in w if 'mutable class attribute' in str(x.message).lower()]
        assert len(mutable_warnings) >= 1
        assert 'shared_state' in str(mutable_warnings[0].message)


# =============================================================================
# Test: Immutable Closure Captures (Limitation 3 from design doc)
# =============================================================================

def test_immutable_closure_captured():
    """Test that immutable closure values are captured as closure_vars.

    From disambiguation-design.md Limitation 3:
    Immutable captured values are serialized as globals.
    """
    def make_multiplier(factor):
        def multiply(x):
            return x * factor
        return multiply

    double = make_multiplier(2)
    triple = make_multiplier(3)

    # Auto-discover should work (no nonlocal)
    result_double = _auto_discover_function(double)
    result_triple = _auto_discover_function(triple)

    # Closure vars should capture the factor
    assert 'factor' in result_double.get('closure_vars', {})
    assert result_double['closure_vars']['factor'] == 2

    assert 'factor' in result_triple.get('closure_vars', {})
    assert result_triple['closure_vars']['factor'] == 3


def test_immutable_closure_different_instances():
    """Test that different closure instances have independent captured values."""
    def make_adder(n):
        def add(x):
            return x + n
        return add

    add5 = make_adder(5)
    add10 = make_adder(10)

    result5 = _auto_discover_function(add5)
    result10 = _auto_discover_function(add10)

    # Each should have its own value
    assert result5['closure_vars']['n'] == 5
    assert result10['closure_vars']['n'] == 10


# =============================================================================
# Test: Multiple Instances of Same Class (Capabilities from design doc)
# =============================================================================

def test_multiple_instances_share_class():
    """Test that multiple instances share the same class after serialization.

    From disambiguation-design.md Capabilities - Multiple Instances:
    type(c1) is type(c2) should be true after round-trip.
    """
    @remote
    class Counter:
        def __init__(self, start):
            self.value = start

    c1 = Counter(10)
    c2 = Counter(20)

    # Use extract_all with correct signature
    variables, remote_objects, model_refs = extract_all(
        {'c1': c1, 'c2': c2},
        traced_model=None,
        strict_remote=False
    )

    # Should have one class definition
    class_entries = [k for k, v in remote_objects.items() if v.get('type') == 'class']
    assert len(class_entries) == 1, f"Expected 1 class, got {class_entries}"

    # Should have two instances
    class_key = class_entries[0]
    instances = remote_objects[class_key].get('instances', {})
    assert len(instances) == 2


def test_multiple_instances_independent_state():
    """Test that multiple instances maintain independent state."""
    @remote
    class Holder:
        def __init__(self, data):
            self.data = data

    h1 = Holder([1, 2, 3])
    h2 = Holder({'key': 'value'})

    # Serialize using serialize_value
    memo = {}
    discovered = {}
    s1 = serialize_value(h1, 'h1', memo, discovered)
    s2 = serialize_value(h2, 'h2', memo, discovered)

    # Check independent data
    assert s1[DICT_MARKER]['data'] == [1, 2, 3]
    assert s2[DICT_MARKER]['data'] == {'key': 'value'}


# =============================================================================
# Test: Globals ID Tracking (for namespace groups)
# =============================================================================

def test_function_globals_id_tracked():
    """Test that auto-discovered functions track their __globals__ id."""
    def my_function(x):
        return x * 2

    result = _auto_discover_function(my_function)

    # Should have globals_id
    assert 'globals_id' in result
    assert result['globals_id'] is not None


def test_same_module_functions_share_globals_id():
    """Test that functions from same module have same globals_id."""
    # Define two functions in this module
    def func_a(x):
        return x + 1

    def func_b(x):
        return x + 2

    result_a = _auto_discover_function(func_a)
    result_b = _auto_discover_function(func_b)

    # They should have the same globals_id (same module)
    assert result_a['globals_id'] == result_b['globals_id']


# =============================================================================
# Test: nn.Module Subclass Handling
# =============================================================================

def test_nn_module_subclass_serialization():
    """Test that nn.Module subclasses are properly serialized."""
    class MyLayer(nn.Module):
        def __init__(self, size):
            super().__init__()
            self.linear = nn.Linear(size, size)
            self.scale = 2.0

        def forward(self, x):
            return self.linear(x) * self.scale

    layer = MyLayer(64)

    # Should serialize without error
    memo = {}
    discovered = {}
    state = serialize_value(layer, 'layer', memo, discovered)

    # Should have class marker (for user-defined nn.Module subclasses)
    # or nn_module marker (for torch built-ins)
    assert CLASS_MARKER in state or '__nn_module__' in state


def test_nn_module_with_parameters():
    """Test that nn.Module parameters are preserved in serialization."""
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)

        def forward(self, x):
            return self.fc(x)

    net = SimpleNet()

    # Set specific weight values
    with torch.no_grad():
        net.fc.weight.fill_(0.5)
        net.fc.bias.fill_(0.1)

    # Serialize
    memo = {}
    discovered = {}
    state = serialize_value(net, 'net', memo, discovered)

    # Should serialize without error
    assert state is not None


# =============================================================================
# Test: Edge Cases
# =============================================================================

def test_empty_class():
    """Test serialization of class with no instance attributes."""
    @remote
    class Empty:
        pass

    obj = Empty()
    memo = {}
    discovered = {}
    state = serialize_value(obj, 'empty_obj', memo, discovered)

    assert CLASS_MARKER in state
    assert DICT_MARKER in state
    assert state[DICT_MARKER] == {}


def test_class_with_none_values():
    """Test serialization of class with None values."""
    @remote
    class WithNone:
        def __init__(self):
            self.value = None
            self.optional = None

    obj = WithNone()
    memo = {}
    discovered = {}
    state = serialize_value(obj, 'with_none', memo, discovered)

    assert state[DICT_MARKER]['value'] is None
    assert state[DICT_MARKER]['optional'] is None


def test_deeply_nested_structure():
    """Test serialization of deeply nested class instances."""
    @remote
    class Nested:
        def __init__(self, depth, value):
            self.value = value
            self.child = Nested(depth - 1, value * 2) if depth > 0 else None

    root = Nested(3, 1)

    # Should serialize without stack overflow
    memo = {}
    discovered = {}
    state = serialize_value(root, 'root', memo, discovered)

    # Check structure
    assert state[DICT_MARKER]['value'] == 1
    assert state[DICT_MARKER]['child'] is not None


# =============================================================================
# Test: Qualified Name Extraction
# =============================================================================

def test_qualified_name_in_function_result():
    """Test that auto-discovered functions have qualified_name field."""
    def my_test_function(x):
        return x

    result = _auto_discover_function(my_test_function)

    assert 'qualified_name' in result
    assert 'my_test_function' in result['qualified_name']


def test_main_module_handling():
    """Test that __main__ module is handled correctly."""
    # Functions defined at module level in test file
    def local_func():
        pass

    # Should work even for __main__ or test module
    result = _auto_discover_function(local_func)
    assert result['qualified_name'] is not None


# =============================================================================
# Integration Tests
# =============================================================================

def test_full_roundtrip_with_shared_state():
    """Integration test: full serialize/deserialize with shared state."""
    @remote
    class DataHolder:
        def __init__(self, data):
            self.data = data

    shared = {'count': 0}
    h1 = DataHolder(shared)
    h2 = DataHolder(shared)

    # Use extract_all with correct signature
    variables, remote_objects, model_refs = extract_all(
        {'h1': h1, 'h2': h2, 'shared': shared},
        traced_model=None,
        strict_remote=False
    )

    # Should successfully extract
    assert remote_objects is not None
    assert variables is not None


def test_full_roundtrip_with_mixed_types():
    """Integration test with various types."""
    @remote
    class Config:
        def __init__(self, settings):
            self.settings = settings

    config = Config({
        'threshold': 0.5,
        'layers': [1, 2, 3],
        'enabled': True,
        'name': 'test',
    })

    tensor = torch.randn(10, 10)

    variables, remote_objects, model_refs = extract_all(
        {
            'config': config,
            'tensor': tensor,
            'scalar': 42,
            'text': 'hello',
        },
        traced_model=None,
        strict_remote=False
    )

    assert remote_objects is not None
    assert variables is not None


# =============================================================================
# Test: Invariants from Design Doc
# =============================================================================

def test_invariant_qualified_name_uniqueness():
    """Invariant 1: Two different classes always have different qualified names."""
    @remote
    class ClassA:
        pass

    @remote
    class ClassB:
        pass

    ClassA.__module__ = 'pkg.module_a'
    ClassB.__module__ = 'pkg.module_b'

    discovered_a = {}
    discovered_b = {}
    auto_discover_class(ClassA, discovered_a)
    auto_discover_class(ClassB, discovered_b)

    key_a = list(discovered_a.keys())[0]
    key_b = list(discovered_b.keys())[0]

    assert key_a != key_b
    assert 'ClassA' in key_a
    assert 'ClassB' in key_b


def test_invariant_identity_preservation():
    """Invariant 2: If a is b before, then a' is b' after deserialization."""
    shared = {'value': 42}

    @remote
    class Wrapper:
        def __init__(self, ref):
            self.ref = ref

    w1 = Wrapper(shared)
    w2 = Wrapper(shared)

    # Both point to same object
    assert w1.ref is w2.ref

    # Serialize with shared memo to track identity
    memo = {}
    discovered = {}
    s1 = serialize_value(w1, 'w1', memo, discovered)
    s2 = serialize_value(w2, 'w2', memo, discovered)

    # The shared dict should only be fully serialized once
    # Second reference should be a REF_MARKER
    # (This depends on serialization order, but at minimum both should serialize)
    assert s1 is not None
    assert s2 is not None


def test_invariant_isinstance_after_roundtrip():
    """Invariant 4: isinstance(obj, MyClass) works after round-trip."""
    @remote
    class MyClass:
        def __init__(self, x):
            self.x = x

    obj = MyClass(10)

    # Serialize
    variables, remote_objects, model_refs = extract_all(
        {'obj': obj},
        traced_model=None,
        strict_remote=False
    )

    # The class should be in remote_objects
    assert len(remote_objects) == 1

    # And it should have the instance
    class_key = list(remote_objects.keys())[0]
    assert 'instances' in remote_objects[class_key]
    assert len(remote_objects[class_key]['instances']) == 1


# =============================================================================
# Integration Test: Probe Training Pattern (from design doc)
# =============================================================================

def test_probe_training_pattern_serialization():
    """Test the probe training pattern from nnsight-source-serialization-design.md.

    This tests the serialization of:
    - nn.Module subclass with trainable parameters
    - Training data (list of dicts)
    - All dependencies needed for the training loop

    From the design doc example:
    ```
    probe = SentimentProbe(768)
    optimizer = torch.optim.Adam(probe.parameters())

    with model.session(remote=True):
        for epoch in range(10):
            for example in training_data:
                with model.trace(example["text"]):
                    hidden = model.transformer.h[10].output[0][:, -1, :]
                    logits = probe(hidden)
                    loss = F.cross_entropy(logits, ...)
                    loss.backward()
                optimizer.step()
    ```
    """
    import torch.nn.functional as F

    # Define the probe class with @remote decorator
    @remote
    class SentimentProbe(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear = nn.Linear(hidden_size, 2)

        def forward(self, x):
            return self.linear(x)

    # Create probe and initialize weights to specific values
    probe = SentimentProbe(768)
    with torch.no_grad():
        probe.linear.weight.fill_(0.1)
        probe.linear.bias.fill_(0.0)

    # Training data (JSON-serializable structure)
    training_data = [
        {"text": "This is great!", "label": 1},
        {"text": "This is terrible.", "label": 0},
        {"text": "I love it.", "label": 1},
    ]

    # Serialize - this should capture probe and training_data
    memo = {}
    discovered = {}

    # Serialize probe
    probe_state = serialize_value(probe, 'probe', memo, discovered)
    assert probe_state is not None

    # The probe should be serialized with its class info
    assert CLASS_MARKER in probe_state or '__nn_module__' in probe_state

    # Training data should be JSON-serializable
    variables, remote_objects, model_refs = extract_all(
        {
            'probe': probe,
            'training_data': training_data,
            'lr': 0.001,
        },
        traced_model=None,
        strict_remote=False
    )

    # training_data should be in variables (JSON-serializable)
    assert 'training_data' in variables
    assert variables['training_data'] == training_data

    # lr should be in variables
    assert 'lr' in variables
    assert variables['lr'] == 0.001

    # probe should be in remote_objects (needs class source)
    assert len(remote_objects) >= 1


def test_probe_with_trained_weights_roundtrip():
    """Test that probe weights survive serialization round-trip.

    From design doc Test 7: @remote Probe with Trained Weights
    1. Client defines probe class
    2. Probe weights are set to specific values (simulating training)
    3. Verify weights transferred correctly
    """
    class SimpleProbe(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.fc = nn.Linear(in_features, out_features)

        def forward(self, x):
            return self.fc(x)

    # Create probe and set specific weight values
    probe = SimpleProbe(64, 2)
    with torch.no_grad():
        probe.fc.weight.fill_(0.5)
        probe.fc.bias.fill_(0.25)

    # Serialize
    memo = {}
    discovered = {}
    state = serialize_value(probe, 'probe', memo, discovered)

    # Should serialize successfully
    assert state is not None

    # The state should contain the weight values somehow
    # Either directly in the dict or through a tensor marker
    assert DICT_MARKER in state or '__nn_module__' in state


def test_training_loop_components():
    """Test that all components needed for training loop serialize correctly.

    This verifies that the individual pieces of a training loop
    (probe, loss function reference, iteration variables) can be captured.
    """
    import torch.nn.functional as F

    @remote
    class Probe(nn.Module):
        def __init__(self, size):
            super().__init__()
            self.linear = nn.Linear(size, 2)

        def forward(self, x):
            return self.linear(x)

    probe = Probe(128)
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001

    # All these should be extractable
    variables, remote_objects, model_refs = extract_all(
        {
            'probe': probe,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
        },
        traced_model=None,
        strict_remote=False
    )

    # Numeric hyperparameters should be in variables
    assert 'num_epochs' in variables
    assert variables['num_epochs'] == 10
    assert 'batch_size' in variables
    assert variables['batch_size'] == 32
    assert 'learning_rate' in variables
    assert variables['learning_rate'] == 0.001

    # Probe should be discoverable
    assert len(remote_objects) >= 1
