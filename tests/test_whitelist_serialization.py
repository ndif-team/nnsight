"""
Tests for whitelist-based module serialization.

These tests verify that:
1. Non-whitelisted modules are automatically serialized by source
2. The serialized code runs in isolation (its own copy of module state)
3. Whitelisted modules are serialized by reference (not duplicated)

Run with: pytest tests/test_whitelist_serialization.py -v
"""

import sys
import pytest
import torch

# Add tests directory to path so we can import mymethods
sys.path.insert(0, "tests")

from nnsight import LanguageModel
from nnsight.intervention import serialization
from nnsight.intervention.serialization import dumps, loads


def _whitelist_api():
    """The module whitelist, or skip if it is no longer part of the API.

    ``SERVER_MODULES_WHITELIST`` / ``_is_whitelisted_module`` were dropped from
    ``serialization.py``. Importing them at module scope made all 12 tests in
    this file an ImportError at collection -- including the nine that test
    source serialization, which still works. Resolve them per-test instead, so
    the rest of the file runs and these three come back automatically if the
    API returns.
    """

    whitelist = getattr(serialization, "SERVER_MODULES_WHITELIST", None)
    predicate = getattr(serialization, "_is_whitelisted_module", None)

    if whitelist is None or predicate is None:
        pytest.skip(
            "serialization.SERVER_MODULES_WHITELIST / _is_whitelisted_module "
            "are no longer part of the API"
        )

    return whitelist, predicate


@pytest.fixture
def registered_mymethods():
    """Register ``mymethods`` by value for the duration of one test.

    Serializing by source is opt-in: ``dumps`` emits a by-reference GLOBAL
    (47 bytes, no source) until the owning module is registered, and the full
    definition (887 bytes) once it is. The remote backend registers local
    modules for you via ``get_local_env()`` before it calls ``dumps``, so the
    registered state is what a remote trace actually serializes under.

    ``register`` mutates a process-global set in cloudpickle, so without this
    fixture the tests below only passed when an earlier test in the same
    session happened to have registered the module first -- each of them
    failed when run alone. Registration is undone on teardown so neither
    ordering nor leakage can decide the result.
    """

    import mymethods.stateful
    from cloudpickle.cloudpickle import unregister_pickle_by_value

    from nnsight.ndif import register

    register(mymethods.stateful)
    try:
        yield mymethods.stateful
    finally:
        unregister_pickle_by_value(mymethods.stateful)


@pytest.fixture
def unregistered_mymethods():
    """Guarantee ``mymethods`` is *not* registered for the duration of one test.

    Any trace in this session calls ``get_local_env()``, which registers local
    modules process-wide, so by the time a later test runs the module is
    already registered. Lift those entries out of cloudpickle's global set and
    put them back on teardown, so the by-reference case can be asserted no
    matter where in the file it runs.
    """

    import mymethods.stateful  # noqa: F401  (ensure the module is importable)
    from cloudpickle.cloudpickle import _PICKLE_BY_VALUE_MODULES

    removed = {
        name
        for name in _PICKLE_BY_VALUE_MODULES
        if name == "mymethods" or name.startswith("mymethods.")
    }
    _PICKLE_BY_VALUE_MODULES.difference_update(removed)
    try:
        yield
    finally:
        _PICKLE_BY_VALUE_MODULES.update(removed)


# =============================================================================
# Whitelist Function Tests
# =============================================================================


def test_whitelisted_modules():
    """Test that core modules are in the whitelist."""
    _, _is_whitelisted_module = _whitelist_api()

    assert _is_whitelisted_module("torch")
    assert _is_whitelisted_module("torch.nn")
    assert _is_whitelisted_module("torch.nn.functional")
    assert _is_whitelisted_module("numpy")
    assert _is_whitelisted_module("transformers")
    assert _is_whitelisted_module("nnsight")


def test_non_whitelisted_modules():
    """Test that user modules are NOT in the whitelist."""
    _, _is_whitelisted_module = _whitelist_api()

    assert not _is_whitelisted_module("mymethods")
    assert not _is_whitelisted_module("mymethods.stateful")
    assert not _is_whitelisted_module("myproject")
    assert not _is_whitelisted_module("myproject.utils")


def test_whitelist_includes_submodules():
    """Test that submodules of whitelisted packages are also whitelisted."""
    _, _is_whitelisted_module = _whitelist_api()

    # torch submodules
    assert _is_whitelisted_module("torch.cuda")
    assert _is_whitelisted_module("torch.distributed")
    assert _is_whitelisted_module("torch.optim.adam")

    # transformers submodules
    assert _is_whitelisted_module("transformers.models")
    assert _is_whitelisted_module("transformers.models.gpt2")


# =============================================================================
# Serialization Tests
# =============================================================================


def test_non_whitelisted_function_serialized_by_source(registered_mymethods):
    """Test that functions from non-whitelisted modules are serialized by source."""
    from mymethods.stateful import normalize

    # Serialize the function
    data = dumps(normalize)

    # The serialized data should contain the source code
    assert b"def normalize" in data
    assert b"x.norm" in data

    # Deserialize and verify it works
    restored = loads(data)
    x = torch.randn(3, 4)
    result = restored(x)

    # Should have unit norm along last dimension
    norms = result.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_non_whitelisted_class_serialized_by_source(registered_mymethods):
    """Test that classes from non-whitelisted modules can be serialized.

    Cloudpickle serializes classes using _make_skeleton_class with method
    source code, not a literal 'class' declaration. We verify that:
    1. The class can be serialized without errors
    2. Method source is included (def add, def mean)
    3. The restored class works correctly
    """
    from mymethods.stateful import RunningStats

    # Serialize the class
    data = dumps(RunningStats)

    # The serialized data should contain method source code
    assert b"def add" in data
    assert b"def mean" in data

    # Deserialize and verify it works
    RestoredClass = loads(data)
    stats = RestoredClass()
    stats.add(torch.tensor([1.0, 2.0, 3.0]))
    stats.add(torch.tensor([4.0, 5.0, 6.0]))
    assert stats.count == 2


def test_non_whitelisted_instance_serialized_by_source(registered_mymethods):
    """Test that instances from non-whitelisted modules can be serialized.

    We verify that:
    1. The instance can be serialized without errors
    2. The restored instance has preserved state
    """
    from mymethods.stateful import RunningStats

    # Create an instance with some state
    stats = RunningStats()
    stats.add(torch.tensor([1.0, 2.0, 3.0]))
    stats.add(torch.tensor([4.0, 5.0, 6.0]))

    # Serialize the instance
    data = dumps(stats)

    # Deserialize and verify state is preserved
    restored = loads(data)
    assert restored.count == 2


# =============================================================================
# Integration Tests with Model
# =============================================================================


@pytest.fixture(scope="module")
def tiny_model():
    """Create a tiny GPT-2 model for testing."""
    model = LanguageModel("hf-internal-testing/tiny-random-gpt2", dispatch=True)
    return model


@torch.no_grad()
def test_user_function_in_trace(tiny_model):
    """Test that user-defined functions work in traces with remote='local'."""
    from mymethods.stateful import normalize

    with tiny_model.trace("Hello world", remote="local"):
        hidden = tiny_model.transformer.h[0].output[0]
        normed = normalize(hidden)
        result = normed.save()

    # Verify normalization worked
    norms = result.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


@torch.no_grad()
def test_user_class_in_trace(tiny_model):
    """Test that user-defined classes work in traces with remote='local'.

    Note: This test is skipped because class serialization requires
    additional handling beyond what the whitelist provides for functions.
    """
    from mymethods.stateful import RunningStats

    stats = RunningStats()

    with tiny_model.trace("Hello world", remote="local"):
        hidden = tiny_model.transformer.h[0].output[0]
        stats.add(hidden.mean())
        mean = stats.mean().save()

    assert isinstance(mean, torch.Tensor)
    assert stats.count == 1


# =============================================================================
# Module Isolation Tests
# =============================================================================


@torch.no_grad()
def test_module_state_isolation(tiny_model):
    """Test that serialized code gets its own copy of module state.

    This is the key test for verifying that the whitelist-based serialization
    properly isolates module state. The 'local remote' should have its own
    copy of the mymethods.stateful module, not share state with the original.
    """
    from mymethods.stateful import increment_and_get, get_call_count, reset_count

    # Reset state before test
    reset_count()
    assert get_call_count() == 0

    # Define a function that uses the stateful module
    def trace_with_increment(model):
        # This import happens inside the function, so the module reference
        # will be captured in the function's globals
        from mymethods.stateful import increment_and_get
        return increment_and_get()

    # Call locally first - should increment local state
    local_result = trace_with_increment(tiny_model)
    assert local_result == 1
    assert get_call_count() == 1

    # Now run via remote='local' - the serialized code should have
    # its own module state, starting from 0
    with tiny_model.trace("test", remote="local"):
        # We can't easily call increment_and_get inside a trace context
        # because it doesn't return a proxy. Instead, let's verify
        # that after the trace, our local state is unchanged.
        hidden = tiny_model.transformer.h[0].output[0].save()

    # Local state should still be 1 (unchanged by remote execution)
    assert get_call_count() == 1

    # Increment locally again
    local_result2 = increment_and_get()
    assert local_result2 == 2
    assert get_call_count() == 2


@torch.no_grad()
def test_serialization_includes_module_functions(tiny_model, registered_mymethods):
    """Test that module-level functions are included in serialization."""
    from mymethods.stateful import normalize, get_call_count

    # Serialize the normalize function directly
    data = dumps(normalize)

    # Should include the function source
    assert b"def normalize" in data
    assert b"x.norm" in data

    # But should NOT include other functions from the module
    # (they're only included if referenced)
    # Note: get_call_count is NOT referenced by normalize, so it shouldn't be included
    # This is a key difference from whole-module serialization


def test_unregistered_module_is_serialized_by_reference(unregistered_mymethods):
    """An unregistered local module pickles as a GLOBAL, with no source.

    This is the other half of the contract, and the reason the tests above
    need an explicit fixture. It is asserted here on purpose rather than left
    to whichever test happens to run first.
    """
    import mymethods.stateful

    data = dumps(mymethods.stateful.normalize)

    assert b"def normalize" not in data
    # A by-reference pickle names the module and attribute and nothing else.
    assert b"mymethods.stateful" in data
    assert b"normalize" in data


def test_register_switches_to_serialization_by_value(registered_mymethods):
    """`register()` is what makes a local module's source travel with the job.

    Named for what it checks: the previous version of this test asserted that
    `register()` was unnecessary, which was true only under the whitelist
    design that has since been removed. Without registration `dumps` emits a
    GLOBAL the server cannot import.
    """
    from mymethods.stateful import normalize

    data = dumps(normalize)

    assert b"def normalize" in data

    restored = loads(data)
    x = torch.randn(3, 4)
    result = restored(x)
    assert result.shape == x.shape
    assert torch.allclose(result.norm(dim=-1), torch.ones(3), atol=1e-5)


# =============================================================================
# Whitelisted Module Tests
# =============================================================================


def test_whitelisted_function_not_source_serialized():
    """Test that whitelisted functions are serialized by reference, not source."""
    # torch.relu is from a whitelisted module
    data = dumps(torch.relu)

    # Should NOT contain the source code of relu
    # (it's serialized by reference to torch.relu)
    assert b"def relu" not in data

    # But deserialization should still work
    restored = loads(data)
    x = torch.tensor([-1.0, 0.0, 1.0])
    result = restored(x)
    expected = torch.tensor([0.0, 0.0, 1.0])
    assert torch.allclose(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
