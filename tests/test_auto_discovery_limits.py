"""
Tests for auto-discovery limits: exploring what CAN and CANNOT be serialized.

These tests explore the boundaries of auto-discovery by testing common
library classes. Some may succeed (and we document why), others fail
(and we document the specific failure mode).

Failure modes:
1. NO_SOURCE: Class is a C extension without Python source
2. COMPLEX_STATE: Instance state contains non-serializable objects
3. IO_STATE: Instance contains file handles, sockets, connections
4. CIRCULAR_REFS: Object graph has cycles we can't handle
5. TOO_LARGE: Would work but dependency graph is impractically large

Run with: pytest tests/test_auto_discovery_limits.py -v

Note: Tests are skipped if the required library is not installed.
"""

import sys
import pytest
import json

sys.path.insert(0, 'src')

from nnsight.intervention.serialization_source import (
    extract_all,
    can_auto_discover,
    is_auto_discoverable_instance,
    SourceSerializationError,
)


def check_serialization(name: str, value, expect_success: bool = False):
    """
    Helper to test serialization and document the result.

    Returns (success, error_message_or_None, variables, remote_objects)
    """
    locals_dict = {name: value}

    try:
        variables, remote_objects, model_refs = extract_all(locals_dict, strict_remote=False)

        # Check if we can JSON-serialize the result (full round-trip test)
        try:
            json.dumps(variables)
            json.dumps(remote_objects)
            return True, None, variables, remote_objects
        except (TypeError, ValueError) as e:
            return False, f"JSON serialization failed: {e}", variables, remote_objects

    except SourceSerializationError as e:
        return False, str(e), None, None


# =============================================================================
# Test 1: transformers.Pipeline
# =============================================================================
# Expected failure mode: COMPLEX_STATE or TOO_LARGE
# - Pure Python, so class IS discoverable
# - But contains deeply nested components (tokenizers, models, preprocessors)
# - Each component has its own class hierarchy and fitted state

@pytest.fixture
def transformers_pipeline():
    """Create a HuggingFace transformers Pipeline."""
    transformers = pytest.importorskip("transformers", reason="transformers not installed")
    # Use a tiny model to avoid large downloads
    try:
        pipe = transformers.pipeline("text-classification", model="prajjwal1/bert-tiny")
        return pipe
    except Exception as e:
        pytest.skip(f"Could not create pipeline: {e}")


def test_transformers_pipeline(transformers_pipeline):
    """
    Test transformers.Pipeline serialization behavior.

    The Pipeline class is pure Python and theoretically discoverable.
    However, it contains deeply nested components (tokenizers, models,
    preprocessors) each with complex internal state.

    This test documents what actually happens - it may succeed or fail
    depending on the specific pipeline configuration.
    """
    pipeline = transformers_pipeline

    success, error, variables, remote_objects = check_serialization('pipeline', pipeline)

    if success:
        # If it succeeds, document that this is a supported case
        pytest.skip(
            "transformers.Pipeline CAN be serialized! "
            "This may be intentional if the dependency graph is manageable."
        )
    else:
        # Document the failure mode
        assert error is not None
        print(f"\ntransformers.Pipeline rejection reason:\n{error[:500]}")
        # Verify error is informative
        assert 'pipeline' in error.lower() or 'cannot' in error.lower()


# =============================================================================
# Test 2: pandas.DataFrame
# =============================================================================
# Expected: EARLY REJECTION via forbidden list
# - DataFrames are on the forbidden list for early, clean rejection
# - Error message suggests converting to tensor with df.values

@pytest.fixture
def pandas_dataframe():
    """Create a pandas DataFrame."""
    pd = pytest.importorskip("pandas", reason="pandas not installed")
    return pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})


def test_pandas_dataframe(pandas_dataframe):
    """
    Test pandas.DataFrame is rejected early with helpful message.

    DataFrames are on the forbidden list because:
    1. They have massive dependency graphs that generate warning cascades
    2. Their internal state (BlockManager) is complex and C-based
    3. Users should convert to tensors before the trace

    The error message should suggest: torch.tensor(df.values)
    """
    df = pandas_dataframe

    success, error, variables, remote_objects = check_serialization('df', df)

    # Should be rejected
    assert not success, "DataFrames should be rejected"
    assert error is not None

    # Error should be actionable
    print(f"\npandas.DataFrame rejection reason:\n{error}")
    assert 'dataframe' in error.lower()
    assert 'torch.tensor' in error or 'tensor' in error.lower()


def test_pandas_series():
    """
    Test pandas.Series is rejected early with helpful message.

    Similar to DataFrame - on forbidden list for early rejection.
    Error message should suggest converting to tensor or list.
    """
    pd = pytest.importorskip("pandas", reason="pandas not installed")
    series = pd.Series([1, 2, 3, 4, 5])

    success, error, variables, remote_objects = check_serialization('series', series)

    assert not success, "Series should be rejected"
    assert error is not None

    print(f"\npandas.Series rejection reason:\n{error}")
    assert 'series' in error.lower()
    assert 'tensor' in error.lower() or 'tolist' in error.lower()


# =============================================================================
# Test 3: matplotlib.Figure
# =============================================================================
# Expected: EARLY REJECTION via forbidden list
# - Figures are on the forbidden list for early, clean rejection
# - Error message suggests using savefig() to get image bytes

@pytest.fixture
def matplotlib_figure():
    """Create a matplotlib Figure."""
    plt = pytest.importorskip("matplotlib.pyplot", reason="matplotlib not installed")
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    return fig


def test_matplotlib_figure(matplotlib_figure):
    """
    Test matplotlib.Figure is rejected early with helpful message.

    Figures are on the forbidden list because:
    1. They have rendering state and callbacks that can't be transferred
    2. They generate large warning cascades during auto-discovery
    3. Users should save to bytes with savefig() if they need image data

    The error message should suggest: fig.savefig(buf, format='png')
    """
    fig = matplotlib_figure

    success, error, variables, remote_objects = check_serialization('fig', fig)

    assert not success, "Figures should be rejected"
    assert error is not None

    print(f"\nmatplotlib.Figure rejection reason:\n{error}")
    assert 'figure' in error.lower()
    assert 'savefig' in error.lower()


def test_matplotlib_axes(matplotlib_figure):
    """
    Test matplotlib.Axes is rejected early with helpful message.

    Same as Figure - on forbidden list for early rejection.
    """
    fig = matplotlib_figure
    ax = fig.axes[0]

    success, error, variables, remote_objects = check_serialization('ax', ax)

    assert not success, "Axes should be rejected"
    assert error is not None

    print(f"\nmatplotlib.Axes rejection reason:\n{error}")
    assert 'axes' in error.lower()


# =============================================================================
# Test 4: requests.Session
# =============================================================================
# Expected failure mode: IO_STATE or possibly SUCCESS
# - Sessions are pure Python and theoretically serializable
# - But connection pools contain socket state
# - The SEMANTIC issue: serialized session can't reuse connections
# - This might actually work but be semantically wrong

@pytest.fixture
def requests_session():
    """Create a requests Session."""
    requests = pytest.importorskip("requests", reason="requests not installed")
    session = requests.Session()
    session.headers.update({'User-Agent': 'test'})
    return session


def test_requests_session(requests_session):
    """
    Test requests.Session serialization behavior.

    Sessions are pure Python, so the class may be discoverable.
    The issue is SEMANTIC: connection pools can't be transferred.

    If this SUCCEEDS, it means we serialize the Session class and state,
    but the deserialized session would create fresh connections (which
    might actually be the desired behavior on a remote server).
    """
    session = requests_session

    success, error, variables, remote_objects = check_serialization('session', session)

    if success:
        # This might actually be OK! A fresh session on the server makes sense.
        print(
            "\nrequests.Session CAN be serialized! "
            "Deserialized session will create fresh connections on server."
        )
        # Document what got serialized
        if remote_objects:
            print(f"Remote objects: {list(remote_objects.keys())}")
    else:
        assert error is not None
        print(f"\nrequests.Session rejection reason:\n{error[:500]}")


def test_requests_prepared_request():
    """
    Test requests.PreparedRequest serialization behavior.

    PreparedRequests are data objects (method, url, headers, body)
    that might actually serialize successfully.
    """
    requests = pytest.importorskip("requests", reason="requests not installed")

    session = requests.Session()
    req = requests.Request('GET', 'http://example.com', headers={'X-Test': 'value'})
    prepared = session.prepare_request(req)

    success, error, variables, remote_objects = check_serialization('prepared', prepared)

    if success:
        print("\nrequests.PreparedRequest CAN be serialized!")
        # This is actually useful - you can prepare a request locally
        # and execute it on the server
    else:
        assert error is not None
        print(f"\nrequests.PreparedRequest rejection reason:\n{error[:500]}")


# =============================================================================
# Test 5: scikit-learn models
# =============================================================================
# Expected failure mode: COMPLEX_STATE or NO_SOURCE
# - Fitted models contain numpy arrays from training (coef_, intercept_)
# - Often have estimator chains (Pipeline) with multiple fitted components
# - Some use Cython extensions without Python source

@pytest.fixture
def sklearn_model():
    """Create a fitted scikit-learn model."""
    sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed")
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    model = LogisticRegression(max_iter=100)
    model.fit(X, y)
    return model


def test_sklearn_model(sklearn_model):
    """
    Test scikit-learn LogisticRegression serialization behavior.

    Fitted models contain internal numpy arrays (coef_, intercept_)
    and often use Cython extensions. The class hierarchy is complex and
    fitted state is not cleanly separable from class definition.

    NOTE: For sklearn models, the right approach is to use joblib.dump()
    or pickle the model and pass the bytes, or compute predictions locally.
    """
    model = sklearn_model

    success, error, variables, remote_objects = check_serialization('model', model)

    if success:
        pytest.skip(
            "sklearn.LogisticRegression CAN be serialized! "
            "This suggests we handle fitted state or the class is simpler than expected."
        )
    else:
        assert error is not None
        print(f"\nsklearn.LogisticRegression rejection reason:\n{error[:500]}")


def test_sklearn_pipeline():
    """
    Test scikit-learn Pipeline serialization behavior.

    Pipelines chain multiple estimators, each with its own
    fitted state and class dependencies. The dependency graph can explode.

    NOTE: For sklearn Pipelines, serialize the entire trained pipeline
    with joblib and pass bytes, or compute predictions locally.
    """
    sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed")
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=100, n_features=10, random_state=42)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=100))
    ])
    pipe.fit(X, y)

    success, error, variables, remote_objects = check_serialization('pipe', pipe)

    if success:
        pytest.skip(
            "sklearn.Pipeline CAN be serialized! "
            "This suggests the dependency graph is manageable."
        )
    else:
        assert error is not None
        print(f"\nsklearn.Pipeline rejection reason:\n{error[:500]}")


# =============================================================================
# Additional edge cases - OS/system resources
# =============================================================================
# These are fundamentally non-serializable: they represent OS-level resources
# (file handles, sockets, locks) or execution state (generators).

def test_file_handle_rejected():
    """
    Test that file handles are rejected.

    Reason: File handles represent OS-level resources that cannot be
    transferred between processes, let alone machines.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test")
        # Don't close yet - we want to test the open file handle
        success, error, variables, remote_objects = check_serialization('file_handle', f)

        if success:
            pytest.fail(
                "File handles should NOT be serializable! "
                "They represent OS resources that cannot be transferred."
            )
        else:
            assert error is not None
            print(f"\nFile handle rejection reason:\n{error[:500]}")


def test_threading_lock_rejected():
    """
    Test that threading primitives are rejected.

    Reason: Locks, semaphores, and other threading primitives are
    process-local and cannot be serialized.
    """
    import threading

    lock = threading.Lock()

    success, error, variables, remote_objects = check_serialization('lock', lock)

    if success:
        pytest.fail(
            "Threading locks should NOT be serializable! "
            "They are process-local resources."
        )
    else:
        assert error is not None
        print(f"\nThreading lock rejection reason:\n{error[:500]}")


def test_socket():
    """
    Test socket is rejected early via forbidden module list.

    Sockets are on the forbidden module list because:
    1. They represent OS-level network resources that cannot be transferred
    2. Previously they would "serialize" with warnings but fail at deserialization
    3. Now they are rejected early with a clear message

    The error message should explain that sockets are OS resources.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        success, error, variables, remote_objects = check_serialization('sock', sock)

        assert not success, "Sockets should be rejected"
        assert error is not None

        print(f"\nSocket rejection reason:\n{error}")
        assert 'socket' in error.lower()
        assert 'os' in error.lower() or 'system' in error.lower() or 'resource' in error.lower()
    finally:
        sock.close()


def test_generator_rejected():
    """
    Test that generators are rejected.

    Reason: Generators have execution state (instruction pointer, locals)
    that cannot be serialized. They must be consumed locally.
    """
    def my_generator():
        for i in range(10):
            yield i

    gen = my_generator()
    next(gen)  # Advance to make state non-trivial

    success, error, variables, remote_objects = check_serialization('gen', gen)

    if success:
        pytest.fail(
            "Generators should NOT be serializable! "
            "They have execution state that cannot be transferred."
        )
    else:
        assert error is not None
        print(f"\nGenerator rejection reason:\n{error[:500]}")


# =============================================================================
# Error message quality tests
# =============================================================================

def test_error_message_suggests_alternatives():
    """
    Test that error messages provide helpful alternatives.
    """
    import threading
    lock = threading.Lock()

    locals_dict = {'my_lock': lock}

    with pytest.raises(SourceSerializationError) as exc_info:
        extract_all(locals_dict, strict_remote=False)

    error_msg = str(exc_info.value)

    # Should mention at least one of these alternatives
    alternatives = [
        'json',
        'JSON',
        '@remote',
        '@nnsight.remote',
        'compute',
        'inside the trace',
    ]

    has_alternative = any(alt.lower() in error_msg.lower() for alt in alternatives)
    assert has_alternative, f"Error message should suggest alternatives. Got: {error_msg}"


def test_error_message_identifies_variable():
    """
    Test that error messages clearly identify which variable caused the problem.
    """
    import threading
    problematic_lock = threading.Lock()

    locals_dict = {
        'good_value': 42,
        'another_good': "string",
        'problematic_lock': problematic_lock,
    }

    with pytest.raises(SourceSerializationError) as exc_info:
        extract_all(locals_dict, strict_remote=False)

    error_msg = str(exc_info.value)

    # Should clearly identify the problematic variable
    assert 'problematic_lock' in error_msg, (
        f"Error should name the problematic variable. Got: {error_msg}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
