"""
Tests for forbidden serialization mechanism.

This test module verifies that:
1. Forbidden objects are caught EARLY with clear, actionable error messages
2. No warning cascade is generated (fast fail)
3. Error messages include helpful suggestions for alternatives
4. Allowed objects (tensors, user classes, etc.) still work

Run with: pytest tests/test_forbidden_serialization.py -v
"""

import sys
import warnings

import pytest
import torch

sys.path.insert(0, 'src')

from nnsight.intervention.serialization_source import (
    extract_all,
    is_forbidden_for_serialization,
    check_forbidden_or_raise,
    SourceSerializationError,
    FORBIDDEN_CLASSES,
    FORBIDDEN_MODULE_PREFIXES,
)


# =============================================================================
# Test Forbidden Detection Function
# =============================================================================

class TestIsForbiddenForSerialization:
    """Tests for the is_forbidden_for_serialization() function."""

    def test_regular_object_not_forbidden(self):
        """Test that regular user objects are not forbidden."""
        class MyClass:
            pass

        obj = MyClass()
        is_forbidden, message = is_forbidden_for_serialization(obj)
        assert is_forbidden is False
        assert message is None

    def test_tensor_not_forbidden(self):
        """Test that tensors are not forbidden."""
        tensor = torch.randn(10)
        is_forbidden, message = is_forbidden_for_serialization(tensor)
        assert is_forbidden is False

    def test_list_not_forbidden(self):
        """Test that lists are not forbidden."""
        is_forbidden, message = is_forbidden_for_serialization([1, 2, 3])
        assert is_forbidden is False

    def test_socket_is_forbidden(self):
        """Test that sockets are forbidden."""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            is_forbidden, message = is_forbidden_for_serialization(sock)
            assert is_forbidden is True
            assert "OS/system resources" in message
            assert "cannot be transferred" in message
        finally:
            sock.close()

    def test_multiprocessing_queue_is_forbidden(self):
        """Test that multiprocessing queues are forbidden."""
        import multiprocessing
        queue = multiprocessing.Queue()
        is_forbidden, message = is_forbidden_for_serialization(queue)
        assert is_forbidden is True
        assert "OS/system resources" in message

    def test_subprocess_is_forbidden(self):
        """Test that subprocess objects would be forbidden."""
        # We don't actually run a subprocess, just check the module detection
        import subprocess
        # The Popen class itself
        is_forbidden, message = is_forbidden_for_serialization(subprocess.Popen)
        assert is_forbidden is True

    @pytest.mark.skipif(
        not any(m.startswith('pandas') for m in sys.modules),
        reason="pandas not imported"
    )
    def test_pandas_dataframe_is_forbidden(self):
        """Test that pandas DataFrames are forbidden."""
        pd = pytest.importorskip("pandas", reason="pandas not installed")
        df = pd.DataFrame({'a': [1, 2, 3]})
        is_forbidden, message = is_forbidden_for_serialization(df)
        assert is_forbidden is True
        assert "pandas.DataFrame" in message
        assert "torch.tensor" in message  # Should suggest conversion

    @pytest.mark.skipif(
        not any(m.startswith('pandas') for m in sys.modules),
        reason="pandas not imported"
    )
    def test_pandas_series_is_forbidden(self):
        """Test that pandas Series are forbidden."""
        pd = pytest.importorskip("pandas", reason="pandas not installed")
        series = pd.Series([1, 2, 3])
        is_forbidden, message = is_forbidden_for_serialization(series)
        assert is_forbidden is True
        assert "pandas.Series" in message

    @pytest.mark.skipif(
        not any(m.startswith('matplotlib') for m in sys.modules),
        reason="matplotlib not imported"
    )
    def test_matplotlib_figure_is_forbidden(self):
        """Test that matplotlib Figures are forbidden."""
        plt = pytest.importorskip("matplotlib.pyplot", reason="matplotlib not installed")
        fig, ax = plt.subplots()
        plt.close(fig)
        is_forbidden, message = is_forbidden_for_serialization(fig)
        assert is_forbidden is True
        assert "Figure" in message
        assert "savefig" in message  # Should suggest saving to bytes

    @pytest.mark.skipif(
        not any(m.startswith('matplotlib') for m in sys.modules),
        reason="matplotlib not imported"
    )
    def test_matplotlib_axes_is_forbidden(self):
        """Test that matplotlib Axes are forbidden."""
        plt = pytest.importorskip("matplotlib.pyplot", reason="matplotlib not installed")
        fig, ax = plt.subplots()
        plt.close(fig)
        is_forbidden, message = is_forbidden_for_serialization(ax)
        assert is_forbidden is True
        assert "Axes" in message


# =============================================================================
# Test Early Rejection in extract_all
# =============================================================================

class TestExtractAllForbiddenRejection:
    """Tests that extract_all rejects forbidden objects early."""

    def test_socket_rejected_early(self):
        """Test that sockets are rejected early with no warnings."""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                with pytest.raises(SourceSerializationError) as exc_info:
                    extract_all({'sock': sock}, strict_remote=False)

                # Should have NO warnings (early rejection)
                serialization_warnings = [
                    x for x in w
                    if "Auto-discovered" in str(x.message)
                ]
                assert len(serialization_warnings) == 0, (
                    f"Expected no auto-discovery warnings, got {len(serialization_warnings)}"
                )

            # Error message should be helpful
            error_msg = str(exc_info.value)
            assert "sock" in error_msg
            assert "OS/system resources" in error_msg
        finally:
            sock.close()

    @pytest.mark.skipif(
        not any(m.startswith('pandas') for m in sys.modules),
        reason="pandas not imported"
    )
    def test_dataframe_rejected_early_no_warnings(self):
        """Test that DataFrames are rejected early with no warning cascade."""
        pd = pytest.importorskip("pandas", reason="pandas not installed")
        df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with pytest.raises(SourceSerializationError) as exc_info:
                extract_all({'df': df}, strict_remote=False)

            # Should have NO auto-discovery warnings (early rejection)
            auto_discovery_warnings = [
                x for x in w
                if "Auto-discovered" in str(x.message)
            ]
            assert len(auto_discovery_warnings) == 0, (
                f"Expected 0 auto-discovery warnings for early rejection, "
                f"got {len(auto_discovery_warnings)}"
            )

        # Error message should suggest alternatives
        error_msg = str(exc_info.value)
        assert "df" in error_msg
        assert "pandas.DataFrame" in error_msg
        assert "torch.tensor" in error_msg

    @pytest.mark.skipif(
        not any(m.startswith('matplotlib') for m in sys.modules),
        reason="matplotlib not imported"
    )
    def test_figure_rejected_early_no_warnings(self):
        """Test that Figures are rejected early with no warning cascade."""
        plt = pytest.importorskip("matplotlib.pyplot", reason="matplotlib not installed")
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                with pytest.raises(SourceSerializationError) as exc_info:
                    extract_all({'fig': fig}, strict_remote=False)

                # Should have NO auto-discovery warnings
                auto_discovery_warnings = [
                    x for x in w
                    if "Auto-discovered" in str(x.message)
                ]
                assert len(auto_discovery_warnings) == 0

            error_msg = str(exc_info.value)
            assert "fig" in error_msg
            assert "Figure" in error_msg
            assert "savefig" in error_msg
        finally:
            plt.close(fig)

    def test_allowed_objects_still_work(self):
        """Test that regular allowed objects are not affected by forbidden checks."""
        # These should all work without being rejected
        locals_dict = {
            'tensor': torch.randn(10),
            'number': 42,
            'string': "hello",
            'list_val': [1, 2, 3],
            'dict_val': {'a': 1, 'b': 2},
        }

        # Should not raise
        variables, remote_objects, model_refs = extract_all(
            locals_dict, strict_remote=False
        )

        assert 'number' in variables
        assert 'string' in variables
        assert 'list_val' in variables
        assert 'dict_val' in variables
        assert 'tensor' in variables


# =============================================================================
# Test Error Message Quality
# =============================================================================

class TestErrorMessageQuality:
    """Tests that error messages are helpful and actionable."""

    def test_socket_error_suggests_alternatives(self):
        """Test that socket errors suggest creating resource inside trace."""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(SourceSerializationError) as exc_info:
                check_forbidden_or_raise('my_socket', sock)

            msg = str(exc_info.value)
            assert "my_socket" in msg
            assert "inside the trace block" in msg.lower() or "Options:" in msg
        finally:
            sock.close()

    @pytest.mark.skipif(
        not any(m.startswith('pandas') for m in sys.modules),
        reason="pandas not imported"
    )
    def test_dataframe_error_shows_tensor_conversion(self):
        """Test that DataFrame errors show how to convert to tensor."""
        pd = pytest.importorskip("pandas", reason="pandas not installed")
        df = pd.DataFrame({'a': [1, 2, 3]})

        with pytest.raises(SourceSerializationError) as exc_info:
            check_forbidden_or_raise('my_df', df)

        msg = str(exc_info.value)
        assert "my_df" in msg
        assert "torch.tensor(df.values)" in msg

    @pytest.mark.skipif(
        not any(m.startswith('matplotlib') for m in sys.modules),
        reason="matplotlib not imported"
    )
    def test_figure_error_shows_savefig(self):
        """Test that Figure errors show how to save to bytes."""
        plt = pytest.importorskip("matplotlib.pyplot", reason="matplotlib not installed")
        fig, _ = plt.subplots()

        try:
            with pytest.raises(SourceSerializationError) as exc_info:
                check_forbidden_or_raise('my_fig', fig)

            msg = str(exc_info.value)
            assert "my_fig" in msg
            assert "savefig" in msg
            assert "BytesIO" in msg
        finally:
            plt.close(fig)

    def test_logging_handler_error_message(self):
        """Test that logging handlers get appropriate error messages."""
        import logging
        handler = logging.StreamHandler()

        with pytest.raises(SourceSerializationError) as exc_info:
            check_forbidden_or_raise('handler', handler)

        msg = str(exc_info.value)
        assert "handler" in msg
        assert "logging" in msg.lower() or "Logging" in msg


# =============================================================================
# Test Forbidden Lists Configuration
# =============================================================================

class TestForbiddenListsConfiguration:
    """Tests for the forbidden lists themselves."""

    def test_forbidden_module_prefixes_is_frozenset(self):
        """Test that FORBIDDEN_MODULE_PREFIXES is immutable."""
        assert isinstance(FORBIDDEN_MODULE_PREFIXES, frozenset)

    def test_key_modules_are_forbidden(self):
        """Test that key problematic modules are in the forbidden list."""
        expected_forbidden = {
            'socket',
            'multiprocessing',
            '_pytest',
            'pytest',
            'subprocess',
            'sqlalchemy',
        }
        for module in expected_forbidden:
            assert module in FORBIDDEN_MODULE_PREFIXES, (
                f"Expected '{module}' to be in FORBIDDEN_MODULE_PREFIXES"
            )

    def test_key_classes_are_forbidden(self):
        """Test that key problematic classes are in the forbidden list."""
        expected_forbidden_prefixes = [
            'pandas.core.frame.DataFrame',
            'pandas.core.series.Series',
            'matplotlib.figure.Figure',
        ]
        for cls_name in expected_forbidden_prefixes:
            assert cls_name in FORBIDDEN_CLASSES, (
                f"Expected '{cls_name}' to be in FORBIDDEN_CLASSES"
            )

    def test_forbidden_class_messages_have_suggestions(self):
        """Test that all forbidden class messages include suggestions."""
        for cls_name, message in FORBIDDEN_CLASSES.items():
            # Each message should have a suggestion or alternative
            assert any(
                keyword in message.lower()
                for keyword in ['convert', 'tensor', 'instead', 'before', 'options', 'save', 'extract', 'access']
            ), f"Message for {cls_name} should include a suggestion. Got: {message[:100]}"


# =============================================================================
# Test Pytest Fixture Leakage Detection
# =============================================================================

class TestPytestFixtureLeakage:
    """Tests that pytest fixtures are properly blocked if they leak into scope."""

    def test_pytest_module_prefix_is_forbidden(self):
        """Test that _pytest module prefix is in forbidden list."""
        assert '_pytest' in FORBIDDEN_MODULE_PREFIXES
        assert 'pytest' in FORBIDDEN_MODULE_PREFIXES

    def test_pytest_fixture_request_detected(self):
        """Test that pytest fixture request objects are detected as forbidden."""
        # Create a mock object that simulates a pytest fixture
        class MockPytestFixture:
            __module__ = '_pytest.fixtures'

        obj = MockPytestFixture()
        is_forbidden, message = is_forbidden_for_serialization(obj)
        assert is_forbidden is True
        assert "test framework" in message.lower()


# =============================================================================
# Integration Test: Mixed Locals
# =============================================================================

class TestMixedLocals:
    """Test extract_all with mixed allowed and forbidden objects."""

    @pytest.mark.skipif(
        not any(m.startswith('pandas') for m in sys.modules),
        reason="pandas not imported"
    )
    def test_forbidden_object_among_allowed_raises(self):
        """Test that one forbidden object causes early failure."""
        pd = pytest.importorskip("pandas", reason="pandas not installed")

        locals_dict = {
            'good_tensor': torch.randn(10),
            'good_number': 42,
            'bad_dataframe': pd.DataFrame({'a': [1]}),  # This should fail
            'good_string': "hello",
        }

        with pytest.raises(SourceSerializationError) as exc_info:
            extract_all(locals_dict, strict_remote=False)

        # Error should mention the DataFrame
        assert "bad_dataframe" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
