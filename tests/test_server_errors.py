"""Unit tests for ``intervention.errors`` — the shared server↔client
deferred-exception helpers.

These tests pin the contract that every serve path (vLLM, future HF) relies
on: ``capture_deferred`` serializes a mediator's deferred exception into
a wire-safe dict, and ``surface_server_errors`` reconstructs a
``RuntimeError`` with type/message/traceback preserved, while filtering
control-flow entries (``EarlyStopException``).
"""

from __future__ import annotations

import pytest

from nnsight.intervention.errors import (
    capture_deferred,
    surface_server_errors,
)
from nnsight.intervention.interleaver import (
    EarlyStopException,
    _store_deferred_exception,
)


class _FakeMediator:
    """Minimal stand-in with the fields the deferral-site helper writes."""

    def __init__(self):
        self.deferred_exception = None
        self._deferred_type_name = None
        self._deferred_traceback = None
        self._deferred_is_control_flow = False


def _deferred_with(exc: Exception) -> _FakeMediator:
    m = _FakeMediator()
    _store_deferred_exception(m, exc)
    return m


class TestCaptureDeferred:
    def test_no_deferred_returns_none(self):
        m = _FakeMediator()
        assert capture_deferred(m) is None

    def test_captures_type_name_and_message(self):
        try:
            raise IndexError("out of range")
        except IndexError as e:
            m = _deferred_with(e)
        entry = capture_deferred(m, req_id="req-1")
        assert entry["type_name"] == "IndexError"
        assert entry["message"] == "out of range"
        assert entry["req_id"] == "req-1"
        assert entry["is_control_flow"] is False

    def test_traceback_captured_with_real_frames(self):
        try:
            raise ValueError("boom")
        except ValueError as e:
            m = _deferred_with(e)
        entry = capture_deferred(m)
        tb = entry["traceback"]
        # Traceback must include the raising frame — without this, the
        # server boundary hides all user-visible debug info.
        assert "ValueError" in tb
        assert "boom" in tb
        assert "test_server_errors.py" in tb

    def test_early_stop_marked_as_control_flow(self):
        try:
            raise EarlyStopException()
        except EarlyStopException as e:
            m = _deferred_with(e)
        entry = capture_deferred(m)
        assert entry["is_control_flow"] is True


class TestSurfaceServerErrors:
    def test_empty_returns_silently(self):
        surface_server_errors([])
        surface_server_errors(None)

    def test_only_control_flow_returns_silently(self):
        surface_server_errors(
            [{"type_name": "EarlyStopException", "is_control_flow": True, "message": "stop"}]
        )

    def test_raises_runtime_error_preserving_fields(self):
        errors = [
            {
                "req_id": "req-42",
                "type_name": "IndexError",
                "message": "list index out of range",
                "traceback": "Traceback (most recent call last):\n  ...\nIndexError: list index out of range\n",
                "is_control_flow": False,
            }
        ]
        with pytest.raises(RuntimeError) as exc_info:
            surface_server_errors(errors, context="[nnsight-serve]")
        msg = str(exc_info.value)
        # Context prefix
        assert "[nnsight-serve]" in msg
        # Original type and message preserved inside the RuntimeError
        assert "IndexError" in msg
        assert "list index out of range" in msg
        # Request id attributed
        assert "req-42" in msg
        # Traceback appended for debugging
        assert "Traceback" in msg

    def test_control_flow_filtered_before_real_error(self):
        errors = [
            {"type_name": "EarlyStopException", "is_control_flow": True, "message": "stop"},
            {
                "type_name": "AttributeError",
                "message": "no such attr",
                "traceback": "",
                "is_control_flow": False,
            },
        ]
        with pytest.raises(RuntimeError, match="AttributeError"):
            surface_server_errors(errors)

    def test_raises_runtime_error_not_original_type(self):
        """The surface path collapses to RuntimeError — reconstructing the
        original class is brittle across process boundaries.
        """
        errors = [{"type_name": "IndexError", "message": "x", "is_control_flow": False}]
        with pytest.raises(RuntimeError):
            surface_server_errors(errors)
        # Must NOT raise IndexError itself.
        with pytest.raises(Exception) as exc_info:
            surface_server_errors(errors)
        assert type(exc_info.value) is RuntimeError
