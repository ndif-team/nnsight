"""Tests for I2: deserialize info-leak — generic 400 to client, full
traceback in server log only.

Background
----------
Both serve paths (HF vanilla, vLLM) previously raised
``HTTPException(status_code=400, detail=f"Deserialization failed: {e}")``
on any failure inside ``RequestModel.deserialize(...)``. The interpolated
``str(e)`` leaks server internals: ``pickle.UnpicklingError`` typically
embeds module paths and file offsets; ``AttributeError`` from a custom
``__reduce__`` chain leaks installed module names; even ``EOFError``
hints at the parser's depth. A malicious client can probe by submitting
deliberately malformed bytes and reading the response.

Fix: ``intervention/errors.py::log_invalid_payload`` logs the full
exception traceback server-side under a generated ``request_id`` and
returns only the generic ``"Invalid request payload (request_id=...)"``
string for the client. Operators correlate by grepping the log.

These tests pin down the helper's contract. End-to-end coverage (POST
malformed bytes, get a 400 with no leak) requires a running server and
is exercised by the existing ``test_serve.py`` / ``test_hf_serve.py``
integration suites.
"""

from __future__ import annotations

import logging
import pickle


def test_returns_generic_message_with_request_id():
    """Client-facing return value contains only the generic prefix and
    the request_id — never the exception class name, message, or
    traceback fragments.
    """
    from nnsight.intervention.errors import log_invalid_payload

    logger = logging.getLogger("nnsight.test.i2")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    request_id = "abc-123"
    try:
        raise pickle.UnpicklingError(
            "invalid load key, '\\xff' (the secret module path is /private/server/internals.so)"
        )
    except pickle.UnpicklingError as e:
        detail = log_invalid_payload(logger, request_id, e)

    assert detail == f"Invalid request payload (request_id={request_id})"

    # Negative assertions — none of these should appear in the
    # client-visible return value.
    forbidden = [
        "UnpicklingError",
        "invalid load key",
        "/private/server/internals.so",
        "Traceback",
        "pickle",
    ]
    for token in forbidden:
        assert token not in detail, (
            f"client-facing detail must not echo {token!r}; got: {detail!r}"
        )


def test_logs_request_id_and_exception_class_server_side():
    """Server log entry must contain the request_id (so operators can
    correlate the client's 400 to the server-side traceback) and the
    exception class (so they know what kind of failure they're looking
    at without having to read the full traceback).
    """
    from nnsight.intervention.errors import log_invalid_payload

    logger = logging.getLogger("nnsight.test.i2.logged")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger.addHandler(_Capture())

    request_id = "req-deadbeef"
    try:
        raise EOFError("ran out of input")
    except EOFError as e:
        log_invalid_payload(logger, request_id, e)

    assert records, "expected a log record to be emitted"
    rec = records[-1]
    msg = rec.getMessage()
    assert request_id in msg
    assert "EOFError" in msg
    # `logger.exception` attaches the traceback as exc_info on the record.
    assert rec.exc_info is not None, (
        "log record must carry exc_info so the full traceback is in the "
        "server log (operator can read it; client cannot)"
    )


def test_handles_arbitrary_exception_classes():
    """The helper accepts any BaseException subclass — pickle can raise
    AttributeError / TypeError / ImportError / KeyError / ... depending
    on what the malicious payload triggers in find_class. The helper
    must not branch on type.
    """
    from nnsight.intervention.errors import log_invalid_payload

    logger = logging.getLogger("nnsight.test.i2.types")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    cases = [
        AttributeError("module 'os' has no attribute 'foo'"),
        TypeError("cannot unpack non-iterable NoneType object"),
        ImportError("No module named 'attacker.probe'"),
        KeyError("missing-persistent-id"),
        ValueError("unsupported pickle protocol: 99"),
        # Even a base Exception works — we don't want a future change
        # to narrow this.
        Exception("generic"),
    ]

    for exc in cases:
        try:
            raise exc
        except BaseException as e:
            detail = log_invalid_payload(logger, "rid", e)
        # Same shape regardless of type.
        assert detail == "Invalid request payload (request_id=rid)"
