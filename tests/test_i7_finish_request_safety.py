"""Tests for I7: ``_finish_request`` must not raise into the bg
generation thread.

Background
----------
``VanillaBatchServer._finish_request`` is called from the background
generation thread to signal a completed request. For asyncio callers
it uses ``loop.call_soon_threadsafe(signal.set_result, saves)``. Two
failure modes existed:

1. **Loop is closed.** ``call_soon_threadsafe`` raises
   ``RuntimeError`` synchronously in the bg thread when the asyncio
   loop has been closed (e.g. the FastAPI handler was cancelled, the
   client disconnected mid-stream, or the process is shutting down).
2. **Future already done.** ``signal.set_result(...)`` raises
   ``asyncio.InvalidStateError`` if the future was set / cancelled by
   a parallel path between the ``signal.done()`` check on the bg
   thread and the callback running on the loop thread.

Either raise propagates into the bg generation thread, which is
caught by ``_generation_loop``'s catch-all. Pre-fix, that catch-all
finalized **every** request in ``_active``, including unrelated
in-flight requests batched alongside the disconnected one. A single
client disconnect could tank the entire batch.

Fix: wrap ``call_soon_threadsafe`` in ``try/except RuntimeError``
(log + continue). Wrap ``signal.set_result`` in ``try/except
InvalidStateError`` (silent continue) by routing through a closure
that the loop callback runs.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace


def _make_self() -> SimpleNamespace:
    """Minimal ``self`` for ``_finish_request``. The method only
    touches three dicts and a logger module-global, so we don't need
    a real ``VanillaBatchServer`` (which would require a model)."""
    return SimpleNamespace(
        _active={"r1": object()},
        _results={},
        _result_signals={},
    )


class _BadFuture(asyncio.Future):
    """Future whose ``set_result`` always raises ``InvalidStateError``.

    Simulates the TOCTOU window where the future was completed by a
    parallel path (handler cancellation, duplicate finalize) between
    the bg thread's ``done()`` check and the loop callback running.
    """

    def set_result(self, value):
        raise asyncio.InvalidStateError("simulated already-done future")


def test_closed_loop_does_not_raise_into_bg_thread(caplog):
    """Closed asyncio loop → ``call_soon_threadsafe`` would raise
    ``RuntimeError``. Bg thread must keep running.

    Pre-fix this raise propagated to ``_generation_loop``'s catch-all,
    finalizing every concurrent request with ``__error__`` — one
    client disconnect tanked the whole batch. Post-fix the bg thread
    logs and continues; the disconnected client just doesn't get a
    result (which is correct — they're gone).
    """
    from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer

    loop = asyncio.new_event_loop()
    future = loop.create_future()
    loop.close()

    self_ = _make_self()
    self_._result_signals["r1"] = future

    with caplog.at_level(logging.WARNING, logger="nnsight.modeling.hf_serve.vanilla_server"):
        # Pre-fix: this would raise RuntimeError("Event loop is closed").
        VanillaBatchServer._finish_request(self_, "r1", {"hello": 1})

    # State updates still happen — the request is moved to results and
    # popped from active even though we couldn't signal the (gone)
    # client. Other requests in the batch are unaffected.
    assert self_._results["r1"] == {"hello": 1}
    assert "r1" not in self_._active
    assert "r1" not in self_._result_signals

    # And we logged the situation so operators can grep for it.
    warned = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warned, "expected a warning log when the loop is closed"


def test_already_done_future_does_not_raise_into_bg_thread():
    """Future raises ``InvalidStateError`` from inside the loop
    callback. The closure scheduled via ``call_soon_threadsafe`` must
    swallow it so the loop's exception handler doesn't escalate.

    Drains pending callbacks via ``loop.run_until_complete(asyncio.sleep(0))``
    to force the scheduled closure to run synchronously in the test.
    """
    from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer

    loop = asyncio.new_event_loop()
    try:
        future = _BadFuture(loop=loop)

        # Capture any exception the loop's default handler reports.
        loop_errors: list[dict] = []

        def _trap(loop, context):
            loop_errors.append(context)

        loop.set_exception_handler(_trap)

        self_ = _make_self()
        self_._result_signals["r1"] = future

        # Schedule the threadsafe callback (called from "bg thread";
        # in the test, same thread — fine, it still goes through
        # call_soon_threadsafe → call_soon).
        VanillaBatchServer._finish_request(self_, "r1", {"hi": 2})

        # Drain pending callbacks. If the closure didn't catch
        # InvalidStateError, the loop exception handler fires and
        # `loop_errors` gets a non-empty entry.
        loop.run_until_complete(asyncio.sleep(0))
    finally:
        loop.close()

    # Closure must have caught the InvalidStateError silently.
    invalid_state_seen = [
        e for e in loop_errors
        if isinstance(e.get("exception"), asyncio.InvalidStateError)
    ]
    assert not invalid_state_seen, (
        f"closure leaked InvalidStateError to the loop: {loop_errors!r}"
    )

    # State updates still happened.
    assert self_._results["r1"] == {"hi": 2}


def test_event_signal_path_unchanged():
    """Sync (threading.Event) path is untouched by the fix — make
    sure we didn't accidentally branch through the asyncio handling.
    """
    import threading

    from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer

    event = threading.Event()
    self_ = _make_self()
    self_._result_signals["r1"] = event

    VanillaBatchServer._finish_request(self_, "r1", {"x": 1})

    assert event.is_set()
    assert self_._results["r1"] == {"x": 1}
    assert "r1" not in self_._active


def test_no_signal_for_req_id_is_a_noop():
    """If the request was finalized via a different code path (e.g.
    handler already gave up and removed its signal), ``_finish_request``
    must still be safe to call. State should still update."""
    from nnsight.modeling.hf_serve.vanilla_server import VanillaBatchServer

    self_ = _make_self()
    # No entry in _result_signals.
    VanillaBatchServer._finish_request(self_, "r1", {"x": 1})

    assert self_._results["r1"] == {"x": 1}
    assert "r1" not in self_._active
