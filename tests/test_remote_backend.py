"""The async remote backend's consumption of a job's websocket status stream.

Exercises the new control flow (await for the result, async-iterate the status
updates, raise on error) against a fake websocket connection and a stubbed
download — no server, no real sockets.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from nnsight.intervention.backends.remote import AsyncRemoteBackend, RemoteError
from nnsight.schema.response import ResponseModel, Status

MODEL_KEY = "nnsight.modeling.transformers.TransformersModel:{}"


class _FakeConnection:
    """A websocket stand-in that hands back canned status messages, then records
    that it was closed. recv/close are synchronous, like the real sync client the
    backend connects with (receive() runs recv off the event loop)."""

    def __init__(self, statuses):
        self._messages = [
            ResponseModel(
                id="job",
                status=status,
                description="boom" if status == Status.ERROR else "",
            ).model_dump_json()
            for status in statuses
        ]
        self.closed = False

    def recv(self):
        return self._messages.pop(0)

    def close(self):
        self.closed = True


def _backend(statuses, result=None):
    # Build a backend and drop a fake, already-subscribed connection onto it (so
    # __call__'s real submit is bypassed), stubbing the async download.
    backend = AsyncRemoteBackend(MODEL_KEY, host="http://ndif.test")
    backend.connection = _FakeConnection(statuses)

    async def _download(url):
        return result

    backend.download = _download
    return backend


class TestAsyncRemoteBackend:
    def test_await_returns_result_on_completed(self):
        backend = _backend([Status.RUNNING, Status.COMPLETED], result={"out": 42})
        connection = backend.connection
        result = asyncio.run(backend.resolve())
        assert result == {"out": 42}
        assert connection.closed  # the subscription is closed when done

    def test_await_dunder_is_the_result(self):
        # `await backend` goes through __await__, same as resolve.
        backend = _backend([Status.COMPLETED], result={"x": 1})

        async def go():
            return await backend

        assert asyncio.run(go()) == {"x": 1}

    def test_aiter_yields_each_status_then_the_saves(self):
        backend = _backend(
            [Status.RECEIVED, Status.RUNNING, Status.COMPLETED], result={"y": 2}
        )
        connection = backend.connection

        async def go():
            items = []
            async for item in backend:
                items.append(item)
            return items

        items = asyncio.run(go())
        # Intermediate items are status responses; the final item is the saves dict.
        assert [r.status for r in items[:-1]] == [
            Status.RECEIVED,
            Status.RUNNING,
            Status.COMPLETED,
        ]
        assert items[-1] == {"y": 2}
        assert connection.closed

    def test_error_status_raises_when_awaited(self):
        backend = _backend([Status.RUNNING, Status.ERROR])
        connection = backend.connection
        with pytest.raises(RemoteError, match="boom"):
            asyncio.run(backend.resolve())
        assert connection.closed  # closed even on failure

    def test_stream_yields_error_without_raising(self):
        # stream() hands back raw responses — an ERROR ends the stream (no saves
        # dict), leaving the caller to inspect it and raise if they want.
        backend = _backend([Status.RUNNING, Status.ERROR])
        connection = backend.connection

        async def go():
            items = []
            async for item in backend:
                items.append(item)
            return items

        items = asyncio.run(go())
        assert [r.status for r in items] == [Status.RUNNING, Status.ERROR]
        assert not any(isinstance(i, dict) for i in items)  # no saves on error
        assert connection.closed

    def test_is_a_remote_backend(self):
        from nnsight.intervention.backends.remote import RemoteBackend

        assert issubclass(AsyncRemoteBackend, RemoteBackend)
