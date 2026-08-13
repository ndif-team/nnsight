"""Running a captured trace on a remote NDIF server.

A trace captured locally can be shipped to NDIF and run against a model that
never leaves the server: the block is serialized source-and-all, POSTed, and the
model is named by a ``model_key`` rather than pickled. The backends here differ
only in *how the client waits* for the job to finish and collect its saves:

- [`RemoteBackend`][nnsight.intervention.backends.remote.RemoteBackend] blocking -- one ``/subscribe`` websocket, streamed
  status updates, result on COMPLETED (``model.trace(..., remote=True)``).
- [`RemoteBackend`][nnsight.intervention.backends.remote.RemoteBackend] non-blocking (``blocking=False``) -- submit over HTTP
  and poll the server's stored status until the result lands.
- [`AsyncRemoteBackend`][nnsight.intervention.backends.remote.AsyncRemoteBackend] -- the blocking stream, awaited on an asyncio event
  loop so a job runs without tying up a thread.

Status updates render through
[`StatusDisplay`][nnsight.intervention.backends.display.StatusDisplay]; a failed job
surfaces as [`RemoteError`][nnsight.intervention.backends.remote.RemoteError].
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from uuid import uuid4
from typing import Any, AsyncIterator, Optional

from ...schema.config import CONFIG
from ...schema.response import RESULT, ResponseModel, Status
from ...schema.request import RequestModel
from ...tracing.backend import Backend
from ...tracing.tracer import Tracer, save
from ...tracing.util import push
from .display import StatusDisplay

# Reported to the server (nnsight-version / python-version headers) so it can
# gate incompatible clients when configured. Resolved once at import; empty if
# nnsight isn't installed as a distribution (e.g. running from a source tree).
try:
    _NNSIGHT_VERSION = _pkg_version("nnsight")
except PackageNotFoundError:
    _NNSIGHT_VERSION = ""

# How often (seconds) to wake the recv loop to animate the status spinner.
_REFRESH = 0.1

# HTTP timeouts (seconds) and streamed-download chunk size.
_CONNECT_TIMEOUT = 10.0
_READ_TIMEOUT = 600.0
_CHUNK_SIZE = 128 * 1024


class RemoteError(Exception):
    """The remote job failed on the server."""


class RemoteBackend(Backend):
    """Run a trace on a remote NDIF server, waiting for it in the calling thread.

    Serializes the captured block (the model is named by ``model_key``, never
    serialized), ships it, and returns the saved values once the job finishes.
    Two waiting modes, chosen by ``blocking``:

    - ``blocking=True`` (default): hold one ``/subscribe`` websocket open, stream
      status updates until COMPLETED, then download the saves and push them back
      into the caller's frame so its ``h = ...save()`` variables populate.
    - ``blocking=False``: [`submit`][nnsight.intervention.backends.remote.RemoteBackend.submit] the job over plain HTTP (no websocket)
      and store its [`job_id`][nnsight.intervention.backends.remote.RemoteBackend.job_id]; the server records each status to the object
      store, and each later call [`poll`][nnsight.intervention.backends.remote.RemoteBackend.poll]\\ s until the result is ready.

    [`AsyncRemoteBackend`][nnsight.intervention.backends.remote.AsyncRemoteBackend] runs the same blocking stream on an asyncio event
    loop instead.
    """

    def __init__(
        self,
        model_key: str,
        host: Optional[str] = None,
        api_key: Optional[str] = None,
        env: Optional[dict] = None,
        blocking: bool = True,
        job_id: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.model_key = model_key
        # Per-request environment (e.g. a PEFT adapter) the server applies to the
        # model before running; travels in the request envelope. See RequestModel.env.
        self.env = env or {}
        # Blocking: hold a websocket open until COMPLETED. Non-blocking: submit
        # without a websocket (the server saves each status to the object store),
        # then poll GET /response/{job_id}. `job_id` is set after a submit — or
        # passed in to construct a poll-only backend for an existing job. `status`
        # holds the last status seen while polling.
        self.blocking = blocking
        self.job_id = job_id
        self.status: Optional[Status] = None
        self.host = host or CONFIG.API.HOST
        if not self.host.startswith(("http://", "https://")):
            raise ValueError(
                f"Invalid host URL: {self.host!r}; must start with http:// or https://"
            )
        self.api_key = api_key or CONFIG.API.APIKEY or ""
        self.compress = CONFIG.API.COMPRESS

        # Derive the websocket scheme from the http scheme.
        scheme, _, rest = self.host.partition("://")
        self.ws_host = ("wss" if scheme == "https" else "ws") + "://" + rest

        # Verbose: log payload/result byte sizes and print each status on its own
        # line. Falls back to the app-wide DEBUG flag.
        self.verbose = verbose or CONFIG.APP.DEBUG
        self.display = StatusDisplay(
            enabled=CONFIG.APP.REMOTE_LOGGING or self.verbose, verbose=self.verbose
        )

    #: How the payload travels, learned from the ``/subscribe`` handshake.
    #: ``None`` is the ordinary NDIF — a multipart body up, a presigned url down.
    #: ``"shm"`` is a singleton deployment on this machine, which takes a
    #: shared-memory path both ways (see [`shm`][nnsight.intervention.backends.shm]).
    #:
    #: **Detected, not configured.** The client reads that frame *before* it
    #: sends anything, which makes it the only point where a server can change
    #: how the payload travels for free — by the time a response could say so,
    #: the payload has already gone the slow way. So pointing a client at a
    #: singleton is the whole of the setup; nothing about `remote=True` changes.
    transport: Optional[str] = None

    def _log(self, message: str) -> None:
        """Print a diagnostic line when verbose (payload/result sizes, etc.)."""
        if self.verbose:
            print(f"[remote] {message}")

    def __call__(self, tracer: Optional[Tracer] = None) -> Optional[RESULT]:
        # Non-blocking: submit on the first call (no job_id yet), poll thereafter.
        # A poll returns the saved values on COMPLETED, else None (still running);
        # call again to re-poll. See submit()/poll().
        if not self.blocking:
            return self.submit(tracer) if self.job_id is None else self.poll(tracer)

        result = self.request(
            RequestModel(
                model_key=self.model_key, compress=self.compress, env=self.env
            ),
            tracer,
        )

        # Push returned values back into the caller's frame, like local execute.
        if isinstance(result, dict):
            self._push(result, tracer)
        return result

    def _push(self, result: RESULT, tracer: Tracer) -> None:
        # Blocking path only, which runs inside the trace's __exit__ (so a trace is
        # active): mark the returned values so an enclosing trace/session keeps them
        # — its push_result propagates only save()-marked values — and write them
        # into the caller's frame so its `h = ...save()` variables populate. The
        # non-blocking and async paths run after the trace has exited, so they don't
        # push or mark; they return the saves dict for the caller to read.
        for value in result.values():
            save(value)
        if tracer is not None and tracer.info is not None and tracer.info.frame is not None:
            push(tracer.info.frame, result)

    def note(self, response: ResponseModel) -> bool:
        """Render a status update, raise on ERROR, and report whether it's final.

        The shared status handling for every update, however it arrived (websocket,
        poll): update the display, raise [`RemoteError`][nnsight.intervention.backends.remote.RemoteError] on ERROR, and return
        whether the status is COMPLETED (so the caller knows to fetch the result).
        """
        self.display.update(response)
        if response.status == Status.ERROR:
            raise RemoteError(response.description)
        return response.status == Status.COMPLETED

    def handle(self, response: ResponseModel) -> Optional[RESULT]:
        """Process a single status update, returning the result on COMPLETED.

        Returns None for intermediate statuses. On COMPLETED the server sends a
        presigned url on ``data``; download and load it into the result dict.
        """
        return self.download_result(response.data) if self.note(response) else None

    def download_result(self, url: Optional[str]) -> Optional[RESULT]:
        """Stream and deserialize the result blob from a presigned url.

        Or, against a singleton, read it from the shared-memory path that took
        the url's place.

        Downloads in chunks behind a tqdm progress bar (shown when remote logging
        is enabled), then decompresses under the same flag the request was
        compressed with and loads it with torch.load.
        """
        if not url:
            return None

        # A singleton put the result in shared memory rather than an object
        # store, so `data` is a path on this machine and there is nothing to
        # download -- reading it *is* the transfer, and reading unlinks it,
        # which is what reclaims the memory.
        if self.transport == "shm":
            from . import shm

            content = shm.read(url)
            self._log(f"result: {len(content):,} bytes read from {url}")
            return self.finalize(content)

        import io

        import httpx
        from tqdm import tqdm

        buffer = io.BytesIO()
        timeout = httpx.Timeout(_CONNECT_TIMEOUT, read=_READ_TIMEOUT)
        with httpx.Client(timeout=timeout) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                # Content-Length may be absent; None gives an indeterminate bar.
                total = int(response.headers.get("Content-Length", 0)) or None
                with tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc="⬇ Downloading result",
                    disable=not self.display.enabled,
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                ) as bar:
                    for chunk in response.iter_bytes(chunk_size=_CHUNK_SIZE):
                        buffer.write(chunk)
                        bar.update(len(chunk))

        return self.finalize(buffer.getvalue())

    def finalize(self, content: bytes) -> RESULT:
        """Decompress (if the request was compressed) and load a result blob.

        The shared tail of the sync and async downloads: everything after the bytes
        are in hand.
        """
        import io

        import torch

        self._log(f"result: {len(content):,} bytes downloaded")
        if self.compress:
            import zstandard as zstd

            content = zstd.ZstdDecompressor().decompress(content)

        return torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)

    def _adopt_transport(self, transport: Optional[str], request: RequestModel) -> None:
        """Take the transport the handshake announced, and its consequences.

        Only one so far: a singleton is never compressed. zstd earns its keep by
        shrinking what crosses a wire, and there is none — measured at 180 ms of
        a 470 ms round trip on a 45 MB result. The request carries the flag too,
        because the *server* matches it when it returns the result.
        """
        self.transport = transport
        if transport == "shm":
            self.compress = False
            request.compress = False

    def _post(self, request: RequestModel, blob: bytes) -> ResponseModel:
        """POST the request over HTTP and return the initial (RECEIVED) response.

        RequestModel.metadata() is the JSON routing envelope (form field); blob is
        the serialized payload. The response carries the server-assigned request
        id (used as the job id for non-blocking polling).
        """
        import httpx  # lazy: only needed for actual remote calls

        # Stamp the send time so the server can measure client->server transit as
        # the first hop of the request's status-time breakdown. (Cross-clock, so
        # it inherits any client/server skew.)
        headers = {
            "ndif-timestamp": repr(time.time()),
            "nnsight-version": _NNSIGHT_VERSION,
            "python-version": sys.version,
        }
        if self.api_key:
            headers["ndif-api-key"] = self.api_key

        timeout = httpx.Timeout(_CONNECT_TIMEOUT, read=_READ_TIMEOUT)
        # A singleton takes the payload through shared memory: the same JSON
        # envelope plus where the bytes are, and no body. Written before the POST
        # so the path names something by the time the server reads it.
        payload_path = self._stage(blob) if self.transport == "shm" else None
        try:
            with httpx.Client(timeout=timeout) as client:
                if payload_path is not None:
                    body = json.loads(request.metadata())
                    body["payload_path"] = payload_path
                    http_response = client.post(
                        f"{self.host}/request", json=body, headers=headers
                    )
                else:
                    http_response = client.post(
                        f"{self.host}/request",
                        data={"data": request.metadata()},
                        files={"blob": ("request", blob, "application/octet-stream")},
                        headers=headers,
                    )
            http_response.raise_for_status()
        except httpx.HTTPStatusError as error:
            # The server explains the failure in the JSON body (FastAPI returns
            # {"detail": ...}); surface that instead of just the bare HTTP status
            # line ("401 Unauthorized" / "503 Service Unavailable" tell the user
            # nothing).
            detail = None
            try:
                body = error.response.json()
                detail = body.get("detail") if isinstance(body, dict) else body
            except ValueError:
                detail = error.response.text or None
            raise RemoteError(f"Failed to send request: {detail or error}") from error
        except httpx.HTTPError as error:
            if payload_path is not None:
                # The server never read it, so nobody will: the reader is what
                # reclaims a segment, and there isn't one.
                from . import shm

                shm.discard(payload_path)
            raise RemoteError(f"Failed to send request: {error}") from error

        return ResponseModel.model_validate_json(http_response.text)

    def _stage(self, blob: bytes) -> str:
        """Put the payload in shared memory and return its path.

        The server unlinks it once read, so nothing here has to — except when
        the POST never arrives, which `_post` handles.
        """
        from . import shm

        if not shm.available():
            raise RemoteError(
                f"This deployment is a singleton and hands its payload over "
                f"through {shm.ROOT}, which is not writable here. That directory "
                "is how the two ends share memory, so it only works from the "
                "machine the singleton runs on."
            )
        # The client's RequestModel carries no id -- the server mints one -- so
        # the segment gets a fresh uuid. It only has to be unique against other
        # in-flight requests on this machine.
        path = shm.write(blob, uuid4().hex)
        self._log(f"payload: {len(blob):,} bytes staged at {path}")
        return path

    def send(self, request: RequestModel, blob: bytes) -> Optional[RESULT]:
        """Submit the request and handle the initial (blocking) response."""
        return self.handle(self._post(request, blob))

    def _serialize(self, tracer: Tracer) -> bytes:
        """Register local modules for by-value pickling, then serialize the block."""
        from ...ndif import pull_env

        # Register local (non-installed) modules for serialize-by-value so their
        # source ships with the request (else the server hits ModuleNotFoundError).
        # Cached — the local-env scan runs once per process.
        pull_env()
        blob = RequestModel.serialize(tracer, self.compress)
        self._log(
            f"payload: {len(blob):,} bytes ({'compressed' if self.compress else 'raw'})"
        )
        return blob

    def submit(self, tracer: Tracer) -> None:
        """Submit a non-blocking job: POST without a websocket and store the job id.

        No `/subscribe`, so the request carries no session id; the server saves
        each status response to the object store for [`poll`][nnsight.intervention.backends.remote.RemoteBackend.poll] to read. Returns
        None — call [`poll`][nnsight.intervention.backends.remote.RemoteBackend.poll] (or the backend again) to fetch the result.
        """
        blob = self._serialize(tracer)
        request = RequestModel(
            model_key=self.model_key, compress=self.compress, env=self.env
        )
        response = self._post(request, blob)
        self.job_id = response.id
        self.status = response.status
        self.display.update(response)
        return None

    def poll(self, tracer: Optional[Tracer] = None) -> Optional[RESULT]:
        """Fetch the latest status of a submitted non-blocking job.

        GETs `/response/{job_id}`. Returns the saved-values dict on COMPLETED (the
        trace has long since exited, so the caller reads it from the return rather
        than from any frame), raises on ERROR, and returns None while the job is
        still running (or before its first status lands). ``status`` holds the latest.
        """
        import httpx  # lazy: only needed for actual remote calls

        headers = {"ndif-api-key": self.api_key} if self.api_key else None
        timeout = httpx.Timeout(_CONNECT_TIMEOUT, read=_READ_TIMEOUT)
        with httpx.Client(timeout=timeout) as client:
            http_response = client.get(
                f"{self.host}/response/{self.job_id}", headers=headers
            )
        if http_response.status_code == 404:
            return None  # no status recorded yet
        http_response.raise_for_status()

        response = ResponseModel.model_validate_json(http_response.text)
        self.status = response.status
        return self.handle(response)

    def request(self, request: RequestModel, tracer: Tracer) -> Optional[RESULT]:
        import websocket  # lazy: only needed for actual remote calls

        # Subscribe *before* serializing, not just before sending: the handshake
        # decides whether this is a singleton, and that decides whether the blob
        # is compressed. Serializing first would compress a payload destined for
        # shared memory, which is pure cost -- there is no wire to shrink it for.
        connection = websocket.create_connection(
            f"{self.ws_host}/subscribe",
            header={"ndif-api-key": self.api_key} if self.api_key else None,
        )
        try:
            handshake = json.loads(connection.recv())
            request.session_id = handshake["session_id"]
            self._adopt_transport(handshake.get("transport"), request)

            blob = self._serialize(tracer)

            # The initial response comes back over HTTP.
            self.send(request, blob)

            # Poll the websocket for updates; wake periodically (when logging)
            # to animate the spinner / elapsed time.
            if self.display.enabled:
                connection.settimeout(_REFRESH)
            while True:
                try:
                    message = connection.recv()
                except websocket.WebSocketTimeoutException:
                    self.display.refresh()
                    continue
                response = ResponseModel.model_validate_json(message)
                result = self.handle(response)
                if response.status == Status.COMPLETED:
                    return result
        finally:
            connection.close()


class AsyncRemoteBackend(RemoteBackend):
    """A [`RemoteBackend`][nnsight.intervention.backends.remote.RemoteBackend] whose waiting is async, over the same websocket.

    `__call__` fires the request the way the blocking parent does — subscribe
    to the ``/subscribe`` websocket, take the session id, POST the payload — and
    returns without consuming any status updates. That initial connection is
    synchronous; only the waiting is async. Await the backend for the saves dict —
    which renders the status display and raises on a server error, like the blocking
    parent — or async-iterate it for the *raw* status updates to handle yourself,
    with the saves dict yielded as the final item:

        backend = AsyncRemoteBackend(model.to_model_key())
        with model.trace(prompt, backend=backend):
            out = model.output.save()

        result = await backend                  # wait for COMPLETED, get the saves
        # or
        async for update in backend:            # raw ResponseModel status updates...
            if isinstance(update, dict):
                result = update                 # ...and the saves dict, yielded last
            else:
                print(update.status)            # you decide what to do with each

    The websocket ``recv`` is blocking, so [`receive`][nnsight.intervention.backends.remote.AsyncRemoteBackend.receive] runs it through
    `asyncio.to_thread` to keep the event loop free while a job runs. Only
    that waiting differs from the parent — the connect, serialization, POST, and the
    decompress/load reuse the parent's synchronous methods.

    The caller's frame is gone by the time the result lands (the trace has exited),
    so — like the parent's non-blocking path — the saved values aren't pushed back
    into it; they come out of the await / the iterator's final item.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The open subscription — set by __call__, consumed by the async methods.
        self.connection: Optional[object] = None

    def __call__(self, tracer: Optional[Tracer] = None) -> "AsyncRemoteBackend":
        import websocket  # lazy: only needed for actual remote calls

        # Fire the request synchronously, then return without waiting: subscribe,
        # take the session id, POST the payload. Only the status stream is awaited.
        blob = self._serialize(tracer)
        self.connection = websocket.create_connection(
            f"{self.ws_host}/subscribe",
            header={"ndif-api-key": self.api_key} if self.api_key else None,
        )
        request = RequestModel(
            model_key=self.model_key, compress=self.compress, env=self.env
        )
        request.session_id = json.loads(self.connection.recv())["session_id"]
        self.send(request, blob)
        return self

    def __await__(self):
        # `await backend` -> the final result once the job COMPLETEs.
        return self.resolve().__await__()

    async def resolve(self) -> Optional[RESULT]:
        """Consume status updates until COMPLETED, returning the saves dict.

        The trace has exited, so — like the non-blocking poll — nothing is pushed
        back into a frame; the caller reads the returned dict.
        """
        try:
            while True:
                response = await self.receive()
                if self.note(response):  # display + raise on ERROR; True if COMPLETED
                    return await self.download(response.data)
        finally:
            self.close()

    def __aiter__(self) -> AsyncIterator[Any]:
        # `async for update in backend` -> each status update as it arrives, then the
        # saves dict as the final item once the job COMPLETEs.
        return self.stream()

    async def stream(self) -> AsyncIterator[Any]:
        """Yield each raw status update as it lands, then the saves dict last.

        Unlike [`resolve`][nnsight.intervention.backends.remote.AsyncRemoteBackend.resolve], this doesn't touch the display or raise on ERROR —
        it hands you each [`ResponseModel`][nnsight.schema.response.ResponseModel] as-is to do with as you like. The
        final item (after COMPLETED) is the downloaded saves dict; an ERROR update
        just ends the stream (inspect it and raise yourself if you want).
        """
        try:
            while True:
                response = await self.receive()
                yield response
                if response.status == Status.COMPLETED:
                    yield await self.download(response.data)
                    return
                if response.status == Status.ERROR:
                    return
        finally:
            self.close()

    async def receive(self) -> ResponseModel:
        """Await the next status update off the websocket (blocking recv in a thread)."""
        message = await asyncio.to_thread(self.connection.recv)
        return ResponseModel.model_validate_json(message)

    async def download(self, url: Optional[str]) -> Optional[RESULT]:
        """Async [`download_result`][nnsight.intervention.backends.remote.RemoteBackend.download_result]: stream the blob with an async client, then
        hand it to the parent's shared decompress/load."""
        if not url:
            return None

        import io

        import httpx

        buffer = io.BytesIO()
        timeout = httpx.Timeout(_CONNECT_TIMEOUT, read=_READ_TIMEOUT)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=_CHUNK_SIZE):
                    buffer.write(chunk)

        return self.finalize(buffer.getvalue())

    def close(self) -> None:
        """Close the subscription websocket (idempotent)."""
        if self.connection is not None:
            self.connection.close()
            self.connection = None
