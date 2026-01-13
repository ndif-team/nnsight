from __future__ import annotations

import inspect
import io
import os
import sys
import time
from sys import version as python_version
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import httpx
import requests
import socketio
import torch
from tqdm.auto import tqdm

from ... import __IPYTHON__, CONFIG, __version__
from ..._c.py_mount import mount, unmount
from ...intervention.serialization import load, save
from ...schema.request import RequestModel
from ...schema.response import RESULT, ResponseModel
from ..tracing.tracer import Tracer
from .base import Backend


def _supports_color():
    """Check if the terminal supports color output."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    # IPython/Jupyter notebooks support ANSI colors
    if __IPYTHON__:
        return True
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    return True


_SUPPORTS_COLOR = _supports_color()


class JobStatusDisplay:
    """Manages single-line status display for remote job execution."""

    # ANSI color codes
    class Colors:
        RESET = "\033[0m" if _SUPPORTS_COLOR else ""
        BOLD = "\033[1m" if _SUPPORTS_COLOR else ""
        DIM = "\033[2m" if _SUPPORTS_COLOR else ""
        CYAN = "\033[36m" if _SUPPORTS_COLOR else ""
        YELLOW = "\033[33m" if _SUPPORTS_COLOR else ""
        GREEN = "\033[32m" if _SUPPORTS_COLOR else ""
        RED = "\033[31m" if _SUPPORTS_COLOR else ""
        MAGENTA = "\033[35m" if _SUPPORTS_COLOR else ""
        BLUE = "\033[34m" if _SUPPORTS_COLOR else ""
        WHITE = "\033[37m" if _SUPPORTS_COLOR else ""

    # Status icons (Unicode)
    class Icons:
        RECEIVED = "◉"
        QUEUED = "◎"
        DISPATCHED = "◈"
        RUNNING = "●"
        COMPLETED = "✓"
        ERROR = "✗"
        LOG = "ℹ"
        STREAM = "⇄"
        SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, enabled: bool = True, verbose: bool = False):
        self.enabled = enabled
        self.verbose = verbose
        self.job_start_time: Optional[float] = None
        self.status_start_time: Optional[float] = None
        self.spinner_idx = 0
        self.last_response: Optional[Tuple[str, str, str]] = (
            None  # (job_id, status, description)
        )
        self._line_written = False
        self._display_handle = None

    def _format_time(self, start_time: Optional[float]) -> str:
        """Format elapsed time from a given start time."""
        if start_time is None:
            return "0.0s"
        elapsed = time.time() - start_time
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        elif elapsed < 3600:
            mins = int(elapsed // 60)
            secs = elapsed % 60
            return f"{mins}m {secs:.0f}s"
        else:
            hours = int(elapsed // 3600)
            mins = int((elapsed % 3600) // 60)
            return f"{hours}h {mins}m"

    def _format_elapsed(self) -> str:
        """Format elapsed time in current status."""
        return self._format_time(self.status_start_time)

    def _format_total(self) -> str:
        """Format total elapsed time since job started."""
        return self._format_time(self.job_start_time)

    def _get_status_style(self, status_name: str) -> tuple:
        """Get icon and color for a status."""
        status_map = {
            "RECEIVED": (self.Icons.RECEIVED, self.Colors.CYAN),
            "QUEUED": (self.Icons.QUEUED, self.Colors.YELLOW),
            "DISPATCHED": (self.Icons.DISPATCHED, self.Colors.MAGENTA),
            "RUNNING": (self.Icons.RUNNING, self.Colors.BLUE),
            "COMPLETED": (self.Icons.COMPLETED, self.Colors.GREEN),
            "ERROR": (self.Icons.ERROR, self.Colors.RED),
            "NNSIGHT_ERROR": (self.Icons.ERROR, self.Colors.RED),
            "LOG": (self.Icons.LOG, self.Colors.DIM),
            "STREAM": (self.Icons.STREAM, self.Colors.CYAN),
        }
        return status_map.get(status_name, ("•", self.Colors.WHITE))

    def _get_spinner(self) -> str:
        """Get next spinner frame."""
        spinner = self.Icons.SPINNER[self.spinner_idx % len(self.Icons.SPINNER)]
        self.spinner_idx += 1
        return spinner

    def update(self, job_id: str = "", status_name: str = "", description: str = ""):
        """Update the status display on a single line."""
        if not self.enabled:
            return

        # Use last response values if not provided (for refresh calls)
        if not job_id and self.last_response:
            job_id, status_name, description = self.last_response

        if not job_id:
            return

        is_log = status_name == "LOG"

        last_status = self.last_response[1] if self.last_response else None
        # LOG status should not be considered a status change for timer purposes
        status_changed = status_name != last_status and not is_log

        # Track job start time (first status received)
        if last_status is None:
            self.job_start_time = time.time()

        # Reset status timer when status changes (but not for LOG)
        if status_changed:
            self.status_start_time = time.time()

        # Store the response (but not for LOG - so we go back to previous status on refresh)
        if not is_log:
            self.last_response = (job_id, status_name, description)

        icon, color = self._get_status_style(status_name)

        # Build the status line
        # Format: ● STATUS (elapsed) [job_id] description

        is_terminal = status_name in ("COMPLETED", "ERROR", "NNSIGHT_ERROR")
        is_active = status_name in ("QUEUED", "RUNNING", "DISPATCHED")

        # For terminal states, show total time; for others, show status elapsed time
        elapsed = self._format_total() if is_terminal else self._format_elapsed()

        # For active states, show spinner
        if is_active:
            prefix = f"{self.Colors.DIM}{self._get_spinner()}{self.Colors.RESET}"
        else:
            prefix = f"{color}{icon}{self.Colors.RESET}"

        # Build status text - full job ID shown so users can reference it
        # LOG status does not show elapsed time
        if is_log:
            status_text = (
                f"{prefix} "
                f"{self.Colors.DIM}[{job_id}]{self.Colors.RESET} "
                f"{color}{self.Colors.BOLD}{status_name.ljust(10)}{self.Colors.RESET}"
            )
        else:
            status_text = (
                f"{prefix} "
                f"{self.Colors.DIM}[{job_id}]{self.Colors.RESET} "
                f"{color}{self.Colors.BOLD}{status_name.ljust(10)}{self.Colors.RESET} "
                f"{self.Colors.DIM}({elapsed}){self.Colors.RESET}"
            )

        if description:
            status_text += f" {self.Colors.DIM}{description}{self.Colors.RESET}"

        # Display the status
        # LOG status should print a newline so it's not cleared
        print_newline = is_terminal or is_log
        self._display(status_text, status_changed, print_newline)

        self._line_written = True

    def _display(self, text: str, status_changed: bool, print_newline: bool = False):
        """Display text, handling terminal vs notebook environments."""
        if __IPYTHON__:
            self._display_notebook(text, status_changed, print_newline)
        else:
            self._display_terminal(text, status_changed, print_newline)

    def _display_terminal(
        self, text: str, status_changed: bool, print_newline: bool = False
    ):
        """Display in terminal with in-place updates."""
        # In verbose mode, print new line when status changes
        if self.verbose and status_changed and self._line_written:
            sys.stdout.write("\n")
        else:
            # Clear current line for in-place update
            sys.stdout.write("\r\033[K")

        sys.stdout.write(text)

        if print_newline:
            sys.stdout.write("\n")

        sys.stdout.flush()

    def _ansi_to_html(self, text: str) -> str:
        """Convert ANSI color codes to HTML spans."""
        import re

        # Map ANSI codes to CSS styles
        ansi_to_css = {
            "0": "",  # Reset
            "1": "font-weight:bold",  # Bold
            "2": "opacity:0.7",  # Dim
            "31": "color:#e74c3c",  # Red
            "32": "color:#2ecc71",  # Green
            "33": "color:#f39c12",  # Yellow
            "34": "color:#3498db",  # Blue
            "35": "color:#9b59b6",  # Magenta
            "36": "color:#00bcd4",  # Cyan
            "37": "color:#ecf0f1",  # White
        }

        result = []
        open_spans = 0
        i = 0

        while i < len(text):
            # Match ANSI escape sequence
            match = re.match(r"\x1b\[([0-9;]+)m", text[i:])
            if match:
                codes = match.group(1).split(";")
                for code in codes:
                    if code == "0":
                        # Close all open spans
                        result.append("</span>" * open_spans)
                        open_spans = 0
                    elif code in ansi_to_css and ansi_to_css[code]:
                        result.append(f'<span style="{ansi_to_css[code]}">')
                        open_spans += 1
                i += len(match.group(0))
            else:
                # Escape HTML special chars
                char = text[i]
                if char == "<":
                    result.append("&lt;")
                elif char == ">":
                    result.append("&gt;")
                elif char == "&":
                    result.append("&amp;")
                else:
                    result.append(char)
                i += 1

        # Close any remaining spans
        result.append("</span>" * open_spans)
        return "".join(result)

    def _display_notebook(
        self, text: str, status_changed: bool, print_newline: bool = False
    ):
        """Display in notebook using DisplayHandle for flicker-free updates."""
        from IPython.display import display, HTML

        html_text = self._ansi_to_html(text)
        html_content = HTML(
            f'<pre style="margin:0;font-family:monospace;background:transparent;">{html_text}</pre>'
        )

        if self.verbose and status_changed and self._line_written:
            # Verbose mode: create new display for new status, keep old one visible
            self._display_handle = display(html_content, display_id=True)
        elif self._display_handle is None:
            # First display
            self._display_handle = display(html_content, display_id=True)
        elif print_newline:
            # LOG status: create new display so it persists, then reset handle for next status
            display(html_content)
            self._display_handle = None
        else:
            # Update existing display in place (no flicker)
            self._display_handle.update(html_content)


class RemoteException(Exception):
    pass


class RemoteBackend(Backend):
    """Backend to execute a context object via a remote service.

    Context object must inherit from RemoteMixin and implement its methods.

    Attributes:

        url (str): Remote host url. Defaults to that set in CONFIG.API.HOST.
    """

    def __init__(
        self,
        model_key: str,
        host: str = None,
        blocking: bool = True,
        job_id: str = None,
        api_key: str = "",
        callback: str = "",
        verbose: bool = False,
    ) -> None:

        self.model_key = model_key

        self.address = host or CONFIG.API.HOST
        self.api_key = (
            api_key or os.environ.get("NDIF_API_KEY", None) or CONFIG.API.APIKEY
        )

        self.job_id = job_id
        self.zlib = CONFIG.API.ZLIB
        self.blocking = blocking
        self.callback = callback

        # Derive websocket protocol from HTTP protocol
        if self.address.startswith("https://"):
            self.ws_address = "wss://" + self.address[8:]
        else:
            self.ws_address = "ws://" + self.address[7:]

        self.job_status = None
        self.status_display = JobStatusDisplay(
            enabled=CONFIG.APP.REMOTE_LOGGING,
            verbose=verbose,
        )

    def request(self, tracer: Tracer) -> Tuple[bytes, Dict[str, str]]:

        interventions = super().__call__(tracer)

        data = RequestModel(interventions=interventions, tracer=tracer).serialize(
            self.zlib
        )

        headers = {
            "nnsight-model-key": self.model_key,
            "nnsight-zlib": str(self.zlib),
            "nnsight-version": __version__,
            "python-version": python_version,
            "ndif-api-key": self.api_key,
            "ndif-timestamp": str(time.time()),
            "ndif-callback": self.callback,
        }

        return data, headers

    def __call__(self, tracer=None):

        if tracer is not None and tracer.asynchronous:
            return self.async_request(tracer)

        if self.blocking:

            # Do blocking request.
            return self.blocking_request(tracer)

        else:

            # Otherwise we are getting the status / result of the existing job.
            return self.non_blocking_request(tracer)

    def handle_response(
        self, response: ResponseModel, tracer: Optional[Tracer] = None
    ) -> Optional[RESULT]:
        """Handles incoming response data.

        Logs the response object.
        If the job is completed, retrieve and stream the result from the remote endpoint.
        Use torch.load to decode and load the `ResultModel` into memory.
        Use the backend object's .handle_result method to handle the decoded result.

        Args:
            response (Any): Json data to concert to `ResponseModel`

        Raises:
            Exception: If the job's status is `ResponseModel.JobStatus.ERROR`

        Returns:
            ResponseModel: ResponseModel.
        """

        self.job_status = response.status

        if response.status == ResponseModel.JobStatus.ERROR:
            self.status_display.update(response.id, response.status.name, "")
            raise RemoteException(f"{response.description}\nRemote exception.")

        # Log response for user (skip STREAM status - it's internal)
        if response.status != ResponseModel.JobStatus.STREAM:
            self.status_display.update(
                response.id, response.status.name, response.description or ""
            )

        # If job is completed:
        if response.status == ResponseModel.JobStatus.COMPLETED:

            return response.data

        elif response.status == ResponseModel.JobStatus.STREAM:

            model = getattr(tracer, "model", None)

            fn = load(response.data, model)

            local_tracer = LocalTracer(_info=tracer.info)

            local_tracer.execute(fn)

    def submit_request(
        self, data: bytes, headers: Dict[str, Any]
    ) -> Optional[ResponseModel]:
        """Sends request to the remote endpoint and handles the response object.

        Raises:
            Exception: If there was a status code other than 200 for the response.

        Returns:
            (ResponseModel): Response.
        """

        from ...schema.response import ResponseModel

        headers["Content-Type"] = "application/octet-stream"

        response = requests.post(
            f"{self.address}/request",
            data=data,
            headers=headers,
        )

        if response.status_code == 200:

            response = ResponseModel(**response.json())

            self.job_id = response.id

            self.handle_response(response)

            return response

        else:
            try:
                msg = response.json()["detail"]
            except:
                msg = response.reason
            raise ConnectionError(msg)

    def get_response(self) -> Optional[RESULT]:
        """Retrieves and handles the response object from the remote endpoint.

        Raises:
            Exception: If there was a status code other than 200 for the response.

        Returns:
            (ResponseModel): Response.
        """

        from ...schema.response import ResponseModel

        response = requests.get(
            f"{self.address}/response/{self.job_id}",
            headers={"ndif-api-key": self.api_key},
        )

        if response.status_code == 200:

            response = ResponseModel(**response.json())

            return self.handle_response(response)

        else:

            raise Exception(response.reason)

    def get_result(self, url: str, content_length: float = None) -> RESULT:

        result_bytes = io.BytesIO()
        result_bytes.seek(0)
        # Get result from result url using job id.

        with httpx.Client() as client:
            with client.stream("GET", url) as stream:

                # Total size of incoming data.
                total_size = content_length or float(stream.headers["Content-length"])

                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc="⬇ Downloading",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                ) as progress_bar:
                    # chunk_size=None so server determines chunk size.
                    for data in stream.iter_bytes(chunk_size=128 * 1024):
                        progress_bar.update(len(data))
                        result_bytes.write(data)

        # Move cursor to beginning of bytes.
        result_bytes.seek(0)

        # Decode bytes with pickle and then into pydantic object.
        result = torch.load(result_bytes, map_location="cpu", weights_only=False)

        # Close bytes
        result_bytes.close()

        return result

    async def async_get_result(self, url: str, content_length: float = None) -> RESULT:

        result_bytes = io.BytesIO()
        result_bytes.seek(0)
        # Get result from result url using job id.

        async with httpx.AsyncClient() as client:
            async with client.stream("GET", url) as stream:

                # Total size of incoming data.
                total_size = content_length or float(stream.headers["Content-length"])

                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc="⬇ Downloading",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                ) as progress_bar:
                    # chunk_size=None so server determines chunk size.
                    async for data in stream.aiter_bytes(chunk_size=128 * 1024):
                        progress_bar.update(len(data))
                        result_bytes.write(data)

        # Move cursor to beginning of bytes.
        result_bytes.seek(0)

        # Decode bytes with pickle and then into pydantic object.
        result = torch.load(result_bytes, map_location="cpu", weights_only=False)

        # Close bytes
        result_bytes.close()

        return result

    def blocking_request(self, tracer: Tracer) -> Optional[RESULT]:
        """Send intervention request to the remote service while waiting for updates via websocket.

        Args:
            request (RequestModel):Request.
        """

        # Create a socketio connection to the server.
        with socketio.SimpleClient(reconnection_attempts=10) as sio:
            # Connect
            sio.connect(
                self.ws_address,
                socketio_path="/ws/socket.io",
                transports=["websocket"],
                wait_timeout=10,
            )

            data, headers = self.request(tracer)

            headers["ndif-session_id"] = sio.sid

            # Submit request via
            response = self.submit_request(data, headers)

            try:
                LocalTracer.register(lambda data: self.stream_send(data, sio))
                # Loop until
                while True:

                    # Use timeout only when remote logging is enabled to update spinner/elapsed time
                    timeout = 0.1 if CONFIG.APP.REMOTE_LOGGING else None
                    try:
                        response = sio.receive(timeout=timeout)[1]
                    except socketio.exceptions.TimeoutError:
                        # Refresh the status display to update spinner and elapsed time
                        self.status_display.update()
                        continue

                    # Convert to pydantic object.
                    response = ResponseModel.unpickle(response)
                    # Handle the response.
                    result = self.handle_response(response, tracer=tracer)
                    # Break when completed.
                    if result is not None:

                        # If the response has no result data, it was too big and we need to stream it from the server.
                        if isinstance(result, str):
                            result = self.get_result(result)
                        elif isinstance(result, (tuple, list)):
                            result = self.get_result(*result)

                        tracer.push(result)

                        return result

            except Exception as e:

                raise e

            finally:
                LocalTracer.deregister()

    async def async_request(self, tracer: Tracer) -> Optional[RESULT]:
        """Send intervention request to the remote service while waiting for updates via websocket.

        Args:
            request (RequestModel):Request.
        """

        # Create a socketio connection to the server.
        async with socketio.AsyncSimpleClient(reconnection_attempts=10) as sio:
            # Connect
            await sio.connect(
                self.ws_address,
                socketio_path="/ws/socket.io",
                transports=["websocket"],
                wait_timeout=10,
            )

            data, headers = self.request(tracer)

            headers["ndif-session_id"] = sio.sid

            # Submit request via
            response = self.submit_request(data, headers)

            try:
                LocalTracer.register(lambda data: self.stream_send(data, sio))
                # Loop until
                while True:

                    # Use timeout only when remote logging is enabled to update spinner/elapsed time
                    timeout = 0.1 if CONFIG.APP.REMOTE_LOGGING else None
                    try:
                        response = (await sio.receive(timeout=timeout))[1]
                    except socketio.exceptions.TimeoutError:
                        # Refresh the status display to update spinner and elapsed time
                        self.status_display.update()
                        continue

                    # Convert to pydantic object.
                    response = ResponseModel.unpickle(response)
                    # Handle the response.
                    result = self.handle_response(response, tracer=tracer)
                    # Break when completed.
                    if result is not None:

                        # If the response has no result data, it was too big and we need to stream it from the server.
                        if isinstance(result, str):
                            result = await self.async_get_result(result)
                        elif isinstance(result, (tuple, list)):
                            result = await self.async_get_result(*result)

                        tracer.push(result)

                        return result

            except Exception as e:

                raise e

            finally:
                LocalTracer.deregister()

    def stream_send(self, values: Dict[int, Any], sio: socketio.SimpleClient):
        """Upload some value to the remote service for some job id.

        Args:
            value (Any): Value to upload
            job_id (str): Job id.
            sio (socketio.SimpleClient): Connected websocket client.
        """

        data = save(values)

        sio.emit(
            "stream_upload",
            data=(data, self.job_id),
        )

    def non_blocking_request(self, tracer: Tracer):
        """Send intervention request to the remote service if request provided. Otherwise get job status.

        Sets CONFIG.API.JOB_ID on initial request as to later get the status of said job.

        When job is completed, clear CONFIG.API.JOB_ID to request a new job.

        Args:
            request (RequestModel): Request if submitting a new request. Defaults to None
        """

        if self.job_id is None:

            data, headers = self.request(tracer)

            # Submit request via
            response = self.submit_request(data, headers)

            self.job_id = response.id

        else:

            result = self.get_response()

            if isinstance(result, str):
                result = self.get_result(result)
            elif isinstance(result, (tuple, list)):
                result = self.get_result(*result)

            return result


class LocalTracer(Tracer):

    _send: Callable = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.remotes = set()

    @classmethod
    def register(cls, send_fn: Callable):

        cls._send = send_fn

    @classmethod
    def deregister(cls):

        cls._send = None

    def _save_remote(self, obj: Any):

        self.remotes.add(id(obj))

    def execute(self, fn: Callable):

        mount(self._save_remote, "remote")

        fn(self, self.info)

        unmount("remote")

        return

    def push(self):

        # Find the frame where the traced code is executing
        state_frame = inspect.currentframe().f_back

        state = state_frame.f_locals

        super().push(state)

        state = {k: v for k, v in state.items() if id(v) in self.remotes}

        LocalTracer._send(state)
