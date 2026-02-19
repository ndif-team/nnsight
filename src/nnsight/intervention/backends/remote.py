"""Remote backend for executing nnsight interventions on NDIF servers.

This module provides the infrastructure for submitting intervention requests
to a remote NDIF (Neural Network Distributed Inference Framework) server and
receiving results back. It handles the complete lifecycle of remote job execution:

    1. Serializing and submitting intervention requests via HTTP
    2. Maintaining WebSocket connections for real-time status updates
    3. Downloading and decompressing results
    4. Displaying job progress to users (terminal and notebook)

Architecture:
    The remote execution flow uses a hybrid HTTP/WebSocket approach:
    - HTTP POST to submit the initial request
    - WebSocket for real-time status updates (QUEUED → RUNNING → COMPLETED)
    - HTTP streaming to download large results

    This design allows for efficient long-polling without blocking HTTP connections
    while providing immediate feedback on job status changes.

Key components:
    - RemoteBackend: Main backend class that orchestrates remote execution
    - JobStatusDisplay: Handles user-facing status output (terminal/notebook)
    - LocalTracer: Tracer subclass for handling streamed remote values locally

Modes of operation:
    - Blocking: Wait for job completion via WebSocket (default)
    - Non-blocking: Submit and poll for results separately
    - Async: Asyncio-compatible versions of blocking operations

        Examples:
    >>> from nnsight import NNsight
    >>> model = NNsight(model_key="openai-community/gpt2")
    >>> with model.trace("Hello", remote=True):
    ...     hidden = model.transformer.h[0].output.save()
    >>> print(hidden.shape)
"""

from __future__ import annotations

import inspect
import io
import os
import sys
import time
from sys import version as python_version
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import httpx
import socketio
import torch
import zstandard as zstd
from tqdm.auto import tqdm

from ... import __IPYTHON__, CONFIG, __version__
from ..._c.py_mount import mount, unmount
from ...intervention.serialization import load, save
from ...schema.request import RequestModel
from ...schema.response import RESULT, ResponseModel
from ..tracing.tracer import Tracer
from ..tracing.util import wrap_exception
from ...ndif import get_remote_env, get_local_env, register
from .base import Backend


def _supports_color() -> bool:
    """Check if the terminal supports ANSI color output.

    Checks environment variables and terminal capabilities to determine
    if color output should be enabled. Respects NO_COLOR and FORCE_COLOR
    environment variables per the no-color.org convention.

    Returns:
        True if color output is supported, False otherwise.
    """
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
    """Manages single-line status display for remote job execution.

    Provides a consistent user interface for displaying job status updates
    in both terminal and Jupyter notebook environments. Features include:
    - Animated spinners for active states (QUEUED, RUNNING)
    - Color-coded status indicators
    - Elapsed time tracking per status and total job time
    - In-place line updates (no scrolling spam)
    - Notebook-compatible HTML rendering

    Args:
        enabled: Whether to display status updates. Controlled by CONFIG.APP.REMOTE_LOGGING.
        verbose: If True, preserve each status on its own line instead of overwriting.
    """

    # ANSI escape codes for terminal coloring (empty strings if color not supported)
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

    # Unicode icons for each job status
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
        self._display_handle = None  # IPython DisplayHandle for flicker-free updates

    def _format_time(self, start_time: Optional[float]) -> str:
        """Format elapsed time as human-readable string (e.g., '5.231s', '2m 30s', '1h 5m')."""
        if start_time is None:
            return "0.000s" if self.verbose else "0.0s"
        elapsed = time.time() - start_time
        # Use .3f precision in verbose mode, .1f otherwise
        precision = 3 if self.verbose else 1
        if elapsed < 60:
            return f"{elapsed:.{precision}f}s"
        elif elapsed < 3600:
            mins = int(elapsed // 60)
            secs = elapsed % 60
            if secs < 10:
                return f"{mins}m {secs:.{precision}f}s"
            else:
                return f"{mins}m {secs:.0f}s"
        else:
            hours = int(elapsed // 3600)
            mins = int((elapsed % 3600) // 60)
            return f"{hours}h {mins}m"

    def _format_elapsed(self) -> str:
        """Format time elapsed in current status phase."""
        return self._format_time(self.status_start_time)

    def _format_total(self) -> str:
        """Format total time elapsed since job was submitted."""
        return self._format_time(self.job_start_time)

    def _get_status_style(self, status_name: str) -> Tuple[str, str]:
        """Map status name to its display icon and color."""
        status_map = {
            "RECEIVED": (self.Icons.RECEIVED, self.Colors.CYAN),
            "QUEUED": (self.Icons.QUEUED, self.Colors.YELLOW),
            "DISPATCHED": (self.Icons.DISPATCHED, self.Colors.MAGENTA),
            "RUNNING": (self.Icons.RUNNING, self.Colors.BLUE),
            "COMPLETED": (self.Icons.COMPLETED, self.Colors.GREEN),
            "ERROR": (self.Icons.ERROR, self.Colors.RED),
            "LOG": (self.Icons.LOG, self.Colors.DIM),
            "STREAM": (self.Icons.STREAM, self.Colors.CYAN),
        }
        return status_map.get(status_name, ("•", self.Colors.WHITE))

    def _get_spinner(self) -> str:
        """Get next frame from the braille spinner animation."""
        spinner = self.Icons.SPINNER[self.spinner_idx % len(self.Icons.SPINNER)]
        self.spinner_idx += 1
        return spinner

    def update(self, job_id: str = "", status_name: str = "", description: str = ""):
        """Update the status display with new job information.

        Args:
            job_id: The remote job identifier.
            status_name: Current status (RECEIVED, QUEUED, RUNNING, COMPLETED, ERROR, etc.)
            description: Optional additional context to display.

        If called with no arguments, refreshes the display with the last known status
        (useful for updating spinner animation and elapsed time).
        """
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

        is_terminal = status_name in ("COMPLETED", "ERROR")
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
            # Don't dim LOG descriptions - they contain important user messages
            if is_log:
                status_text += f" {description}"
            else:
                status_text += f" {self.Colors.DIM}{description}{self.Colors.RESET}"

        # Display the status
        # LOG status should print a newline so it's not cleared
        print_newline = is_terminal or is_log
        self._display(status_text, status_changed, print_newline)

        self._line_written = True

    def _display(self, text: str, status_changed: bool, print_newline: bool = False):
        """Route display to appropriate handler based on environment."""
        if __IPYTHON__:
            self._display_notebook(text, status_changed, print_newline)
        else:
            self._display_terminal(text, status_changed, print_newline)

    def _display_terminal(
        self, text: str, status_changed: bool, print_newline: bool = False
    ):
        """Display in terminal using carriage return for in-place updates."""
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
        """Convert ANSI escape codes to HTML span elements with inline CSS."""
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
        """Display in Jupyter notebook using IPython DisplayHandle for flicker-free updates."""
        from IPython.display import HTML, display

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


_PULLED_ENV = False


def pull_env():
    """Pull the NDIF environment information from the remote server, and register any locally-available modules not present remotely."""
    global _PULLED_ENV
    if not _PULLED_ENV:
        local_env = get_local_env()

        for package, version in local_env.get("packages", {}).items():
            if version == "local":
                register(package)

        # remote_env = get_remote_env()
        # local_modules = set(local_env.get("packages", {}).keys())
        # remote_modules = set(remote_env.get("packages", {}).keys())
        # missing_modules = local_modules - remote_modules
        # for module in missing_modules:
        #     register(module)
        _PULLED_ENV = True


class RemoteException(Exception):
    """Exception raised when a remote job fails on the NDIF server.

    Wraps error information returned from the server, including tracebacks
    from the remote execution environment.
    """

    def __init__(self, tb_string: str):
        super().__init__(tb_string)
        self.tb_string = tb_string

    def __str__(self) -> str:
        return self.tb_string


class RemoteBackend(Backend):
    """Backend for executing nnsight interventions on a remote NDIF server.

    This backend serializes intervention graphs and submits them to a remote
    service for execution on cloud-hosted models. It supports both synchronous
    (blocking) and asynchronous execution modes.

    The execution flow:
        1. Serialize the intervention graph via RequestModel
        2. Submit via HTTP POST, receive job ID
        3. Listen for status updates via WebSocket
        4. Download results when job completes

    Args:
        model_key: Identifier for the remote model (e.g., "openai-community/gpt2").
        host: Remote server URL. Defaults to CONFIG.API.HOST.
        blocking: If True (default), wait for job completion. If False, submit
            and return immediately (use job_id to poll for results later).
        job_id: Existing job ID to retrieve results for (non-blocking mode).
        api_key: NDIF API key. Falls back to NDIF_API_KEY env var or CONFIG.
        callback: Optional webhook URL to receive job completion notification.
        verbose: If True, preserve each status update on its own line.

    Attributes:
        address: HTTP address of the remote server.
        ws_address: WebSocket address (derived from HTTP address).
        job_id: Current job ID (set after submission).
        job_status: Last known job status.
        compress: Whether to use zstd compression for requests/responses.
    """

    # Class-level constants for HTTP timeouts
    CONNECT_TIMEOUT: float = 10.0  # Timeout for establishing connection
    READ_TIMEOUT: float = (
        300.0  # Timeout for reading response (5 min for large requests)
    )

    # Type hints for instance attributes
    model_key: str
    address: str
    ws_address: str
    api_key: str
    job_id: Optional[str]
    compress: bool
    blocking: bool
    callback: str
    job_status: Optional[Any]  # ResponseModel.JobStatus
    status_display: JobStatusDisplay

    def __init__(
        self,
        model_key: str,
        host: Optional[str] = None,
        blocking: bool = True,
        job_id: Optional[str] = None,
        api_key: str = "",
        callback: str = "",
        verbose: bool = False,
    ) -> None:

        self.model_key = model_key

        self.address = host or CONFIG.API.HOST

        # Validate URL protocol
        if not self.address.startswith(("http://", "https://")):
            raise ValueError(
                f"Invalid host URL: {self.address}. Must start with http:// or https://"
            )

        self.api_key = (
            api_key or os.environ.get("NDIF_API_KEY", None) or CONFIG.API.APIKEY
        )

        self.job_id = job_id
        self.compress = CONFIG.API.COMPRESS
        self.blocking = blocking
        self.callback = callback

        # Derive WebSocket protocol from HTTP protocol (https → wss, http → ws)
        if self.address.startswith("https://"):
            self.ws_address = "wss://" + self.address[8:]
        else:
            self.ws_address = "ws://" + self.address[7:]

        self.verbose = verbose or CONFIG.APP.DEBUG

        self.job_status = None
        self.status_display = JobStatusDisplay(
            enabled=CONFIG.APP.REMOTE_LOGGING,
            verbose=self.verbose,
        )

    def request(self, tracer: Tracer) -> Tuple[bytes, Dict[str, str]]:
        """Prepare a request payload and headers for submission to the remote server.

        Extracts interventions from the tracer, serializes them into a RequestModel,
        and builds the HTTP headers required by the NDIF API.

        Args:
            tracer: The tracer containing the intervention graph to execute.

        Returns:
            Tuple of (serialized_data, headers_dict) ready for HTTP POST.
        """
        interventions = super().__call__(tracer)

        pull_env()

        data = RequestModel(interventions=interventions, tracer=tracer).serialize(
            self.compress
        )

        if self.verbose:
            print(f"[RemoteBackend] Payload: {len(data)} bytes")

        headers = {
            "nnsight-model-key": self.model_key,
            "nnsight-compress": str(self.compress),
            "nnsight-version": __version__,
            "python-version": python_version,
            "ndif-api-key": self.api_key or "",
            "ndif-timestamp": str(time.time()),
            "ndif-callback": self.callback or "",
        }

        return data, headers

    def __call__(self, tracer: Optional[Tracer] = None) -> Optional[RESULT]:
        """Execute the backend, dispatching to the appropriate request mode.

        Routes to async, blocking, or non-blocking execution based on the
        tracer's configuration and the backend's blocking setting.

        Args:
            tracer: The tracer to execute. May be None for non-blocking result retrieval.

        Returns:
            The execution result, or None if non-blocking and job not yet complete.
        """
        try:
            if tracer is not None and tracer.asynchronous:
                return self.async_request(tracer)

            if self.blocking:
                # Blocking mode: wait for completion via WebSocket
                return self.blocking_request(tracer)
            else:
                # Non-blocking mode: submit or poll for existing job
                return self.non_blocking_request(tracer)
        except RemoteException as e:
            raise wrap_exception(e, None) from None

    def handle_response(
        self, response: ResponseModel, tracer: Optional[Tracer] = None
    ) -> Optional[RESULT]:
        """Process an incoming response from the remote server.

        Handles all response types: status updates, errors, completion, and streaming.
        Updates the status display and takes appropriate action based on job status.

        Args:
            response: The response model from the server.
            tracer: The original tracer, needed for STREAM responses to access model info.

        Returns:
            For COMPLETED status: the result data (URL string or actual data).
            For other statuses: None (job still in progress).

        Raises:
            RemoteException: If the job status is ERROR.
        """
        self.job_status = response.status

        if response.status == ResponseModel.JobStatus.ERROR:
            self.status_display.update(response.id, response.status.name, "")
            raise RemoteException(response.description)

        # Log response for user (skip STREAM status - it's internal)
        if response.status != ResponseModel.JobStatus.STREAM:
            self.status_display.update(
                response.id, response.status.name, response.description or ""
            )

        if response.status == ResponseModel.JobStatus.COMPLETED:
            # Job finished - return the result (may be data or URL to download)
            return response.data

        elif response.status == ResponseModel.JobStatus.STREAM:
            # Server is streaming a function to execute locally
            # This enables hybrid local/remote execution patterns
            model = getattr(tracer, "model", None)

            fn = load(response.data, model)

            local_tracer = LocalTracer(_info=tracer.info)

            local_tracer.execute(fn)

    def submit_request(
        self, data: bytes, headers: Dict[str, Any]
    ) -> Optional[ResponseModel]:
        """Submit the serialized request to the remote server via HTTP POST.

        Args:
            data: Serialized request payload (potentially compressed).
            headers: HTTP headers including API key, version info, etc.

        Returns:
            The initial ResponseModel containing the assigned job ID.

        Raises:
            ConnectionError: If the server returns a non-200 status code.
            httpx.TimeoutException: If the request times out.
        """
        from ...schema.response import ResponseModel

        headers["Content-Type"] = "application/octet-stream"

        timeout = httpx.Timeout(self.CONNECT_TIMEOUT, read=self.READ_TIMEOUT)

        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{self.address}/request",
                content=data,
                headers=headers,
            )

        if response.status_code == 200:
            response_model = ResponseModel(**response.json())

            # Store job ID for subsequent status checks
            self.job_id = response_model.id

            self.handle_response(response_model)

            return response_model
        else:
            # Extract error message from response
            try:
                msg = response.json()["detail"]
            except Exception:
                msg = response.reason_phrase
            raise ConnectionError(msg)

    def get_response(self) -> Optional[RESULT]:
        """Poll the server for the current job status (non-blocking mode).

        Used when not connected via WebSocket to check if a previously
        submitted job has completed.

        Returns:
            The result if job is complete, None otherwise.

        Raises:
            Exception: If the server returns a non-200 status code.
            httpx.TimeoutException: If the request times out.
        """
        from ...schema.response import ResponseModel

        timeout = httpx.Timeout(self.CONNECT_TIMEOUT, read=self.READ_TIMEOUT)

        with httpx.Client(timeout=timeout) as client:
            response = client.get(
                f"{self.address}/response/{self.job_id}",
                headers={"ndif-api-key": self.api_key},
            )

        if response.status_code == 200:
            response_model = ResponseModel(**response.json())
            return self.handle_response(response_model)
        else:
            raise Exception(response.reason_phrase)

    def _decompress_and_load(self, result_bytes: io.BytesIO) -> RESULT:
        """Decompress (if needed) and deserialize result bytes.

        Args:
            result_bytes: BytesIO containing the downloaded result data.

        Returns:
            The deserialized result object.
        """

        if self.verbose:
            result_bytes.seek(0)
            print(f"[RemoteBackend] Result: {result_bytes.getbuffer().nbytes} bytes")

        result_bytes.seek(0)

        # Decompress if compression was enabled
        if self.compress:
            cctx = zstd.ZstdDecompressor()
            dst = io.BytesIO()

            with cctx.stream_writer(dst, closefd=False) as writer:
                while chunk := result_bytes.read(64 * 1024):
                    writer.write(chunk)

            result_bytes.close()
            result_bytes = dst
            result_bytes.seek(0)

        # Deserialize with torch.load (handles tensors and pickled objects)
        result = torch.load(result_bytes, map_location="cpu", weights_only=False)
        result_bytes.close()

        return result

    def _fetch_result_if_url(self, result: Any) -> RESULT:
        """Download result if it's a URL reference, otherwise return as-is.

        Args:
            result: Either the actual result data, a URL string, or a tuple of (url, content_length).

        Returns:
            The deserialized result object.
        """
        if isinstance(result, str):
            return self.get_result(result)
        elif isinstance(result, (tuple, list)):
            return self.get_result(*result)
        return result

    async def _async_fetch_result_if_url(self, result: Any) -> RESULT:
        """Async version of _fetch_result_if_url()."""
        if isinstance(result, str):
            return await self.async_get_result(result)
        elif isinstance(result, (tuple, list)):
            return await self.async_get_result(*result)
        return result

    def get_result(self, url: str, content_length: Optional[float] = None) -> RESULT:
        """Download and deserialize the result from the server.

        For large results, the server returns a URL instead of inline data.
        This method streams the result with a progress bar, decompresses
        if needed, and deserializes via torch.load.

        Args:
            url: URL to download the result from (typically a presigned S3 URL).
            content_length: Optional content length hint for progress bar.

        Returns:
            The deserialized result object.
        """
        result_bytes = io.BytesIO()

        timeout = httpx.Timeout(self.CONNECT_TIMEOUT, read=self.READ_TIMEOUT)

        # Stream download with progress bar
        with httpx.Client(timeout=timeout) as client:
            with client.stream("GET", url) as stream:
                # Handle missing Content-Length header gracefully
                total_size = content_length or float(
                    stream.headers.get("Content-length", 0)
                )

                with tqdm(
                    total=total_size or None,  # None for indeterminate progress
                    unit="B",
                    unit_scale=True,
                    desc="⬇ Downloading",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                ) as progress_bar:
                    for data in stream.iter_bytes(chunk_size=128 * 1024):
                        progress_bar.update(len(data))
                        result_bytes.write(data)

        return self._decompress_and_load(result_bytes)

    async def async_get_result(
        self, url: str, content_length: Optional[float] = None
    ) -> RESULT:
        """Async version of get_result(). See get_result() for full documentation."""
        result_bytes = io.BytesIO()

        timeout = httpx.Timeout(self.CONNECT_TIMEOUT, read=self.READ_TIMEOUT)

        # Stream download with progress bar (async version)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("GET", url) as stream:
                # Handle missing Content-Length header gracefully
                total_size = content_length or float(
                    stream.headers.get("Content-length", 0)
                )

                with tqdm(
                    total=total_size or None,  # None for indeterminate progress
                    unit="B",
                    unit_scale=True,
                    desc="⬇ Downloading",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                ) as progress_bar:
                    async for data in stream.aiter_bytes(chunk_size=128 * 1024):
                        progress_bar.update(len(data))
                        result_bytes.write(data)

        return self._decompress_and_load(result_bytes)

    def blocking_request(self, tracer: Tracer) -> Optional[RESULT]:
        """Execute the intervention request and wait for completion via WebSocket.

        This is the primary execution path for remote jobs. It establishes a
        WebSocket connection for real-time status updates while the job executes
        on the server.

        Args:
            tracer: The tracer containing the intervention graph.

        Returns:
            The execution result after the job completes.

        Raises:
            RemoteException: If the job fails on the server.
        """
        # Establish WebSocket connection for real-time updates
        with socketio.SimpleClient(reconnection_attempts=10) as sio:
            sio.connect(
                self.ws_address,
                socketio_path="/ws/socket.io",
                transports=["websocket"],
                wait_timeout=10,
            )

            # Prepare and submit the request
            data, headers = self.request(tracer)
            headers["ndif-session_id"] = sio.sid  # Link WebSocket to this request
            self.submit_request(data, headers)

            try:
                # Register callback for streaming values back to server
                LocalTracer.register(lambda data: self.stream_send(data, sio))

                # Main event loop: receive status updates until completion
                while True:
                    # Short timeout enables spinner animation updates
                    timeout = None
                    if CONFIG.APP.REMOTE_LOGGING:
                        timeout = 0.001 if self.status_display.verbose else 0.1
                    try:
                        response = sio.receive(timeout=timeout)[1]
                    except socketio.exceptions.TimeoutError:
                        # No message received - refresh display (updates spinner/elapsed time)
                        self.status_display.update()
                        continue

                    # Parse and handle the response
                    response = ResponseModel.unpickle(response)
                    result = self.handle_response(response, tracer=tracer)

                    if result is not None:
                        # Job completed - download result if it's a URL
                        result = self._fetch_result_if_url(result)
                        tracer.push(result)
                        return result

            finally:
                LocalTracer.deregister()

    async def async_request(self, tracer: Tracer) -> Optional[RESULT]:
        """Async version of blocking_request(). See blocking_request() for full documentation."""
        # Establish async WebSocket connection
        async with socketio.AsyncSimpleClient(reconnection_attempts=10) as sio:
            await sio.connect(
                self.ws_address,
                socketio_path="/ws/socket.io",
                transports=["websocket"],
                wait_timeout=10,
            )

            data, headers = self.request(tracer)
            headers["ndif-session_id"] = sio.sid
            self.submit_request(data, headers)

            try:
                LocalTracer.register(lambda data: self.stream_send(data, sio))

                # Async event loop
                while True:
                    # Short timeout enables spinner animation updates
                    timeout = None
                    if CONFIG.APP.REMOTE_LOGGING:
                        timeout = 0.001 if self.status_display.verbose else 0.1
                    try:
                        response = (await sio.receive(timeout=timeout))[1]
                    except socketio.exceptions.TimeoutError:
                        self.status_display.update()
                        continue

                    response = ResponseModel.unpickle(response)
                    result = self.handle_response(response, tracer=tracer)

                    if result is not None:
                        result = await self._async_fetch_result_if_url(result)
                        tracer.push(result)
                        return result

            finally:
                LocalTracer.deregister()

    def stream_send(self, values: Dict[int, Any], sio: socketio.SimpleClient):
        """Send computed values back to the server during hybrid execution.

        When the server streams a function to execute locally (via STREAM status),
        local results may need to be uploaded back. This method serializes and
        sends those values over the WebSocket connection.

        Args:
            values: Dictionary of values to send (keyed by intervention ID).
            sio: The active WebSocket client connection.
        """
        data = save(values)

        sio.emit(
            "stream_upload",
            data=(data, self.job_id),
        )

    def non_blocking_request(self, tracer: Tracer) -> Optional[RESULT]:
        """Submit a job or poll for results without blocking.

        This mode allows submitting a job and retrieving results in separate calls,
        useful for long-running jobs or when you want to do other work while waiting.

        First call (job_id is None): Submits the request and stores the job_id.
        Subsequent calls (job_id is set): Polls for completion and returns result.

        Args:
            tracer: The tracer to execute (used only on first call).

        Returns:
            None on first call (job submitted).
            The result on subsequent calls if job is complete, None if still running.
        """
        if self.job_id is None:
            # First call: submit the job
            data, headers = self.request(tracer)
            self.submit_request(data, headers)
            # job_id is set by submit_request
        else:
            # Subsequent calls: poll for result
            result = self.get_response()
            if result is not None:
                result = self._fetch_result_if_url(result)
            return result


class LocalTracer(Tracer):
    """Tracer subclass for executing streamed functions locally during hybrid execution.

    When the server streams a function to execute on the client (via STREAM response),
    LocalTracer handles the local execution and sends results back to the server.

    This enables hybrid execution patterns where some computations happen locally
    (e.g., on user's GPU) while others happen remotely.

    Class Attributes:
        _send: Callback function to send values back to the server (set via register()).
    """

    _send: Callable = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.remotes = set()  # Track objects marked for remote upload

    @classmethod
    def register(cls, send_fn: Callable):
        """Register the send callback for uploading values to the server."""
        cls._send = send_fn

    @classmethod
    def deregister(cls):
        """Clear the send callback after execution completes."""
        cls._send = None

    def _save_remote(self, obj: Any):
        """Mark an object for upload back to the server."""
        self.remotes.add(id(obj))

    def execute(self, fn: Callable):
        """Execute a streamed function with remote value tracking.

        Args:
            fn: The function streamed from the server to execute locally.
        """
        # Mount the remote-saving hook so the function can mark values for upload
        mount(self._save_remote, "remote")

        fn(self, self.info)

        unmount("remote")

    def push(self):
        """Push local state and send remote-marked values back to server.

        Inspects the caller's local variables, pushes them to the parent tracer,
        then filters for remote-marked objects and sends them to the server.
        """
        # Find the frame where the traced code is executing
        state_frame = inspect.currentframe().f_back
        state = state_frame.f_locals

        super().push(state)

        # Filter to only objects marked for remote upload
        state = {k: v for k, v in state.items() if id(v) in self.remotes}

        LocalTracer._send(state)
