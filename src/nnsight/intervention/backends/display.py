"""In-place status line for a remote NDIF job, in a terminal or a notebook.

A remote job moves through a lifecycle (RECEIVED -> QUEUED -> ... -> RUNNING ->
COMPLETED/ERROR); [`StatusDisplay`][nnsight.intervention.backends.display.StatusDisplay] renders that progress as a single
color-coded line that updates in place -- a spinner while a stage is active, the
final icon when it ends. The same line is built with ANSI escapes for a TTY and
re-emitted as HTML for a notebook cell, so one renderer serves both.
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import Optional

from ...schema.response import ResponseModel, Status


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    return get_ipython() is not None


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if _in_notebook():
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

# status -> (icon, ansi color code)
_STYLES = {
    Status.RECEIVED: ("◉", "36"),  # cyan
    Status.QUEUED: ("◎", "33"),  # yellow
    # provisioning -> deploying -> dispatched: light -> deep purple gradient
    Status.PROVISIONING: ("◔", "38;5;141"),  # light purple
    Status.DEPLOYING: ("◑", "38;5;134"),  # medium magenta-purple
    Status.DISPATCHED: ("◈", "38;5;127"),  # deep magenta
    Status.RUNNING: ("●", "34"),  # blue
    Status.COMPLETED: ("✓", "32"),  # green
    Status.ERROR: ("✗", "31"),  # red
    Status.LOG: ("ℹ", "2"),  # dim
}
_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_ACTIVE = {
    Status.QUEUED,
    Status.PROVISIONING,
    Status.DEPLOYING,
    Status.DISPATCHED,
    Status.RUNNING,
}
_TERMINAL = {Status.COMPLETED, Status.ERROR}


class StatusDisplay:
    """In-place, color-coded status line for a remote job (terminal or notebook).

    Active states animate a spinner (advanced by [`refresh`][nnsight.intervention.backends.display.StatusDisplay.refresh] between
    messages); terminal states end the line. LOG messages print on their own
    line and don't replace the current lifecycle status.
    """

    def __init__(self, enabled: bool = True, verbose: bool = False) -> None:
        self.enabled = enabled
        # Verbose: print each status on its own line (a persisted timeline with
        # per-status timers) instead of animating one in-place spinner line.
        self.verbose = verbose
        self.color = _supports_color()
        self.notebook = _in_notebook()
        self.job_start: Optional[float] = None
        self.status_start: Optional[float] = None
        self.response: Optional[ResponseModel] = None
        self.spinner = 0
        self._handle = None  # IPython display handle for in-place notebook updates

    def update(self, response: ResponseModel) -> None:
        if not self.enabled:
            return
        if response.status is Status.LOG:
            self._log(response)
            return
        now = time.time()
        if self.job_start is None:
            self.job_start = now
        changed = self.response is None or response.status is not self.response.status
        if changed:
            self.status_start = now
        self.response = response
        # Verbose keeps every status; render only on a change so repeated
        # heartbeats (e.g. RUNNING) don't spam duplicate lines. Non-verbose
        # re-renders each time to advance the in-place spinner.
        if not self.verbose or changed:
            self._render()

    def refresh(self) -> None:
        # Re-render the current status with no new message (advances the spinner
        # and the elapsed time). A no-op in verbose mode — nothing animates.
        if self.enabled and not self.verbose and self.response is not None:
            self._render()

    # -- rendering -----------------------------------------------------------

    def _render(self) -> None:
        response = self.response
        icon, _ = _STYLES.get(response.status, ("•", "37"))
        terminal = response.status in _TERMINAL

        if response.status in _ACTIVE and not self.verbose:
            symbol = self._style(_SPINNER[self.spinner % len(_SPINNER)], _DIM)
            self.spinner += 1
        else:
            symbol = self._color(icon, response.status)

        elapsed = self._elapsed(self.job_start if terminal else self.status_start)
        line = (
            f"{symbol} "
            f"{self._style('[' + response.id + ']', _DIM)} "
            f"{self._color(response.status.value.ljust(12), response.status, bold=True)} "
            f"{self._style('(' + elapsed + ')', _DIM)}"
        )
        # An ERROR's description is the full traceback, re-raised by the caller
        # as a RemoteError; not rendered here.
        if response.description and response.status is not Status.ERROR:
            line += " " + self._style(response.description, _DIM)

        # Verbose persists every status on its own line (a timeline).
        self._write(line, terminal=terminal, persist=self.verbose)

    def _log(self, response: ResponseModel) -> None:
        # A standalone message: print on its own line, keep the current status.
        line = (
            f"{self._color('ℹ', Status.LOG)} "
            f"{self._style('[' + response.id + ']', _DIM)} "
            f"{self._color(Status.LOG.value.ljust(12), Status.LOG, bold=True)}"
        )
        if response.description:
            line += f" {response.description}"
        self._write(line, terminal=False, persist=True)

    def _write(self, line: str, terminal: bool, persist: bool) -> None:
        if self.notebook:
            self._write_notebook(line, terminal, persist)
        else:
            # End the line for terminal statuses and standalone logs.
            sys.stdout.write("\r\033[K" + line + ("\n" if terminal or persist else ""))
            sys.stdout.flush()

    def _write_notebook(self, line: str, terminal: bool, persist: bool) -> None:
        from IPython.display import HTML, display

        html = HTML(
            f'<pre style="margin:0;font-family:monospace">{_ansi_to_html(line)}</pre>'
        )
        if persist:
            # A standalone log: its own cell, then start fresh for the next status.
            display(html)
            self._handle = None
        elif self._handle is None:
            self._handle = display(html, display_id=True)
        else:
            self._handle.update(html)
        if terminal:
            self._handle = None

    # -- helpers -------------------------------------------------------------

    def _elapsed(self, start: Optional[float]) -> str:
        seconds = 0.0 if start is None else time.time() - start
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"

    def _style(self, text: str, code: str) -> str:
        return f"{code}{text}{_RESET}" if self.color else text

    def _color(self, text: str, status: Status, bold: bool = False) -> str:
        if not self.color:
            return text
        prefix = f"\033[{_STYLES.get(status, ('', '37'))[1]}m" + (_BOLD if bold else "")
        return f"{prefix}{text}{_RESET}"


_ANSI_CSS = {
    "1": "font-weight:bold",
    "2": "opacity:0.6",
    "31": "color:#e74c3c",
    "32": "color:#2ecc71",
    "33": "color:#f39c12",
    "34": "color:#3498db",
    "35": "color:#9b59b6",
    "36": "color:#00bcd4",
    "37": "color:#ecf0f1",
}

# xterm-256 codes used above, mapped to their hex for notebook rendering.
_ANSI_256 = {
    "141": "#af87ff",  # light purple (provisioning)
    "134": "#af5fd7",  # medium magenta-purple (deploying)
    "127": "#af00af",  # deep magenta (dispatched)
}


def _ansi_to_html(text: str) -> str:
    out, depth, i = [], 0, 0
    while i < len(text):
        match = re.match(r"\033\[([0-9;]+)m", text[i:])
        if match:
            codes = match.group(1).split(";")
            j = 0
            while j < len(codes):
                code = codes[j]
                if code == "0":
                    out.append("</span>" * depth)
                    depth = 0
                    j += 1
                elif code == "38" and codes[j + 1 : j + 2] == ["5"]:
                    # 256-color: "38;5;N" -> map N to its hex.
                    css = _ANSI_256.get(codes[j + 2]) if j + 2 < len(codes) else None
                    if css:
                        out.append(f'<span style="color:{css}">')
                        depth += 1
                    j += 3
                elif code in _ANSI_CSS:
                    out.append(f'<span style="{_ANSI_CSS[code]}">')
                    depth += 1
                    j += 1
                else:
                    j += 1
            i += len(match.group(0))
        else:
            out.append({"<": "&lt;", ">": "&gt;", "&": "&amp;"}.get(text[i], text[i]))
            i += 1
    out.append("</span>" * depth)
    return "".join(out)
