import sys
import time
import os
from . import CONFIG
from typing import Optional

# Check if we're in an interactive terminal that supports ANSI
def _supports_color():
    """Check if the terminal supports color output."""
    if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
        return False
    if os.environ.get('NO_COLOR'):
        return False
    if os.environ.get('FORCE_COLOR'):
        return True
    return True

SUPPORTS_COLOR = _supports_color()

# ANSI color codes
class Colors:
    RESET = '\033[0m' if SUPPORTS_COLOR else ''
    BOLD = '\033[1m' if SUPPORTS_COLOR else ''
    DIM = '\033[2m' if SUPPORTS_COLOR else ''
    
    # Status colors
    CYAN = '\033[36m' if SUPPORTS_COLOR else ''
    YELLOW = '\033[33m' if SUPPORTS_COLOR else ''
    GREEN = '\033[32m' if SUPPORTS_COLOR else ''
    RED = '\033[31m' if SUPPORTS_COLOR else ''
    MAGENTA = '\033[35m' if SUPPORTS_COLOR else ''
    BLUE = '\033[34m' if SUPPORTS_COLOR else ''
    WHITE = '\033[37m' if SUPPORTS_COLOR else ''
    
    # Background colors
    BG_GREEN = '\033[42m' if SUPPORTS_COLOR else ''
    BG_RED = '\033[41m' if SUPPORTS_COLOR else ''


# Status icons (Unicode)
class StatusIcons:
    RECEIVED = "◉"
    QUEUED = "◎"  
    DISPATCHED = "◈"
    RUNNING = "●"
    COMPLETED = "✓"
    ERROR = "✗"
    LOG = "ℹ"
    STREAM = "⇄"
    SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class JobStatusDisplay:
    """Manages single-line status display for remote job execution."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.job_id: Optional[str] = None
        self.spinner_idx = 0
        self.last_status = None
        self._line_written = False
        
    def _format_elapsed(self) -> str:
        """Format elapsed time since job start."""
        if self.start_time is None:
            return "0.0s"
        elapsed = time.time() - self.start_time
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
    
    def _get_status_style(self, status_name: str) -> tuple:
        """Get icon and color for a status."""
        status_map = {
            "RECEIVED": (StatusIcons.RECEIVED, Colors.CYAN),
            "QUEUED": (StatusIcons.QUEUED, Colors.YELLOW),
            "DISPATCHED": (StatusIcons.DISPATCHED, Colors.MAGENTA),
            "RUNNING": (StatusIcons.RUNNING, Colors.BLUE),
            "COMPLETED": (StatusIcons.COMPLETED, Colors.GREEN),
            "ERROR": (StatusIcons.ERROR, Colors.RED),
            "NNSIGHT_ERROR": (StatusIcons.ERROR, Colors.RED),
            "LOG": (StatusIcons.LOG, Colors.DIM),
            "STREAM": (StatusIcons.STREAM, Colors.CYAN),
        }
        return status_map.get(status_name, ("•", Colors.WHITE))
    
    def _get_spinner(self) -> str:
        """Get next spinner frame."""
        spinner = StatusIcons.SPINNER[self.spinner_idx % len(StatusIcons.SPINNER)]
        self.spinner_idx += 1
        return spinner
    
    def _clear_line(self):
        """Clear the current line."""
        sys.stdout.write('\r\033[K')
        sys.stdout.flush()
    
    def update(self, job_id: str, status_name: str, description: str = ""):
        """Update the status display on a single line."""
        
        # Initialize on first call
        if self.start_time is None:
            self.start_time = time.time()
            self.job_id = job_id
        
        icon, color = self._get_status_style(status_name)
        elapsed = self._format_elapsed()
        
        # Build the status line
        # Format: ● STATUS (elapsed) [job_id] description
        
        is_terminal = status_name in ("COMPLETED", "ERROR", "NNSIGHT_ERROR")
        is_active = status_name in ("RUNNING", "DISPATCHED")
        
        # For active states, show spinner
        if is_active:
            prefix = f"{Colors.DIM}{self._get_spinner()}{Colors.RESET}"
        else:
            prefix = f"{color}{icon}{Colors.RESET}"
        
        # Build status text - full job ID shown so users can reference it
        status_text = (
            f"{prefix} "
            f"{Colors.DIM}[{job_id}]{Colors.RESET} "
            f"{color}{Colors.BOLD}{status_name.ljust(10)}{Colors.RESET} "
            f"{Colors.DIM}({elapsed}){Colors.RESET}"
        )
        
        if description:
            status_text += f" {Colors.DIM}{description}{Colors.RESET}"
        
        # Clear line and write new status
        self._clear_line()
        sys.stdout.write(status_text)
        
        # For terminal states, add newline and reset
        if is_terminal:
            sys.stdout.write('\n')
            self._reset()
        
        sys.stdout.flush()
        self._line_written = True
        self.last_status = status_name
    
    def _reset(self):
        """Reset the display state for a new job."""
        self.start_time = None
        self.job_id = None
        self.spinner_idx = 0
        self.last_status = None
        self._line_written = False
    
    def finalize(self):
        """Ensure the display ends with a newline."""
        if self._line_written and self.last_status not in ("COMPLETED", "ERROR", "NNSIGHT_ERROR"):
            sys.stdout.write('\n')
            sys.stdout.flush()
        self._reset()


# Global status display instance
_status_display = JobStatusDisplay()


def log_status(job_id: str, status_name: str, description: str = ""):
    """Log a job status update with nice formatting."""
    if CONFIG.APP.REMOTE_LOGGING:
        _status_display.update(job_id, status_name, description)


def finalize_status():
    """Finalize the status display (ensures newline at end)."""
    if CONFIG.APP.REMOTE_LOGGING:
        _status_display.finalize()


def get_status_display() -> JobStatusDisplay:
    """Get the global status display instance."""
    return _status_display
