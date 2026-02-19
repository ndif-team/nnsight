import contextlib
import ctypes
import inspect
import os
import re
import sys
from builtins import open
from types import FrameType
from typing import TYPE_CHECKING, Callable, Dict, List

if TYPE_CHECKING:
    from .base import Tracer


def indent(source: List[str], indent: int = 1):
    """
    Indents each line in the source list by a specified number of indentation levels.

    Args:
        source: List of strings to indent
        indent: Number of indentation levels to apply (default: 1)

    Returns:
        List of indented strings
    """
    return ["    " * indent + line for line in source]


def try_catch(
    source: List[str],
    exception_source: List[str] = ["raise\n"],
    else_source: List[str] = ["pass\n"],
    finally_source: List[str] = ["pass\n"],
):
    """
    Wraps source code in a try-except-else-finally block.

    Args:
        source: The code to be wrapped in the try block
        exception_source: Code for the except block (default: ["raise\n"])
        else_source: Code for the else block (default: ["pass\n"])
        finally_source: Code for the finally block (default: ["pass\n"])

    Returns:
        List of strings representing the complete try-catch block, properly indented
    """
    source = [
        "try:\n",
        *source,
        "except Exception as exception:\n",
        *indent(exception_source),
        "else:\n",
        *indent(else_source),
        "finally:\n",
        *indent(finally_source),
    ]

    return indent(source)


_devnull = open(os.devnull, "w")


@contextlib.contextmanager
def suppress_all_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = _devnull
        sys.stderr = _devnull
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def get_dependencies(fn: Callable):
    """
    Extracts global dependencies used by a function.

    Args:
        fn: The function to analyze for dependencies

    Returns:
        Dictionary mapping names to their corresponding global objects used by the function
    """
    used_names = fn.__code__.co_names
    return {name: fn.__globals__[name] for name in used_names if name in fn.__globals__}


from ... import CONFIG


class ExceptionWrapper(Exception):
    """
    Wrapper for exceptions that provides additional details for tracer created code.

    This class helps provide better error messages by including source code context
    and proper line numbers from the original code being traced.
    """

    def __init__(self, info: "Tracer.Info", original: Exception, *args, **kwargs):
        """
        Initialize the exception wrapper.

        Args:
            info: Tracer information containing context about where the exception occurred
            original: The original exception being wrapped
            *args, **kwargs: Additional arguments passed to the parent Exception class
        """
        super().__init__(*args, **kwargs)

        self.original = original

        self.infos = []

        if info is not None:
            self.set_info(info)

    def set_info(self, info: "Tracer.Info"):
        """
        Updates the tracer information and recalculates line offsets.

        Args:
            info: New tracer information to use
        """

        # ex_info = ExceptionWrapper.Info(self.accumulator, info.frame.f_code.co_filename, info.frame.f_code.co_firstlineno, info.start_line, info.source, info.frame.f_code.co_name)

        self.infos.append(info)

    def _collect_frames(self, outer_tb=None):
        """
        Collect all traceback frames for display.

        Args:
            outer_tb: Optional outer traceback to include user frames from

        Returns:
            List of tuples: (filename, lineno, func_name, code_line, is_internal)
        """
        import linecache

        frames = []

        # First, collect outer traceback frames (skip nnsight internals and <nnsight> frames)
        if outer_tb is not None:
            current_tb = outer_tb
            while current_tb is not None:
                frame = current_tb.tb_frame
                fname = frame.f_code.co_filename
                lineno = current_tb.tb_lineno
                name = frame.f_code.co_name

                if (
                    CONFIG.APP.DEBUG
                    or "nnsight/" not in fname
                    and not fname.startswith("<nnsight")
                ):
                    try:
                        line = linecache.getline(fname, lineno).strip()
                    except:
                        line = ""
                    frames.append((fname, lineno, name, line, False))

                current_tb = current_tb.tb_next

        # Build mappings for reconstructing <nnsight...> frames
        accumulator = 0
        co_first_line = 0
        filename = ""
        co_name = ""

        start_lines = {}
        filename_mapping = {}
        co_names = {}
        source_lines = {}

        for info in reversed(self.infos):

            if not info.frame.f_code.co_filename.startswith("<nnsight"):
                accumulator = info.frame.f_code.co_firstlineno - 1
                filename = info.frame.f_code.co_filename
                co_name = info.frame.f_code.co_name

            accumulator += info.start_line - 1

            start_lines[info.filename] = accumulator
            filename_mapping[info.filename] = filename
            co_names[info.filename] = co_name
            source_lines[info.filename] = info.source

        # Collect inner traceback frames
        current_tb = self.original.__traceback__

        while current_tb is not None:
            frame = current_tb.tb_frame
            fname = frame.f_code.co_filename
            lineno = current_tb.tb_lineno
            name = frame.f_code.co_name

            if fname.startswith("<nnsight"):
                # Reconstruct to original user code location
                real_fname = filename_mapping[fname]
                start_line = start_lines[fname]
                func_name = (
                    co_names[fname]
                    if "__nnsight_tracing_info__" in frame.f_locals
                    else frame.f_code.co_name
                )
                source = source_lines[fname]
                line_number = lineno - 1 + start_line + 1 + co_first_line
                code_line = source[lineno - 1].strip()
                frames.append((real_fname, line_number, func_name, code_line, False))

            elif "nnsight/" in fname:
                # Internal nnsight code - only include in DEBUG mode
                if CONFIG.APP.DEBUG:
                    try:
                        line = linecache.getline(fname, lineno).strip()
                    except:
                        line = ""
                    frames.append((fname, lineno, name, line, True))

            else:
                # Regular code
                try:
                    line = linecache.getline(fname, lineno).strip()
                except:
                    line = ""
                frames.append((fname, lineno, name, line, False))

            current_tb = current_tb.tb_next

        return frames

    def format_traceback(self, outer_tb=None):
        """
        Format the traceback as a string.

        Args:
            outer_tb: Optional outer traceback to include user frames from

        Returns:
            Formatted traceback string
        """
        frames = self._collect_frames(outer_tb)

        tb_lines = ["Traceback (most recent call last):"]
        for fname, lineno, func_name, code_line, _ in frames:
            tb_lines.append(f'  File "{fname}", line {lineno}, in {func_name}')
            if code_line:
                tb_lines.append(f"    {code_line}")

        tb_lines.append("")  # Blank line before exception
        tb_lines.append(f"{type(self.original).__name__}: {self.original}")

        return "\n".join(tb_lines)

    def __str__(self):
        """Generates a formatted traceback string (without outer traceback)."""
        return self.format_traceback(outer_tb=None)

    def print_exception(self, file=None, outer_tb=None):
        """
        Print the formatted traceback.

        Args:
            file: Output file (default: sys.stderr)
            outer_tb: Optional outer traceback to include user frames from
        """
        if file is None:
            file = sys.stderr

        print(self.format_traceback(outer_tb), file=file)


def wrap_exception(exception: Exception, info: "Tracer.Info" = None):
    """
    Wraps an exception with additional context from the tracer.

    This function either updates an existing ExceptionWrapper or creates a new
    dynamically-typed exception class that inherits from both the original exception
    type and ExceptionWrapper.

    Args:
        exception: The exception to wrap
        info: Tracer information containing context about where the exception occurred

    Returns:
        A wrapped exception with enhanced traceback information

    """

    if isinstance(exception, ExceptionWrapper):
        # If already wrapped, just update the info
        exception.__suppress_context__ = True  # Kills "... during handling ..."
        exception.__traceback__ = None

        if info is not None:
            exception.set_info(info)
        return exception

    # Create a dynamic exception type that inherits from both the original exception type
    # and our ExceptionWrapper
    exception_type = type(exception)

    class NNsightException(exception_type, ExceptionWrapper):

        __qualname__ = "NNsightException"
        __module__ = "nnsight"

        def __init__(self, *args, **kwargs):

            ExceptionWrapper.__init__(self, info, exception)

        def __str__(self):
            return ExceptionWrapper.__str__(self)

    # Create a new instance of the same type, with overridden __str__
    wrapped = NNsightException(*exception.args)
    wrapped.__dict__.update(exception.__dict__)

    return wrapped


def get_non_nnsight_frame() -> FrameType:
    frame = inspect.currentframe()

    while frame:
        frame = frame.f_back
        if frame:
            # Match if filename contains 'nnsight/tests' or 'nnsight\tests'
            # OR if it does NOT contain '/nnsight/' or '\nnsight\'

            norm = frame.f_code.co_filename.replace("\\", "/")
            if "/nnsight/tests" in norm or "/nnsight/" not in norm:
                break

    return frame


_py_object = ctypes.py_object
_c_int_0 = ctypes.c_int(0)
_locals_to_fast = ctypes.pythonapi.PyFrame_LocalsToFast


def push_variables(frame: FrameType, variables: Dict):

    is_generated_frame = frame.f_code.co_filename.startswith("<nnsight")

    if is_generated_frame:

        global_variables = {
            k: v for k, v in variables.items() if k not in frame.f_locals
        }

        for key, value in global_variables.items():
            frame.f_globals[key] = value

    # Get the f_locals dict once to avoid re-sync on each property access,
    # then batch all assignments and sync back in a single call.
    locals_dict = frame.f_locals
    for key, value in variables.items():
        locals_dict[key] = value

    _locals_to_fast(_py_object(frame), _c_int_0)
