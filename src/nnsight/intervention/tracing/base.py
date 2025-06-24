import ast
import ctypes
import inspect
import sys
from types import FrameType
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ..backends.base import Backend
from ..backends.execution import ExecutionBackend
from .globals import Globals
from .util import suppress_all_output


class ExitTracingException(Exception):
    """Exception raised to exit the tracing process.

    This exception is used as a control flow mechanism to cleanly exit a with block without executing the code inside it.
    """

    pass


class WithBlockNotFoundError(Exception):
    """Exception raised when a with block is not found in the source code.

    This exception is used to indicate that a with block was not found at the specified line number.
    """

    pass


class Tracer:
    """
    Captures and executes code within a tracing context.

    This class allows for capturing code blocks within a 'with' statement,
    compiling them into callable functions, and executing them with access
    to the model and local variables. It provides a mechanism for intercepting
    and manipulating the execution flow of Python code.

    The tracing process works by:
    1. Capturing the code inside a 'with' block
    2. Compiling it into a callable function
    3. Executing it with the appropriate context
    """

    class Info:
        """
        Container for information about the traced code.

        This class stores metadata about the code being traced, including
        the source code itself and frame information from the call stack.

        Attributes:
            source: List of source code lines from the traced block
            frame: Frame information from the call stack where tracing occurred
            indent: Number of spaces/tabs used for indentation in the original code
        """

        def __init__(
            self, source: List[str], frame: FrameType, start_line: int, node:ast.With, filename: str = None
        ):
            """
            Initialize Info with source code and frame information.

            Args:
                source: List of source code lines from the traced block
                frame: Frame information from the call stack
                indent: Number of spaces/tabs used for indentation in the original code
                node: AST node of the with block
            """
            self.source = source
            self.frame = frame
            self.start_line = start_line
            self.node = node
            self.filename = filename if filename is not None else f"<nnsight {id(self)}>"
            
        def copy(self):
            return Tracer.Info(self.source, self.frame, self.start_line, self.node, self.filename)
        
        def __getstate__(self):
            """Get the state of the info for serialization."""
            return {
                "source": self.source,
                "start_line": self.start_line,
                "filename": self.filename,
                "frame": self.frame,
            }
        
        def __setstate__(self, state):
            """Set the state of the info for deserialization."""
            self.source = state["source"]
            self.start_line = state["start_line"]
            self.filename = state["filename"]
            self.frame = state["frame"]
            
            self.node = None

    def __init__(self, *args, backend: Backend = None, _info:Info = None, **kwargs):
        """
        Initialize a Tracer instance.

        Args:
            *args: Additional arguments to pass to the traced function
            backend: Backend implementation for executing the traced code
                    (defaults to ExecutionBackend if None)
            **kwargs: Additional keyword arguments to pass to the traced function
        """
        self.args = args
        self.kwargs = kwargs

        self.backend = ExecutionBackend() if backend is None else backend

        self.info = _info if _info is not None else None

        if self.info is None:   
            self.capture()

    def capture(self):
        """
        Capture the code block within the 'with' statement.

        This method walks up the call stack to find the frame outside of nnsight,
        extracts the source code of the 'with' block, and prepares it for later
        execution. It identifies the exact code block to be traced by analyzing
        the source code structure.
        """
        # Find the frame outside of nnsight by walking up the call stack
        frame = inspect.currentframe()
        
        while frame:
            frame = frame.f_back
            if frame and (
                frame.f_code.co_filename.find("nnsight/tests") != -1
                or frame.f_code.co_filename.find("nnsight/") == -1
            ):
                break

        # Get source code lines from the appropriate location
        start_line = frame.f_lineno

        if "__nnsight_tracing_info__" in frame.f_locals:
            # For dynamically generated code, get source from tracing info
            source_lines = frame.f_locals["__nnsight_tracing_info__"].source

        elif "_ih" in frame.f_locals:
            import IPython

            ipython = IPython.get_ipython()
            source_lines = ipython.user_global_ns["_ih"][-1].splitlines(keepends=True)

            if not source_lines[-1].endswith("\n"):
                source_lines[-1] += "\n"

        elif not frame.f_code.co_filename.startswith("<nnsight"):
            # For regular files, get source lines using inspect
            source_lines, offset = inspect.getsourcelines(frame)

            start_line = start_line if offset == 0 else start_line - offset + 1

        else:
            raise ValueError("No source code found")

        # Calculate indentation level of the source code itself
        stripped = source_lines[0].lstrip(
            "\t "
        )  # indent for removing leading tabs/spaces
        indent = len(source_lines[0]) - len(stripped)
        
        # If theres an indent, we need to remove it. This handles the case of say a trace in an indented function. E.x. a trace inside a method on a class.
        if indent > 0:
                
            source_lines = [
                line[indent:] if line.strip() else line for line in source_lines
            ]

        # Extract the code using AST parsing
        start_line, source_lines, node = self.parse(source_lines, start_line)

        # Calculate indentation level of the Tracer creation line.
        stripped = source_lines[0].lstrip(
            "\t "
        )  # removes leading tabs/spaces
        indent = len(source_lines[0]) - len(stripped) - 4
        
         # If theres an indent (more than just the indentation of the with block), we need to remove it. This handles the case of say a trace in an indented block. E.x. a trace inside a for loop or another with block.
        if indent > 0:
        
            source_lines = [
                line[indent:] if line.strip() else line for line in source_lines
            ]

        # Store the captured information for later use
        self.info = Tracer.Info(
            source_lines, frame, start_line, node
        )

    def parse(self, source_lines, start_line):
        """
        Parse the source code to extract the source code.

        Uses the Abstract Syntax Tree (AST) to identify the exact boundaries
        of the code from the specified line number.

        Args:
            source_lines: List of source code lines
            start_line: Line number where the tracer creation statement begins

        Returns:
            List of source code lines.
        """
        # Parse the entire source into an AST

        tree = ast.parse("".join(source_lines))

        class Visitor(ast.NodeVisitor):
            """AST visitor to find the 'with' node at the specified line."""

            def __init__(self, line_no):
                self.target = None
                self.line_no = line_no

            def visit_With(self, node):
                if node.lineno == self.line_no:
                    self.target = node
                else:
                    self.generic_visit(node)

        visitor = Visitor(start_line)
        visitor.visit(tree)

        if visitor.target is None:
            raise WithBlockNotFoundError(f"With block not found at line {start_line}")

        end_line = visitor.target.end_lineno
        
        start_line = visitor.target.body[0].lineno - 1

        return start_line, source_lines[start_line:end_line], visitor.target

    def compile(self) -> Callable:
        """
        Compile the captured source code as a callable function.

        Wraps the captured code in a function definition that accepts the
        necessary context parameters for execution.

        Returns:
            A callable function that executes the captured code block
        """
        # Wrap the captured code in a function definition with appropriate parameters
        self.info.source = [
            f"def __nnsight_tracer_{id(self)}__(__nnsight_tracer__, __nnsight_tracing_info__):\n",
            *self.info.source,
            "    __nnsight_tracer__.push()\n",
        ]

    def execute(self, fn: Callable):
        """
        Execute the compiled function.

        Runs the compiled function with the necessary context to execute
        the traced code block.

        Args:
            fn: The compiled function to execute
        """
        fn(self, self.info)

    def push(self, state: Dict = None):
        """
        Push local variables back to the original execution frame.

        This allows changes made during tracing to affect the original scope.

        Args:
            state: Dictionary of variable names and values to push to the frame.
                  If None, automatically collects variables from the current frame.
        """
        frame = self.info.frame

        if state is None:
            # Find the frame where the traced code is executing
            state_frame = inspect.currentframe()

            while state_frame:
                state_frame = state_frame.f_back
                if state_frame and state_frame.f_code.co_filename.startswith(
                    "<nnsight"
                ):
                    break

            state = state_frame.f_locals

            # Collect all non-nnsight variables from the frame

        state = {k: v for k, v in state.items() if not k.startswith("__nnsight")}

        if frame.f_code.co_filename.startswith("<nnsight"):
            # For dynamically generated code, update both globals and locals
            frame.f_globals.update(state)
            frame.f_locals.update(state)

            # Ensure locals are properly synchronized with the frame
            ctypes.pythonapi.PyFrame_LocalsToFast(
                ctypes.py_object(frame), ctypes.c_int(0)
            )

        else:
            # For regular files, just update locals
            for key, value in state.items():

                if Globals.stack == 1 and id(value) not in Globals.saves:
                    continue

                frame.f_locals[key] = value

                ctypes.pythonapi.PyFrame_LocalsToFast(
                    ctypes.py_object(frame), ctypes.c_int(0)
                )
                
        state.clear()
                

    def __enter__(self):
        """
        Enter the tracing context.

        Captures the code block and sets up a trace function to exit normal
        execution flow once the block is captured.

        Returns:
            The Tracer instance for use in the 'with' statement
        """

        if self.info is None:
            self.capture()

        def skip(new_frame, event, arg):
            """
            Trace function that raises ExitTracingException when the traced code is reached.

            This prevents the actual execution of the traced code in its original context,
            allowing us to execute it later with our custom handling.
            """
            new_lineno = new_frame.f_lineno - new_frame.f_code.co_firstlineno

            if (
                new_frame.f_code.co_filename == self.info.frame.f_code.co_filename
                and new_lineno >= self.info.start_line
            ):
                # To remove colab warning
                with suppress_all_output():
                    sys.settrace(None)

                self.info.frame.f_trace = None
                raise ExitTracingException()

        # Set the trace function at both global and frame level

        # To remove colab warning
        with suppress_all_output():
            sys.settrace(skip)

        self.info.frame.f_trace = skip

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the tracing context.


        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised

        Returns:
            True if an ExitTracingException was caught (to suppress it),
            None otherwise (to propagate other exceptions)
        """
        # Suppress the ExitTracingException but let other exceptions propagate
        if exc_type is ExitTracingException:

            # Execute the traced code using the configured backend
            self.backend(self)

            return True
        
    ### Serialization ###
    
    def __getstate__(self):
        """Get the state of the tracer for serialization."""
        return {
            "args": self.args,
            "kwargs": self.kwargs,
            "info": self.info,
        }
        
    def __setstate__(self, state):
        """Set the state of the tracer for deserialization."""
        self.args = state["args"]
        self.kwargs = state["kwargs"]
        self.info = state["info"]
        self.info.start_line = 0
        self.backend = ExecutionBackend()
        