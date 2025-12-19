"""Tracing module for capturing and executing code blocks within 'with' statements.

This module provides the core Tracer class that enables dynamic code capture,
compilation, and execution. It allows intercepting Python code blocks and
executing them in controlled environments with access to the original context.
"""

import ast
import linecache
import inspect

import sys
from types import FrameType
from typing import Callable, Dict, List

from ..backends.base import Backend
from ..backends.execution import ExecutionBackend
from .globals import Globals
from .util import get_non_nnsight_frame, push_variables, suppress_all_output
from ...util import Patch, Patcher
from ... import CONFIG


class ExitTracingException(Exception):
    """Exception raised to exit the tracing process.

    This exception is used as a control flow mechanism to cleanly exit a with block
    without executing the code inside it. When the tracer detects that execution
    has reached the traced code block, it raises this exception to prevent normal
    execution and instead execute the code through the tracing backend.
    """

    pass


class WithBlockNotFoundError(Exception):
    """Exception raised when a with block is not found in the source code.

    This exception indicates that the AST parser could not locate a with block
    at the expected line number during the code capture process. This typically
    occurs when there are issues with source code parsing or line number mapping.
    """

    pass


class Tracer:
    """
    Captures and executes code within a tracing context.

    This class provides a sophisticated mechanism for intercepting Python code blocks
    within 'with' statements, capturing their source code, and executing them in
    controlled environments. It's designed to enable dynamic code manipulation and
    execution with full access to the original context and variables.

    The tracing process follows four main steps:
    1. **Capture**: Finds and extracts the source code from the with block
    2. **Parse**: Uses AST to identify the exact code boundaries within the block
    3. **Compile**: Wraps the source code in a function definition for execution
    4. **Execute**: Runs the compiled function with appropriate context

    Example:
        ```python
        with Tracer() as tracer:
            # This code block will be captured and executed later
            x = some_computation()
            result = process(x)
        ```

    The tracer supports various execution environments including:
    - Regular Python files
    - IPython/Jupyter notebooks
    - Interactive consoles
    - Nested tracing contexts

    Attributes:
        args: Arguments passed to the tracer for use during execution
        kwargs: Keyword arguments passed to the tracer
        backend: Backend implementation that handles the actual code execution
        info: Info object containing captured code metadata and context
    """

    class Info:
        """
        Container for metadata about the traced code block.

        This nested class stores all necessary information about the code being traced,
        including the source code, execution frame context, and AST node information.
        It serves as a data transfer object between the capture, parse, and execution phases.

        Attributes:
            source (List[str]): Source code lines from the traced block
            frame (FrameType): Frame information from the call stack where tracing occurred
            start_line (int): Line number where the traced code block begins
            node (ast.With): AST node representing the with block
            filename (str): Filename for the traced code (real file or generated identifier)
        """

        def __init__(
            self,
            source: List[str],
            frame: FrameType,
            start_line: int,
            node: ast.With,
            filename: str = None,
        ):
            """
            Initialize Info with captured code metadata.

            Args:
                source: List of source code lines from the traced block
                frame: Frame information from the call stack where tracing occurred
                start_line: Line number where the traced code block begins
                node: AST node representing the with block
                filename: Optional filename; generates unique identifier if None
            """
            self.source = source
            self.frame = frame
            self.start_line = start_line
            self.node = node
            self.filename = (
                filename if filename is not None else f"<nnsight {id(self)}>"
            )

        def copy(self):
            """Create a deep copy of this Info instance.

            Returns:
                A new Info instance with the same metadata
            """
            return Tracer.Info(
                self.source, self.frame, self.start_line, self.node, self.filename
            )

        def __getstate__(self):
            """Get the state of the info for serialization.

            Returns:
                Dict containing serializable state information
            """
            return {
                "source": self.source,
                "start_line": self.start_line,
                "filename": self.filename,
                "frame": self.frame,
            }

        def __setstate__(self, state):
            """Restore the state of the info from serialization.

            Args:
                state: Dictionary containing the serialized state

            Note:
                AST node is set to None as it cannot be serialized
            """
            self.source = state["source"]
            self.start_line = state["start_line"]
            self.filename = state["filename"]
            self.frame = state["frame"]
            # AST nodes cannot be serialized, so we reset to None
            self.node = None

    # === Initialization ===

    def __init__(self, *args, backend: Backend = None, _info: Info = None, **kwargs):
        """
        Initialize a Tracer instance.

        Args:
            *args: Additional arguments to pass to the traced function
            backend: Backend implementation for executing the traced code.
                    Defaults to ExecutionBackend if None.
            _info: Pre-existing Info object for deserialization. If None,
                  capture() will be called to extract code from the with block.
            **kwargs: Additional keyword arguments to pass to the traced function
        """
        # Store arguments for later use during execution
        self.args = args
        self.kwargs = kwargs

        # Set up the execution backend (defaults to direct execution)
        self.backend = ExecutionBackend() if backend is None else backend

        # Initialize or use provided tracing info
        self.info = _info if _info is not None else None

        self.asynchronous = False

        # If no pre-existing info, attempt to capture the code block
        if self.info is None:
            self.capture()

    # === Core Tracing Methods ===

    def capture(self):
        """
        Capture the code block within the 'with' statement.

        This is step 1 of the tracing process. It walks up the call stack to find
        the frame outside of nnsight, extracts the source code of the 'with' block,
        and prepares it for later execution. The method handles various execution
        environments including regular files, IPython notebooks, and interactive consoles.

        The capture process:
        1. Identifies the execution context (file, notebook, console, nested trace)
        2. Extracts source code lines from the appropriate source
        3. Handles indentation normalization for proper AST parsing
        4. Calls parse() to identify the exact with block boundaries
        5. Stores all metadata in a Tracer.Info object

        Raises:
            ValueError: If no source code can be found for the current context
        """
        # Find the frame outside of nnsight by walking up the call stack
        frame = get_non_nnsight_frame()

        # Get the line number where the tracer was created
        start_line = frame.f_lineno

        cache_key = (
            frame.f_code.co_filename,
            start_line,
            frame.f_code.co_name,
            frame.f_code.co_firstlineno,
        )

        cached = Globals.cache.get(cache_key)

        if CONFIG.APP.TRACE_CACHING and cached is not None:

            source_lines, start_line, node, filename = cached

            self.info = Tracer.Info(
                source_lines,
                frame,
                start_line,
                node,
                filename,
            )

            return

        # Determine the execution context and extract source code accordingly

        # CASE 1: Already inside another nnsight trace (nested tracing)
        if "__nnsight_tracing_info__" in frame.f_locals:
            # For dynamically generated code, get source from parent tracing info
            source_lines = frame.f_locals["__nnsight_tracing_info__"].source

        # CASE 2: IPython/Jupyter notebook environment
        elif "_ih" in frame.f_locals:
            import IPython

            # Get the current cell's source code from IPython's input history
            ipython = IPython.get_ipython()
            source_lines = ipython.user_global_ns["_ih"][-1].splitlines(keepends=True)

            # Ensure the last line ends with a newline for proper parsing
            if not source_lines[-1].endswith("\n"):
                source_lines[-1] += "\n"

        # CASE 3: Regular Python file
        elif not frame.f_code.co_filename.startswith("<nnsight"):

            def noop(*args, **kwargs):
                """No-op function to prevent linecache from clearing during tracing."""
                pass

            # Prevent linecache from clearing cache during tracing to handle file edits
            with Patcher([Patch(linecache, noop, "checkcache")]):
                # Extract source lines using inspect module
                source_lines, offset = inspect.getsourcelines(frame)

            # Adjust start line based on any offset from inspect
            start_line = start_line if offset == 0 else start_line - offset + 1

        # CASE 4: Interactive Python console (nnsight-specific)
        elif frame.f_code.co_filename == "<nnsight-console>":
            from ... import __INTERACTIVE_CONSOLE__

            # Get source from the interactive console buffer
            source_lines = __INTERACTIVE_CONSOLE__.buffer
            # Ensure all lines end with newlines for consistent parsing
            source_lines = [
                line if line.endswith("\n") else line + "\n" for line in source_lines
            ]

        else:
            raise ValueError("No source code found for current execution context")

        # STEP 1: Normalize indentation for AST parsing
        # Find the first non-blank line to determine base indentation level
        first_non_blank = None
        for line in source_lines:
            if line.strip():
                first_non_blank = line
                break

        if first_non_blank is not None:
            # Calculate base indentation level (handles code inside functions/classes)
            stripped = first_non_blank.lstrip("\t ")
            indent = len(first_non_blank) - len(stripped)

            # Remove base indentation if present (e.g., tracer inside a method)
            if indent > 0:
                source_lines = [
                    line[indent:] if line.strip() else line for line in source_lines
                ]

        # STEP 2: Parse the source code to find the with block boundaries
        start_line, source_lines, node = self.parse(source_lines, start_line)

        # STEP 3: Remove additional indentation from the with block itself

        # Calculate indentation level of the with statement line
        stripped = source_lines[0].lstrip("\t ")
        indent = len(source_lines[0]) - len(stripped)

        # Remove with block indentation (handles nested with blocks, loops, etc.)
        if indent > 0:
            source_lines = [
                line[indent:] if line.strip() else line for line in source_lines
            ]

        # STEP 4: Store all captured information for later compilation and execution
        self.info = Tracer.Info(source_lines, frame, start_line, node)

        if CONFIG.APP.TRACE_CACHING:
            Globals.cache.add(
                cache_key,
                (source_lines, start_line, node, self.info.filename),
            )

    def parse(self, source_lines: List[str], start_line: int):
        """
        Parse the source code to extract the with block contents.

        This is step 2 of the tracing process. Uses the Abstract Syntax Tree (AST)
        to identify the exact boundaries of the with block and extract only the
        code that should be traced and executed later.

        Args:
            source_lines: List of source code lines to parse
            start_line: Line number where the tracer creation statement begins

        Returns:
            Tuple containing:
                - start_line (int): Adjusted start line of the with block body
                - source_lines (List[str]): Extracted source lines from the with block
                - node (ast.With): AST node representing the with block

        Raises:
            WithBlockNotFoundError: If no with block is found at the specified line
        """
        # Parse the entire source code into an Abstract Syntax Tree
        tree = ast.parse("".join(source_lines))

        class WithBlockVisitor(ast.NodeVisitor):
            """AST visitor to find the 'with' or 'async with' node at the specified line."""

            def __init__(self, target_line_no):
                self.target = None
                self.line_no = target_line_no

            def visit_With(self, node):
                """Visit with statement nodes and check if they match our target line."""
                # Check each context expression in the with statement
                for item_node in node.items:
                    if item_node.context_expr.lineno == self.line_no:
                        self.target = node
                        return

                # Continue visiting child nodes if this isn't the target
                self.generic_visit(node)

            def visit_AsyncWith(self, node):
                """Visit async with statement nodes and check if they match our target line."""
                # Check each context expression in the async with statement
                for item_node in node.items:
                    if item_node.context_expr.lineno == self.line_no:
                        self.target = node
                        return

                # Continue visiting child nodes if this isn't the target
                self.generic_visit(node)

        # Use the visitor to find the target with block
        visitor = WithBlockVisitor(start_line)
        visitor.visit(tree)

        # Handle case where with block is not found
        if visitor.target is None:
            # Provide helpful context for debugging
            context_start = max(0, start_line - 5)
            context_end = min(len(source_lines), start_line + 6)
            context_lines = source_lines[context_start:context_end]

            # Mark the problematic line
            target_index = start_line - context_start - 1
            if 0 <= target_index < len(context_lines):
                context_lines[target_index] = (
                    context_lines[target_index].rstrip("\n") + " <--- HERE\n"
                )

            context_str = "".join(context_lines)
            message = f"With block not found at line {start_line}\n"
            message += f"We looked here:\n\n{context_str}"
            raise WithBlockNotFoundError(message)

        # Extract the boundaries of the with block
        end_line = visitor.target.end_lineno
        # Start from the first line of the with block body (not the with statement itself)
        body_start_line = visitor.target.body[0].lineno - 1

        return body_start_line, source_lines[body_start_line:end_line], visitor.target

    def compile(self) -> Callable:
        """
        Compile the captured source code into a callable function.

        This is step 3 of the tracing process. Takes the captured and parsed source
        code and wraps it in a function definition with the necessary context parameters.
        The resulting function can be executed with proper variable scoping and
        access to the original execution environment.

        The compiled function signature is:
        `__nnsight_tracer_{id}__(__nnsight_tracer__, __nnsight_tracing_info__)`

        The function includes:
        - A call to tracer.pull() to import variables from the original scope
        - The original traced code block
        - A call to tracer.push() to export variables back to the original scope

        Returns:
            A callable function that executes the captured code block with proper context
        """
        # Wrap the captured code in a function definition with context parameters
        function_name = f"__nnsight_tracer_{id(self)}__"

        # Build the complete function with:
        # 1. Function definition with tracer and info parameters
        # 2. Variable import from original scope (pull)
        # 3. The original traced code block
        # 4. Variable export back to original scope (push)

        self.info.source = [
            f"def {function_name}(__nnsight_tracer__, __nnsight_tracing_info__):\n",
            "    __nnsight_tracer__.pull()\n",
            *self.info.source,
            "    __nnsight_tracer__.push()\n",
        ]

        # Adjust the start line to account for the added function definition
        self.info.start_line -= 1

    def execute(self, fn: Callable):
        """
        Execute the compiled function with proper context.

        This is step 4 of the tracing process. Runs the compiled function that was
        created in the compile() step, passing in the tracer instance and info object
        as context. This allows the traced code to access the original variables and
        execution environment.

        Args:
            fn: The compiled function to execute (created by compile() method)
        """
        # Execute the compiled function with tracer and info as context
        return fn(self, self.info)

    # === Variable Management ===

    def push(self, state: Dict = None):
        """
        Push local variables back to the original execution frame.

        This method exports variables from the traced code execution back to the
        original scope where the tracer was created. This allows changes made
        during tracing to persist and affect the original execution environment.

        The method handles variable filtering to only push non-nnsight variables,
        and includes special logic for nested tracing contexts using Globals.stack.

        Args:
            state: Dictionary of variable names and values to push to the frame.
                  If None, automatically collects variables from the current execution frame.
        """
        # Get the original frame where the tracer was created
        target_frame = self.info.frame

        if state is None:
            # Find the current execution frame by walking up the call stack
            current_frame = inspect.currentframe()

            # Walk up until we find the nnsight execution frame
            while current_frame:
                current_frame = current_frame.f_back
                if current_frame and current_frame.f_code.co_filename.startswith(
                    "<nnsight"
                ):
                    break

            # Extract local variables from the execution frame
            state = current_frame.f_locals

        # Filter out internal nnsight variables (they shouldn't be pushed back)
        filtered_state = {
            k: v for k, v in state.items() if not k.startswith("__nnsight")
        }

        # Special handling for nested tracing contexts
        # When stack == 1, only push variables that were explicitly saved
        if Globals.stack == 1:
            filtered_state = {
                k: v for k, v in filtered_state.items() if id(v) in Globals.saves
            }

        # Push the filtered variables back to the original frame
        push_variables(target_frame, filtered_state)

    def pull(self):
        """
        Pull variables from the original execution frame into the current context.

        This method imports variables from the original scope where the tracer was
        created into the current traced code execution context. This ensures that
        the traced code has access to all variables that were available when the
        tracer was instantiated.

        This is the opposite operation of push() and is called at the beginning
        of traced code execution.
        """
        # Find the current execution frame by walking up the call stack
        current_frame = inspect.currentframe()

        # Walk up until we find the nnsight execution frame
        while current_frame:
            current_frame = current_frame.f_back
            if current_frame and current_frame.f_code.co_filename.startswith(
                "<nnsight"
            ):
                break

        # Get variables from the original frame where the tracer was created
        original_state = self.info.frame.f_locals

        # Filter out internal nnsight variables
        filtered_state = {
            k: v for k, v in original_state.items() if not k.startswith("__nnsight")
        }

        # Push the original variables into the current execution context
        push_variables(current_frame, filtered_state)

    # === Context Manager Methods ===

    def __enter__(self):
        """
        Enter the tracing context.

        This method is called when entering the 'with' statement. It sets up the
        tracing mechanism that will capture the code block and prevent its normal
        execution. Instead, the code will be executed later through the backend.

        The method:
        1. Captures the code block if not already done
        2. Checks for empty code blocks (just 'pass' statements)
        3. Sets up a trace function that raises ExitTracingException when the
           traced code is reached, preventing normal execution

        Returns:
            The Tracer instance for use in the 'with' statement
        """

        # Capture the code block if it hasn't been done yet
        if self.info is None:
            self.capture()

        # Handle empty code blocks (containing only 'pass' statements)
        # These don't need special tracing, just return normally
        if isinstance(self.info.node.body[0], ast.Pass):
            return self

        def skip_traced_code(frame, event, arg):
            """
            Trace function that intercepts execution and prevents normal code execution.

            This function is called by Python's tracing mechanism. When execution
            reaches the captured code block, it raises ExitTracingException to
            prevent normal execution and trigger our custom backend execution instead.

            Args:
                frame: The current execution frame
                event: The trace event type (we respond to all events)
                arg: Additional event argument (unused)
            """
            # Calculate the relative line number within the code object
            relative_line_no = frame.f_lineno - frame.f_code.co_firstlineno

            # Check if we've reached the traced code block
            if (
                frame.f_code.co_filename == self.info.frame.f_code.co_filename
                and relative_line_no >= self.info.start_line
            ):
                # Disable tracing to prevent further trace calls
                # (suppress_all_output prevents Colab warnings during trace manipulation)
                with suppress_all_output():
                    sys.settrace(None)

                # Clear the frame-level trace function
                self.info.frame.f_trace = None

                # Raise exception to exit normal execution flow
                raise ExitTracingException()

        # Set up the trace function at both global and frame levels
        # This ensures our trace function intercepts execution when the traced code is reached

        # Set global trace (suppress_all_output prevents Colab warnings)
        with suppress_all_output():
            sys.settrace(skip_traced_code)

        # Set frame-level trace as backup
        self.info.frame.f_trace = skip_traced_code

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the tracing context and execute the captured code.

        This method is called when exiting the 'with' statement. It handles the
        execution of the captured code through the configured backend, and manages
        exception handling for the tracing mechanism.

        Args:
            exc_type: Exception type if an exception was raised in the with block
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised

        Returns:
            True if an ExitTracingException was caught (suppresses the exception),
            None otherwise (allows other exceptions to propagate)
        """
        # Handle the ExitTracingException (our control flow mechanism)
        if exc_type is ExitTracingException or exc_type is None:
            with suppress_all_output():
                sys.settrace(None)

            # Clear the frame-level trace function
            self.info.frame.f_trace = None

            # This is the expected case - the traced code was intercepted
            # Execute the captured code using the configured backend
            if self.asynchronous:
                return self.backend(self)

            self.backend(self)

            return True

    async def __aenter__(self):

        self.asynchronous = True

        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):

        await self.__exit__(exc_type, exc_val, exc_tb)

        return True

    # === Serialization Methods ===

    def __getstate__(self):
        """Get the state of the tracer for serialization.

        Returns:
            Dict containing the serializable state of the tracer
        """
        return {
            "args": self.args,
            "kwargs": self.kwargs,
            "info": self.info,
            "asynchronous": self.asynchronous,
        }

    def __setstate__(self, state):
        """Restore the state of the tracer from serialization.

        Args:
            state: Dictionary containing the serialized tracer state

        Note:
            The backend is reset to ExecutionBackend and start_line is reset to 0
            since these cannot be reliably serialized across different contexts.
        """
        self.args = state["args"]
        self.kwargs = state["kwargs"]
        self.info = state["info"]
        self.asynchronous = state["asynchronous"]

        # Reset values that cannot be reliably serialized
        self.info.start_line = 0  # Line numbers may not be valid in new context
        self.backend = ExecutionBackend()  # Backend needs to be recreated
