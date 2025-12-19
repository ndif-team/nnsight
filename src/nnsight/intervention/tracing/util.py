
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


@contextlib.contextmanager
def suppress_all_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def get_dependencies(fn:Callable):
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
    def __init__(self, info:"Tracer.Info", original:Exception, *args, **kwargs):
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
        
        self.set_info(info)
        
        
    def set_info(self, info:"Tracer.Info"):
        """
        Updates the tracer information and recalculates line offsets.
        
        Args:
            info: New tracer information to use
        """

        
        # ex_info = ExceptionWrapper.Info(self.accumulator, info.frame.f_code.co_filename, info.frame.f_code.co_firstlineno, info.start_line, info.source, info.frame.f_code.co_name)
        
        self.infos.append(info)
        
    def __str__(self):
        """
        Generates a formatted traceback string with proper context.
        
        Returns:
            A string containing the formatted traceback with source code context
        """
    
        accumulator = 0
        co_first_line = 0
        filename = ""
        co_name = ""
            
        start_lines = {}
        filename_mapping = {}
        co_names = {}
        source_lines = {}
        
        for info in reversed(self.infos):
            
            if isinstance(info.frame, FrameType) and not info.frame.f_code.co_filename.startswith("<nnsight"):
            
                accumulator = info.frame.f_code.co_firstlineno - 1
                filename = info.frame.f_code.co_filename
                co_name = info.frame.f_code.co_name
                
            accumulator += info.start_line - 1
                
            start_lines[info.filename] = accumulator
            filename_mapping[info.filename] = filename
            co_names[info.filename] = co_name
            source_lines[info.filename] = info.source
            
        traceback = self.original.__traceback__
        
        tb_frames = []
        current_tb = traceback
        
        import linecache
        
        while current_tb is not None:
            frame = current_tb.tb_frame
            filename = frame.f_code.co_filename
            lineno = current_tb.tb_lineno
            name = frame.f_code.co_name
            
            # Case 1: <nnsight> - our traced code
            if filename.startswith("<nnsight"):                
                
                fname = filename_mapping[filename]
                start_line = start_lines[filename]
                co_name = co_names[filename] if '__nnsight_tracing_info__' in frame.f_locals else frame.f_code.co_name
                source = source_lines[filename]
                                
                line_number = lineno - 1 + start_line

                tb_frames.append(f'  File "{fname}", line {line_number+1 + co_first_line}, in {co_name}')
                tb_frames.append(f'    {source[lineno-1].strip()}')
    
            # Case 2: Skip internal nnsight code
            elif "nnsight/" in filename:
                if CONFIG.APP.DEBUG:
                    tb_frames.append(f'  File "{filename}", line {lineno}, in {name}')
                    try:
                        line = linecache.getline(filename, lineno).strip()
                        if line:
                            tb_frames.append(f'    {line}')
                    except:
                        pass
            # Case 3: Regular code - use normal traceback
            else:
                tb_frames.append(f'  File "{filename}", line {lineno}, in {name}')
                try:
                    line = linecache.getline(filename, lineno).strip()
                    if line:
                        tb_frames.append(f'    {line}')
                except:
                    pass
            
            current_tb = current_tb.tb_next
        
        traceback = [
            "\n\nTraceback (most recent call last):"
        ] + tb_frames + [
            f'\n{type(self.original).__name__}: {self.original}',
        ]
        
        return "\n".join(traceback)
        

def wrap_exception(exception:Exception, info:"Tracer.Info"):
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
        
        exception.set_info(info)
        return exception
    
    # Create a dynamic exception type that inherits from both the original exception type
    # and our ExceptionWrapper
    exception_type = type(exception)
    class NNsightException(exception_type, ExceptionWrapper):
        
        __qualname__ = "NNsightException"
        __module__ = "nnsight"
        
        def __init__(self, *args, **kwargs):
            
            exception_type.__init__(self, *args, **kwargs)
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
            
            
def push_variables(frame:FrameType, variables:Dict):
    
    is_generated_frame = frame.f_code.co_filename.startswith("<nnsight")
    
    if is_generated_frame:
        
        global_variables = {k: v for k, v in variables.items() if k not in frame.f_locals}
    
        for key, value in global_variables.items():
            frame.f_globals[key] = value
            
            ctypes.pythonapi.PyFrame_LocalsToFast(
                ctypes.py_object(frame), ctypes.c_int(0)
            )
    
    for key, value in variables.items():
        frame.f_locals[key] = value
        
        ctypes.pythonapi.PyFrame_LocalsToFast(
            ctypes.py_object(frame), ctypes.c_int(0)
        )
                
