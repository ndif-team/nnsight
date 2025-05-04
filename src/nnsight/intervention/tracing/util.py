import inspect
from typing import List, Callable, TYPE_CHECKING

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


def get_frame(frame: inspect.FrameInfo, until:str="nnsight"):
    """
    Traverses up the call stack until finding a frame outside the specified module.
    
    Args:
        frame: The starting frame to traverse from
        until: The module name to traverse until (default: "nnsight")
        
    Returns:
        The first frame found outside the specified module, or None
    """
    while frame:
        frame = frame.f_back
        if frame and frame.f_code.co_filename.find(until) == -1:
            break
    return frame


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
        
        self.offset = 0
        self.info = None
        
        self.set_info(info)
        
        
    def set_info(self, info:"Tracer.Info"):
        """
        Updates the tracer information and recalculates line offsets.
        
        Args:
            info: New tracer information to use
        """
        print(info.start_line, self.offset)
        self.info = info
        self.offset += info.start_line - 1
            
    def __str__(self):
        """
        Generates a formatted traceback string with proper context.
        
        Returns:
            A string containing the formatted traceback with source code context
        """
        
        source_lines, _ = inspect.getsourcelines(self.info.frame)
        
        traceback = self.original.__traceback__
        
        #TODO handle multiple levels of traceback
        #    only build the traceback if the code is from <nnsight> otherwise we can get it from the frame
        
        # Find the deepest frame in the traceback
        while traceback.tb_next is not None:
            traceback = traceback.tb_next
            
        offset = traceback.tb_lineno - 1 + self.offset
        
        traceback = [
            "\n\nTraceback (most recent call last):",
            f'  File "{self.info.frame.f_code.co_filename}", line {offset+1}, in {self.info.frame.f_code.co_name}',
            f'    {source_lines[offset].strip()}\n',
            f'{type(self.original).__name__}: {self.original}',
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
