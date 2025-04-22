import inspect
import ast
import sys
from typing import TYPE_CHECKING, Any, Optional, Callable, List

class ExitTracingException(Exception):
    """Exception raised to exit the tracing process."""
    pass

class Tracer:
    """
    Captures and executes code within a tracing context.
    
    This class allows for capturing code blocks within a 'with' statement,
    compiling them into callable functions, and executing them with access
    to the model and local variables.
    """
    
    class Info:
        """
        Container for information about the traced code.
        
        Attributes:
            source: List of source code lines
            frame: Frame information from the call stack
        """
        
        def __init__(self, source: List[str], frame: inspect.FrameInfo):
            """
            Initialize Info with source code and frame information.
            
            Args:
                source: List of source code lines
                frame: Frame information from the call stack
            """
            self.source = source
            self.frame = frame
            
    def __init__(self, *args, **kwargs):
        """
        Initialize a Tracer instance.
        
        Args:
            *args: Additional arguments to pass to the traced function
            **kwargs: Additional keyword arguments to pass to the traced function
        """
        self.args = args
        self.kwargs = kwargs
        
        self.info = None   
         
        

    def capture(self):
        """
        Capture the code block within the 'with' statement.
        
        This method walks up the call stack to find the frame outside of nnsight,
        extracts the source code of the 'with' block, and raises an exception
        to exit the normal execution flow.
        """
        # Find the frame outside of nnsight
        frame = inspect.currentframe()
        while frame:
            frame = frame.f_back
            if frame and frame.f_code.co_filename.find('nnsight') == -1:
                break
            
        # Get source code lines
        
        if frame.f_code.co_filename != '<string>':
            source_lines, _ = inspect.getsourcelines(frame)
        elif 'tracing_info' in frame.f_locals:
            source_lines = frame.f_locals['tracing_info'].source
        else:
            raise ValueError('No source code found')
            
        start_line = frame.f_lineno
        
        # Parse the source code to find the 'with' block
        tree = ast.parse("".join(source_lines))
        
        end_line = self.parse(tree, start_line)
        
        # Store the captured information
        self.info = Tracer.Info(source_lines[start_line:end_line], frame)
         
        # Set up trace function to exit normal execution
        def skip(new_frame, event, arg):
            if new_frame is frame and event == 'line' and new_frame.f_lineno >= start_line-1:
                sys.settrace(None)
                frame.f_trace = None
                raise ExitTracingException()
     
        sys.settrace(skip)
        frame.f_trace = skip
        
        
    def parse(self, tree, start_line):
    
        class Visitor(ast.NodeVisitor):
            """AST visitor to find the 'with' node at the specified line."""
            def __init__(self, line_no):
                self.target = None
                self.line_no = line_no
                
            def visit_With(self, node):
                if node.lineno == self.line_no:
                    self.target = node
                    
        visitor = Visitor(start_line)
        visitor.visit(tree)
                
        end_line = visitor.target.end_lineno
        
        return end_line

    def compile(self) -> Callable:
        """
        Compile the captured code block into a callable function.
        
        Returns:
            A callable function that executes the captured code block
        """
        # Wrap the captured code in a function definition
        self.info.source = [
            "def fn(model, tracer, user_locals, tracing_info):\n",
            *self.info.source
        ]
                
        source = "".join(self.info.source)
        
        local_namespace = {}

        # Execute the function definition in the local namespace
        exec(source, self.info.frame.f_globals, local_namespace)
        
        return local_namespace["fn"]
    
    def execute(self, fn: Callable):
        """
        Execute the compiled function.
        
        Args:
            fn: The compiled function to execute
        """
        fn(self, self.info.frame.f_locals, self.info)
    
    def __enter__(self):
        """
        Enter the tracing context.
        
        Returns:
            The Tracer instance
        """
        
        self.capture()
        
        return self
                
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the tracing context.
        
        If an ExitTracingException is caught, compile and execute the captured code.
        
        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
            
        Returns:
            True if an ExitTracingException was caught, None otherwise
        """
        # Suppress the ExitTracingException but let other exceptions propagate
        if exc_type is ExitTracingException:
            fn = self.compile()
            self.execute(fn)
            return True
