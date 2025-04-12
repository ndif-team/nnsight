import inspect
import ast
import sys
import re
from .Interleaver import Interleaver
from typing import TYPE_CHECKING, Any, Optional, Callable
from .Compiler import Compiler
if TYPE_CHECKING:
    from .Envoy import Envoy

    
class ExitTracingException(Exception):
    pass

class Tracer:
    
    class Info:
        
        def __init__(self, node: ast.AST, source: ast.List[str], start_line: int, end_line: int):
            self.node = node
            self.source = source
            self.start_line = start_line
            self.end_line = end_line
            
            
            
    
    def __init__(self, model: "Envoy", fn: Callable, *args, backend: Optional[str] = None, info: Optional[Info] = None,  **kwargs):
        self.model = model
        self.fn = fn    
        
        self.backend = backend
        
        self.args = args
        self.kwargs = kwargs
        
        self.source = []
        
        if info is None:
            self.trace(info)
        
        
    def root(self):
                
        # Get the caller's frame
        frame = inspect.currentframe().f_back

        # Extract function name & source lines
        source_lines, _ = inspect.getsourcelines(frame)
        start_line = frame.f_lineno
        
        #TODO only past start line
        
        # Find the 'with' statement and extract its block
        tree = ast.parse("".join(source_lines))
        
        class Visitor(ast.NodeVisitor):
            def __init__(self, line_no):
                self.target = None
               
                self.line_no = line_no
                
                
            def visit_With(self, node):
                if node.lineno == self.line_no:
                    self.target = node

        visitor = Visitor(start_line)
        visitor.visit(tree)
        
        end_line = visitor.target.end_lineno
        
        self.info = self.Info(visitor.target, source_lines[start_line:end_line], start_line, end_line)
        
        
        
        self.trace()
        
                
        breakpoint()
        
        
        
        self.source.extend(source_lines[start_line:end_line])   
        
        def skip(new_frame, event, arg):
            if new_frame is frame and event == 'line' and new_frame.f_lineno >= start_line:
                sys.settrace(None)
                frame.f_trace = None
                raise ExitTracingException()
     
        sys.settrace(skip)
        frame.f_trace = skip
        
    def trace(self):
        
        class Visitor(ast.NodeVisitor):
            def __init__(self, line_no):
                self.target = None
                self.object_name = None
                self.context_name = None
                self.line_no = line_no
                
                self.inner_traces = []
                
            def visit_With(self, node):
                if node.lineno == self.line_no:
                    self.target = node
                    
                    self.object_name = node.items[0].context_expr.func.value.id
                    self.context_name = node.items[0].optional_vars.id if node.items[0].optional_vars else None
                    
                    self.generic_visit(node)
                    
                elif node.lineno <= self.target.end_lineno and node.items[0].context_expr.func.value.id == self.object_name:
                    self.inner_traces.append(node)
                else:
                    self.generic_visit(node)    
                    
                
                
        
        visitor = Visitor(start_line)
        visitor.visit(tree)
        
        self.source.append("async def root(user_locals):\n")
        
        
    def __enter__(self):
        
        self.root()
            
        
    
    def __exit__(self, exc_type, exc_val, exc_tb):        
        # Suppress the ExitTracingException but let other exceptions propagate
        if exc_type is ExitTracingException:
            frame_locals = inspect.currentframe().f_back.f_locals  
            frame_globals = inspect.currentframe().f_back.f_globals
            
            interventions = Compiler()(self.source, frame_globals)
            
            # Returning True tells Python to suppress the exception
            with Interleaver(interventions) as interleaver:
                self.model._interleaver = interleaver
                interleaver(self.fn, frame_locals, *self.args, **self.kwargs)    
            return True





