import inspect
import ast
import sys
import re
from typing import TYPE_CHECKING, Any, Optional, Callable, List
from .compiler import Compiler
if TYPE_CHECKING:
    from ..intervention.envoy import Envoy
    
class ExitTracingException(Exception):
    pass

class Tracer:
    
    class Info:
        
        def __init__(self, node: ast.AST, source: List[str], start_line: int, end_line: int):
            self.node = node
            self.source = source
            self.start_line = start_line
            self.end_line = end_line
            self.indent = 1
            
            
            
    
    def __init__(self, model: "Envoy", fn: Callable, *args, backend: Optional[str] = None, info: Optional[Info] = None,  **kwargs):
        
        self.model = model
        self.fn = fn    
        
        self.backend = backend
        
        self.args = args
        self.kwargs = kwargs
        
        self.source = []
        
        self.frame = None
        
        if info is not None:
            self.trace(info)
        
        
    def root(self):
                
        # Get the caller's frame that is outside of nnsight code
        frame = inspect.currentframe()
        while frame:
            
            frame = frame.f_back
            if frame and frame.f_code.co_filename.find('nnsight') == -1:
                break
        # Extract function name & source lines
        
        self.frame = frame
        
        source_lines, _ = inspect.getsourcelines(self.frame)
        start_line = self.frame.f_lineno
        
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
        
        
        
        info = self.Info(visitor.target, source_lines[start_line-1:end_line], start_line, end_line)
         
        self.trace(info)        
        
        def skip(new_frame, event, arg):
            if new_frame is frame and event == 'line' and new_frame.f_lineno >= start_line:
                sys.settrace(None)
                frame.f_trace = None
                raise ExitTracingException()
     
        sys.settrace(skip)
        self.frame.f_trace = skip
        
    def trace(self, info:Info):
        
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
                    
                
                
        
        visitor = Visitor(info.start_line)
        visitor.visit(info.node)
        
        source = info.source
        
        
        source = [
            "async def inner(interleaver):\n",
            "    try:\n",
            *["    " + line for line in source[1:]],
            "    except Exception as exception:\n",
            "        interleaver.exception(exception)\n",
            "    else:\n",
            "        interleaver.continue_execution()\n"
        ]
                                
        self.source.extend(source)
        
        def convert_with_trace(code: str) -> str:
            # Pattern with optional "as" clause
            pattern = r'with\s+(model\.trace\([^)]*\))(\s+as\s+(\w+))?:'
            match = re.search(pattern, code)
            if match:
                expr = match.group(1)            # model.trace(...) part
                var = match.group(3)             # optional variable after "as"
                if var:
                    return f"{var} = {expr}.execute(inner)"
                else:
                    return f"{expr}.execute(inner)"
            return code  # Return original if no match
        

        
        self.source.append(f"{convert_with_trace(info.source[0])}\n")
                
        
        
    def __enter__(self):
        
        self.root()
            
        
    
    def __exit__(self, exc_type, exc_val, exc_tb):        
        # Suppress the ExitTracingException but let other exceptions propagate
        if exc_type is ExitTracingException:
            frame_locals = self.frame.f_locals
            frame_globals = self.frame.f_globals
            

            root = Compiler()(self.source, frame_globals)

            root(self.model,frame_locals)
            
            # Returning True tells Python to suppress the exception
            
            return True
        
    
    def execute(self, fn: Callable):
        
        fn(*self.args, **self.kwargs)
        
        return self





