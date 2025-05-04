import ast
from typing import Callable, TYPE_CHECKING, Any
import torch
from .base import Tracer
from .util import try_catch
from ..interleaver import Mediator

if TYPE_CHECKING:
    from ..interleaver import Interleaver
else:
    Interleaver = Any
    
class BackwardsTracer(Tracer):
    
    def __init__(self, tensor: torch.Tensor, fn: Callable, interleaver: Interleaver, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        self.tensor = tensor
        self.fn = fn
        self.interleaver = interleaver
        self.with_node = True
        
        
        self.capture()
        
    def compile(self):
        """
        Compile the captured code block into an intervention function.
        
        The function is wrapped with try-catch logic to handle exceptions
        and signal completion to the mediator.
        
        Returns:
            A callable intervention function
        """
        self.info.source = [
            "def ifn(__nnsight_mediator__, __nnsight_tracing_info__):\n",
            *try_catch(self.info.source, 
                       exception_source=["__nnsight_mediator__.exception(exception)\n"],
                       else_source=["__nnsight_mediator__.end()\n"],)
        ]
        
        # Because of the "try" line
        self.info.start_line -= 1
        
        
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
            """AST visitor to find either a 'with' node or a backward() call at the specified line."""
            def __init__(self, line_no):
                self.target = None
                self.line_no = line_no
                
            def visit_With(self, node):
                if node.lineno == self.line_no:
                    self.target = node
                    
            def visit_Expr(self, node):
                if self.target is None and node.lineno == self.line_no:
                    # Check if this is a backward() call
                    if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                        if node.value.func.attr == 'backward':
                            self.target = node
                                                
        visitor = Visitor(start_line)
        visitor.visit(tree)

        end_line = visitor.target.end_lineno
        
        self.with_node = isinstance(visitor.target, ast.With)
        
        if not self.with_node:
            
            return start_line, source_lines[start_line:start_line+1]
        
        start_line = visitor.target.body[0].lineno - 1        
                
        return start_line, source_lines[start_line:end_line]
        
                
    def execute(self, fn: Callable):
                
        mediator = Mediator(fn, self.info) 
                 
        self.interleaver.register(mediator, lambda: self.fn(self.tensor, *self.args, **self.kwargs))