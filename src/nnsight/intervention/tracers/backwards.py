import torch
from ...tracing.tracer import Tracer
import ast

class BackwardsTracer(Tracer):
    
    def __init__(self, tensor: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.tensor = tensor
        
        
    def parse(self, tree, start_line):
        
        class Visitor(ast.NodeVisitor):
            """AST visitor to find the 'with' node at the specified line."""
            def __init__(self, line_no):
                self.target = None
                self.line_no = line_no  
                self.end_line = None
                
            def visit_Call(self, node):
                if hasattr(node.func, 'attr') and node.func.attr == 'backward' and node.lineno == self.line_no:
                    self.target = node
                    self.end_line = node.end_lineno
                self.generic_visit(node)
                    
            def visit_Attribute(self, node):
                if hasattr(node, 'attr') and node.attr == 'grad':
                    if self.target is None or node.lineno > self.target.lineno:
                        self.end_line = node.end_lineno
                self.generic_visit(node)
                        
        visitor = Visitor(start_line)
        visitor.visit(tree)
        
        breakpoint()
        
        return visitor.end_line
    
    
    # def execute(self):
        
                