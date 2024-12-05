import ast
from types import FrameType

from ..graph import Graph
from .conditional import handle as handle_conditional
from .iterator import handle as handle_iterator


def handle_inner(node:ast.stmt, frame: FrameType, graph: Graph):
    
    if isinstance(node, ast.If):
        
        handle_conditional(node, frame, graph)
        
        return True
    
    elif isinstance(node, ast.For):
        
        handle_iterator(node, frame, graph)
        
        return True
    
    return False