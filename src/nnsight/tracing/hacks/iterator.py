import ast
import ctypes
import inspect
import sys
from types import FrameType
from typing import TYPE_CHECKING

from ..contexts import Iterator
from ..graph import Graph
from .util import execute, execute_body, execute_until, visit
from .comprehension import handle_proxy as handle_comprehension
if TYPE_CHECKING:
    from ..graph import Proxy

COMPS = [ast.SetComp, ast.DictComp, ast.ListComp, ast.GeneratorExp]

def handle(node: ast.For, frame: FrameType, graph: Graph):

    iter_expr = ast.Expression(
        body=node.iter, lineno=node.lineno, col_offset=node.col_offset
    )

    iter = execute(iter_expr, frame)

    context = Iterator(iter, parent=graph)

    target = node.target

    with context as item:
        if isinstance(target, ast.Name):
            frame.f_locals[target.id] = item
        elif isinstance(target, ast.Tuple):
            for t, v in zip(target.elts, item):
                if isinstance(t, ast.Name):
                    frame.f_locals[t.id] = v
                    
        ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), 0)
        
        execute_body(node.body, frame, context.graph)


def handle_proxy(frame: FrameType, collection: "Proxy"):
    
    class Visitor(ast.NodeVisitor):
        def __init__(self, line_no):
            self.target = None
            self.line_no = line_no
            self.assign = None
            self.nodes_on_line = []
            
            
        def generic_visit(self, node):
            if hasattr(node, 'lineno') and node.lineno == self.line_no:
                self.nodes_on_line.append(node)
            super().generic_visit(node)
        def visit_Assign(self, node):
            if node.lineno == self.line_no:
                self.assign = node
            self.generic_visit(node)

        def visit_For(self, node):
            if node.lineno == self.line_no:
                self.target = node
            self.generic_visit(node)
            
        def visit_ListComp(self, node):
            if self.target is None and node.lineno == self.line_no:
                self.target = node
            self.generic_visit(node)

        def visit_DictComp(self, node):
            if self.target is None and node.lineno == self.line_no:
                self.target = node
            self.generic_visit(node)

        def visit_SetComp(self, node):
            if self.target is None and node.lineno == self.line_no:
                self.target = node
            self.generic_visit(node)

        def visit_GeneratorExp(self, node):
            if self.target is None and node.lineno == self.line_no:
                self.target = node
            self.generic_visit(node)

    visitor = visit(frame, Visitor)

    for_node:ast.If = visitor.target 

 
    if type(for_node) in COMPS:
        return handle_comprehension(for_node, frame, collection)
    
    graph = collection.node.graph
    
    iterator = Iterator(collection, parent=graph)

    item = iterator.__enter__()
    
    def callback(iterator: Iterator):
                
        iterator.__exit__(None, None, None)
    end = frame.f_lineno + (for_node.end_lineno - for_node.lineno)
    execute_until(frame.f_lineno, end, frame, callback=lambda _: callback(iterator))

    return iter([item])
