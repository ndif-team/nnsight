import ast
import ctypes
import inspect
import sys
from types import FrameType
from typing import TYPE_CHECKING

from ..contexts import Iterator
from ..graph import Graph
from .util import execute, execute_body, execute_until, visit

if TYPE_CHECKING:
    from ..graph import Proxy


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
            ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), 0)
            
  
        # elif isinstance(target, ast.Tuple):
        #     for t, v in zip(target.elts, item):
        #         if isinstance(t, ast.Name):
        #             frame.f_locals[t.id] = v
        execute_body(node.body, frame, context.graph)


def handle_proxy(frame: FrameType, collection: "Proxy"):

    class Visitor(ast.NodeVisitor):
        def __init__(self, line_no):
            self.target = None
            self.line_no = line_no

        def visit_If(self, node):
            if node.lineno == self.line_no:
                self.target = node
            self.generic_visit(node)

    for_node:ast.For = visit(frame, Visitor)

    graph = collection.node.graph

    iterator = Iterator(collection, parent=graph)

    item = iterator.__enter__()

    execute_until(iterator, frame.f_lineno, frame.f_lineno + len(for_node.body), frame)

    return iter([item])
