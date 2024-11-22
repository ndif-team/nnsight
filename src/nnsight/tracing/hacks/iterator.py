import ast
import inspect
import sys
from types import FrameType
from typing import TYPE_CHECKING

from ..contexts import Iterator
from ..graph import Graph
from .util import execute, execute_body, execute_until

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
        # elif isinstance(target, ast.Tuple):
        #     for t, v in zip(target.elts, item):
        #         if isinstance(t, ast.Name):
        #             frame.f_locals[t.id] = v
        execute_body(node.body, frame, context.graph)


def handle_proxy(frame: FrameType, collection: "Proxy"):

    line_no = frame.f_lineno
    source_lines, _ = inspect.getsourcelines(frame)
    source = "".join(source_lines)
    tree = ast.parse(source)

    class Visitor(ast.NodeVisitor):
        def __init__(self, line_no):
            self.target = None
            self.line_no = line_no

        def visit_For(self, node):
            if node.lineno == self.line_no:
                self.target = node
            self.generic_visit(node)

    visitor = Visitor(line_no)
    visitor.visit(tree)

    for_node = visitor.target

    graph = collection.node.graph

    iterator = Iterator(collection, parent=graph)

    item = iterator.__enter__()

    execute_until(iterator, for_node, frame)

    return iter([item])
