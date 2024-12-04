import ast
import inspect
import sys
from types import FrameType
from typing import TYPE_CHECKING

from ..contexts import Condition
from .util import execute, execute_body, execute_until, visit
from ..graph import Graph
if TYPE_CHECKING:
    from ..graph import Proxy

def get_else(node: ast.If):

    return (
        node.orelse[0]
        if isinstance(node.orelse[0], ast.If)
        else ast.If(
            test=ast.Constant(value=None),
            body=node.orelse,
            orelse=[],
            lineno=node.lineno,
            col_offset=node.col_offset,
        )
    )
    
def handle(node: ast.If, frame:FrameType, graph:Graph, branch:Condition = None):

    condition_expr = ast.Expression(
        body=node.test, lineno=node.lineno, col_offset=node.col_offset
    )

    condition = execute(condition_expr, frame)
        
    context = Condition(condition, parent = graph) if branch is None else branch.else_(condition)

    with context as branch:
        execute_body(node.body, frame, branch.graph)

    if node.orelse:
        return handle(get_else(node), frame, graph, branch)
    
def handle_proxy(frame: FrameType, condition: "Proxy"):

    class Visitor(ast.NodeVisitor):
        def __init__(self, line_no):
            self.target = None
            self.line_no = line_no

        def visit_If(self, node):
            if node.lineno == self.line_no:
                self.target = node
            self.generic_visit(node)

    visitor = visit(frame, Visitor)

    if_node:ast.If = visitor.target
    
    graph = condition.node.graph

    branch = Condition(condition, parent=graph)

    def callback(node: ast.If, frame: FrameType, graph:Graph, branch:Condition):
        
        branch.__exit__(None, None, None)

        if node.orelse:
            handle(get_else(if_node), frame, graph, branch)

    branch.__enter__()
    end = frame.f_lineno + (if_node.end_lineno - if_node.lineno)
    execute_until(frame.f_lineno, end, frame, callback=lambda _: callback(if_node, frame, graph, branch))

    return True