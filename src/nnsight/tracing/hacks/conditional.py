import ast
import inspect
import sys
from types import FrameType
from typing import TYPE_CHECKING

from ..contexts import Condition, Context
from .util import execute, execute_body, execute_until

if TYPE_CHECKING:
    from ..graph import Proxy


def handle_conditional(frame: FrameType, condition: "Proxy"):

    line_no = frame.f_lineno
    source_lines, _ = inspect.getsourcelines(frame)
    source = "".join(source_lines)
    tree = ast.parse(source)

    class Visitor(ast.NodeVisitor):
        def __init__(self, line_no):
            self.target = None
            self.line_no = line_no

        def visit_If(self, node):
            if node.lineno == self.line_no:
                self.target = node
            self.generic_visit(node)

    visitor = Visitor(line_no)
    visitor.visit(tree)

    if_node = visitor.target

    graph = condition.node.graph

    branch = Condition(condition, parent=graph)

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

    def evaluate_and_execute(node: ast.stmt):

        nonlocal branch

        if isinstance(node, ast.If):

            condition_expr = ast.Expression(
                body=node.test, lineno=node.lineno, col_offset=node.col_offset
            )

            condition = execute(condition_expr, frame)

            with branch.else_(condition) as branch:
                execute_body(node.body, frame)

            if node.orelse:
                return evaluate_and_execute(get_else(node))

    def callback(node: ast.If, context: Context, frame: FrameType):

        if node.orelse:
            evaluate_and_execute(get_else(if_node))

    branch.__enter__()
    execute_until(branch, if_node, frame, callback=callback)

    return True
