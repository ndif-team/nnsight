import ast
import inspect
import sys
from types import FrameType
from typing import Any, Callable, List, Optional, Type

from ..contexts import Context
from ..graph import Graph


def execute(expr: ast.expr, frame: FrameType) -> Any:
    ast.fix_missing_locations(expr)
    return eval(
        compile(expr, "<string>", "eval"),
        frame.f_globals,
        frame.f_locals,
    )


def execute_body(body: List[ast.stmt], frame: FrameType, graph: Graph) -> None:

    from . import handle_inner

    for stmt in body:

        if not handle_inner(stmt, frame, graph):
            module = ast.Module(body=[stmt], type_ignores=[])
            ast.fix_missing_locations(module)
            exec(
                compile(module, "<string>", "exec"),
                frame.f_globals,
                frame.f_locals,
            )


def execute_until(
    context: Context,
    first_line: int,
    last_line: int,
    frame: FrameType,
    callback: Optional[Callable] = None,
):

    prev_trace = frame.f_trace

    def trace(new_frame: FrameType, *args):

        if new_frame.f_code.co_filename == frame.f_code.co_filename and (
            new_frame.f_lineno > last_line or new_frame.f_lineno < first_line
        ):
            frame.f_trace = prev_trace
            sys.settrace(prev_trace)

            context.__exit__(None, None, None)

            if callback is not None:

                callback()

    frame.f_trace = trace
    sys.settrace(trace)


def visit(frame: FrameType, visitor_cls: Type[ast.NodeVisitor]) -> ast.stmt:

    line_no = frame.f_lineno
    source_lines, inner_line_no = inspect.getsourcelines(frame)
    if inner_line_no > 0:
        line_no = line_no - inner_line_no + 1

    source = "".join(source_lines).lstrip()
    tree = ast.parse(source)

    visitor = visitor_cls(line_no)
    visitor.visit(tree)

    return visitor.target
