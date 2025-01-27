import ast
import ctypes
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

            if prev_trace is not None:

                prev_trace(new_frame, *args)

            if callback is not None:

                callback(new_frame)

    frame.f_trace = trace
    sys.settrace(trace)


def is_ipython():
    return "_ih" in locals()


def visit(frame: FrameType, visitor_cls: Type[ast.NodeVisitor]) -> ast.stmt:

    line_no = frame.f_lineno

    if "_ih" in frame.f_locals:
        import IPython

        ipython = IPython.get_ipython()
        source_lines = ipython.user_global_ns["_ih"][-1]
        inner_line_no = 0

    else:
        source_lines, inner_line_no = inspect.getsourcelines(frame)

    if inner_line_no > 0:
        line_no = line_no - inner_line_no + 1

        shift = len(source_lines[0]) - len(source_lines[0].lstrip())

        if shift > 0:

            source_lines = [source_line[shift:] for source_line in source_lines]

    source = "".join(source_lines)

    tree = ast.parse(source)

    visitor = visitor_cls(line_no)
    visitor.visit(tree)

    return visitor
