import ast
import sys
from types import FrameType
from typing import Any, Callable, List, Optional

from ..contexts import Context


def execute(expr: ast.expr, frame: FrameType) -> Any:
    ast.fix_missing_locations(expr)
    return eval(
        compile(expr, "<string>", "eval"),
        frame.f_globals,
        frame.f_locals,
    )


def execute_body(body: List[ast.stmt], frame: FrameType) -> None:

    for stmt in body:
        module = ast.Module(body=[stmt], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(
            compile(module, "<string>", "exec"),
            frame.f_globals,
            frame.f_locals,
        )


def execute_until(
    context: Context,
    node: ast.If,
    frame: FrameType,
    callback: Optional[Callable] = None,
):

    last_line = node.body[-1].lineno
    first_line =  node.body[0].lineno
    
    prev_trace = frame.f_trace

    def trace(new_frame: FrameType, *args):

        if (
            new_frame.f_code.co_filename == frame.f_code.co_filename
            and (new_frame.f_lineno > last_line or new_frame.f_lineno < first_line)
        ):
            frame.f_trace = prev_trace
            sys.settrace(prev_trace)

            context.__exit__(None, None, None)

            if callback is not None:

                callback(node, context, frame)

    frame.f_trace = trace
    sys.settrace(trace)
