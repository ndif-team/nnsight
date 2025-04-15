import ast
import re
from .interleaver import Interleaver
from typing import TYPE_CHECKING, Any, Optional, Callable, List
from ..tracing.tracer import Tracer
from ..tracing.compiler import indent, try_catch

if TYPE_CHECKING:
    from .envoy import Envoy


class InterleavingTracer(Tracer):

    def __init__(self, fn: Callable, *args, **kwargs):
        self.fn = fn
        super().__init__(fn, *args, **kwargs)

    def trace(self, info: Tracer.Info):
        class Visitor(ast.NodeVisitor):
            def __init__(self, line_no):
                self.target = None
                self.object_name = None
                self.context_name = None
                self.line_no = line_no
                self.inner_traces = []

            def visit_With(self, node):
                if node.lineno == self.line_no:
                    self.target = node
                    self.object_name = node.items[0].context_expr.func.value.id
                    self.context_name = (
                        node.items[0].optional_vars.id
                        if node.items[0].optional_vars
                        else None
                    )
                    self.generic_visit(node)
                elif (
                    node.lineno <= self.target.end_lineno
                    and node.items[0].context_expr.func.value.id
                    == self.object_name
                ):
                    self.inner_traces.append(node)
                else:
                    self.generic_visit(node)

        visitor = Visitor(info.start_line)
        visitor.visit(info.node)

        source = info.source
        
        source = [
            "async def inner(interleaver):",
            *indent(
                try_catch(
                    [line[4:] for line in source[1:]] + ["user_locals.update(locals())"],
                    exception_source=["interleaver.exception(exception)"],
                    else_source=["interleaver.continue_execution()"],
                )
            ),
        ]

        self.source.extend(source)
    

        def convert_with_trace(code: str) -> str:
            # Pattern with optional "as" clause
            pattern = r'with\s+([\w\.]+\([^)]*\))(\s+as\s+(\w+))?:'
            match = re.search(pattern, code)
            if match:
                expr = match.group(1)  # model.trace(...) part
                var = match.group(3)  # optional variable after "as"
                if var:
                    return f"{var} = {expr}.execute(inner, model)"
                else:
                    return f"{expr}.execute(inner, model)"
            return code  # Return original if no match

        self.source.append(f"{convert_with_trace(info.source[0])}\n")

    def execute(self, interventions: Callable, model: "Envoy"):
        with Interleaver(interventions) as interleaver:
            model._set_interleaver(interleaver)
            interleaver(self.fn, *self.args, **self.kwargs)
            
        model._clear()
