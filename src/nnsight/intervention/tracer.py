import ast
import re
from typing import TYPE_CHECKING, Any, Optional, Callable, List

from .interleaver import Interleaver
from ..tracing.tracer import Tracer
from ..tracing.compiler import indent, try_catch

if TYPE_CHECKING:
    from .envoy import Envoy


class InterleavingTracer(Tracer):
    """
    A tracer that enables interleaving of model execution with user-defined interventions.
    
    This tracer extends the base Tracer class to support intercepting model execution
    and allowing for custom interventions at specific points in the model's forward pass.
    """

    def __init__(self, model: "Envoy", module: Any, fn: Callable, *args, **kwargs):
        """
        Initialize the InterleavingTracer.
        
        Args:
            model: The Envoy object representing the model to trace
            module: The module being traced
            fn: The function to be traced
            *args: Additional arguments to pass to the function
            **kwargs: Additional keyword arguments to pass to the function
        """
        self.model = model
        self.fn = fn
        super().__init__(model, fn, *args, **kwargs)

    def trace(self, info: Tracer.Info):
        """
        Process the AST to extract information about with-statements and prepare the source code
        for execution with interleaving.
        
        Args:
            info: Information about the AST node and source code to trace
        """
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
        
        # Prepare the inner function that will be executed with interleaving
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
            """
            Convert a 'with model.trace()' statement to an execute call.
            
            Args:
                code: The source code line containing the with statement
                
            Returns:
                The converted code as an execute call
            """
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
        """
        Execute the traced function with the provided interventions.
        
        Args:
            interventions: A callable containing the intervention logic
            model: The Envoy object representing the model
        """
        with Interleaver(interventions) as interleaver:
            model._set_interleaver(interleaver)
            interleaver(self.fn, *self.args, **self.kwargs)
            
        model._clear()
