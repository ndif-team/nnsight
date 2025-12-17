import ast
import inspect
import sys
import textwrap
from builtins import compile, exec
from collections import defaultdict
from typing import Callable

import astor

# Use ast.unparse if available (Python 3.9+), otherwise fall back to astor
if sys.version_info >= (3, 9):
    _ast_to_source = ast.unparse
else:
    _ast_to_source = astor.to_source


class FunctionCallWrapper(ast.NodeTransformer):

    def __init__(self, name: str):
        self.name_index = defaultdict(int)
        self.line_numbers = {}
        self.name = name
        # Cache the name prefix to avoid repeated string operations
        self._name_prefix = f"{name}."
        # Track the line number of the first function definition encountered
        # Only wrap calls that occur after this line (inside the function body)
        self._function_start_line = None

    def get_name(self, node: ast.Call):
        """Extract and index function name from a Call node."""
        func = node.func
        if isinstance(func, ast.Name):
            # Simple function call like foo()
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            # Method call like obj.method() or module.submodule.func()
            # Build parts in reverse order to avoid reversing later
            parts = []
            current = func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            # Join in reverse order (most specific to least specific)
            func_name = "_".join(reversed(parts))
        else:
            # Fallback for other call types
            func_name = "unknown"

        index = self.name_index[func_name]
        self.name_index[func_name] = index + 1
        return f"{func_name}_{index}"

    def visit_Call(self, node):
        # Only wrap calls that occur after the function definition line
        # This excludes decorators which appear before the function definition
        if (
            self._function_start_line is not None
            and node.lineno <= self._function_start_line
        ):
            return self.generic_visit(node)

        self.generic_visit(node)  # First, process nested calls
        # Get the fully qualified name of the function being called
        func_name = self.get_name(node)
        self.line_numbers[func_name] = node.lineno - 2

        # Build the wrapped call name string using cached prefix
        wrapped_name = f"{self._name_prefix}{func_name}"

        return ast.Call(
            func=ast.Call(
                func=ast.Name(id="wrap", ctx=ast.Load()),
                args=[node.func],
                keywords=[
                    ast.keyword(arg="name", value=ast.Constant(value=wrapped_name))
                ],
            ),
            args=node.args,
            keywords=node.keywords,
        )

    def visit_FunctionDef(self, node):
        # Record the first function definition's line number
        if self._function_start_line is None:
            self._function_start_line = node.lineno
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        # Record the first async function definition's line number
        if self._function_start_line is None:
            self._function_start_line = node.lineno
        return self.generic_visit(node)


def convert(fn: Callable, wrap: Callable, name: str):

    # TODO what about exceptions?

    source = textwrap.dedent(inspect.getsource(fn))

    # Get the module where the forward method is defined
    module_globals = inspect.getmodule(fn).__dict__

    tree = ast.parse(source)
    transformer = FunctionCallWrapper(name)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    local_namespace = {"wrap": wrap}

    # Include both globals from this module and the module where forward is defined
    # Optimize: avoid creating intermediate dicts when possible
    global_namespace = globals().copy()
    global_namespace.update(module_globals)
    global_namespace["wrap"] = wrap

    filename = "<nnsight>"

    # Compile directly from AST (Python 3.8+), which is faster than converting to source first
    # However, compile() with AST requires mode='exec' and the AST must be a Module node
    if isinstance(tree, ast.Module):
        code_obj = compile(tree, filename, "exec")
    else:
        # Fallback: convert to source if tree structure is unexpected
        code_obj = compile(_ast_to_source(tree), filename, "exec")
    try:
        exec(code_obj, global_namespace, local_namespace)
    except Exception as exc:
        ee = exc
        breakpoint()

    fn = local_namespace[fn.__name__]

    return source, transformer.line_numbers, fn
