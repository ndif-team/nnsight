"""
Source-based serialization decorator for nnsight remote execution.

The @nnsight.remote decorator marks functions and classes as safe for remote
execution on NDIF servers. It validates at import time that the code:
- Has available source (via inspect.getsource)
- Only imports allowed modules (torch, numpy, math, builtins)
- Doesn't call disallowed functions (open, exec, eval, etc.)
- Only references JSON-serializable module-level constants
"""

from __future__ import annotations

import ast
import builtins
import inspect
import json
import sys
import types
from typing import Any, Callable, Dict, List, Set, Tuple, Type, Union

# Modules that are available on NDIF servers and don't need to be serialized
ALLOWED_MODULES = {'torch', 'numpy', 'math', 'builtins', 'collections', 'itertools', 'functools', 'operator'}

# Function calls that are never allowed in @remote code
DISALLOWED_CALLS = {
    'open', 'exec', 'eval', 'compile', 'input',
    '__import__',
}

# Attribute chains that indicate file/network/subprocess operations
DISALLOWED_ATTR_PATTERNS = {
    ('os', 'system'), ('os', 'popen'), ('os', 'spawn'),
    ('subprocess', 'run'), ('subprocess', 'call'), ('subprocess', 'Popen'),
    ('socket',), ('urllib',), ('requests',),
    ('pathlib', 'Path', 'read_text'), ('pathlib', 'Path', 'write_text'),
}

# Builtin names that don't need to be captured
BUILTIN_NAMES = set(dir(builtins))


class RemoteValidationError(ImportError):
    """Raised when @nnsight.remote validation fails."""
    pass


def remote(obj: Union[Type, Callable] = None, *, version: str = None, library: str = None) -> Union[Type, Callable]:
    """
    Decorator that marks a function or class as safe for NDIF remote execution.

    This decorator:
    1. Validates the code at import time for remote-safe patterns
    2. Captures source code and module-level references
    3. Enables source-based serialization instead of cloudpickle

    Usage:
        @nnsight.remote
        def normalize(x):
            return x / x.norm(dim=-1, keepdim=True)

        @nnsight.remote
        class Analyzer:
            def __init__(self, model, top_k=10):
                self.model = model
                self.top_k = top_k

        # With version info (for library authors):
        @nnsight.remote(library="nnterp", version="0.1.0")
        class StandardizedTransformer:
            ...

    Args:
        obj: Function or class to decorate
        version: Optional version string for the decorated object
        library: Optional library name (enables server-side caching)

    Returns:
        The decorated function or class with validation metadata

    Raises:
        RemoteValidationError: If the code fails validation
    """
    # Support both @remote and @remote(version="...", library="...")
    if obj is None:
        # Called with arguments: @remote(version="...")
        def wrapper(obj):
            return _apply_remote(obj, version=version, library=library)
        return wrapper
    else:
        # Called without arguments: @remote
        return _apply_remote(obj, version=version, library=library)


def _apply_remote(obj: Union[Type, Callable], version: str = None, library: str = None) -> Union[Type, Callable]:
    """Internal implementation of the @remote decorator."""

    # Auto-detect library/version from package metadata if not provided
    if library is None and hasattr(obj, '__module__'):
        module_name = obj.__module__
        if module_name:
            # Try to get package info
            root_package = module_name.split('.')[0]
            try:
                from importlib.metadata import version as get_version
                detected_version = get_version(root_package)
                library = root_package
                version = detected_version
            except Exception:
                pass  # Not a installed package, no version info

    # Step 1: Verify source is available
    try:
        source = inspect.getsource(obj)
    except OSError:
        raise RemoteValidationError(
            f"@nnsight.remote requires source code for '{obj.__name__}'. "
            f"Ensure .py files are included in your package distribution."
        )

    # Step 2: Parse AST
    # Dedent the source code to handle functions/classes defined inside other functions
    try:
        import textwrap
        dedented_source = textwrap.dedent(source)
        tree = ast.parse(dedented_source)
    except SyntaxError as e:
        raise RemoteValidationError(
            f"@nnsight.remote failed to parse source for '{obj.__name__}': {e}"
        )

    # Step 3: Find external references (names used but not defined locally)
    external_names = find_external_references(tree, obj)

    # Step 4: Resolve module-level references and validate them
    module_refs, resolution_errors = resolve_module_references(external_names, obj)

    # Step 5: Validate AST for disallowed patterns
    ast_errors = validate_ast(tree, obj.__name__)

    # Step 6: For classes, additional validation
    class_errors = []
    if isinstance(obj, type):
        class_errors = validate_class(obj)

    # Collect all errors
    all_errors = resolution_errors + ast_errors + class_errors

    if all_errors:
        raise RemoteValidationError(format_validation_errors(obj.__name__, all_errors))

    # Step 7: Store metadata for serialization
    obj._remote_source = dedented_source  # Use dedented source for clean serialization
    obj._remote_module_refs = module_refs
    obj._remote_validated = True
    obj._remote_library = library  # Package name, e.g., "nnterp"
    obj._remote_version = version  # Package version, e.g., "0.1.0"

    return obj


def find_external_references(tree: ast.AST, obj: Union[Type, Callable]) -> Set[str]:
    """
    Find names that are referenced but not defined locally within the code.

    This walks the AST to find all Name nodes in Load context, excluding
    names that are defined locally (function parameters, local variables,
    class attributes, etc.).
    """

    class ReferenceCollector(ast.NodeVisitor):
        def __init__(self):
            self.referenced = set()
            self.defined = set()
            self.scopes = [set()]  # Stack of local scopes

        def push_scope(self):
            self.scopes.append(set())

        def pop_scope(self):
            self.scopes.pop()

        def define(self, name: str):
            self.scopes[-1].add(name)
            self.defined.add(name)

        def is_defined(self, name: str) -> bool:
            return any(name in scope for scope in self.scopes)

        def visit_FunctionDef(self, node):
            # Function name is defined in outer scope
            self.define(node.name)

            # Function parameters are defined in inner scope
            self.push_scope()
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                self.define(arg.arg)
            if node.args.vararg:
                self.define(node.args.vararg.arg)
            if node.args.kwarg:
                self.define(node.args.kwarg.arg)

            # Visit function body
            for child in node.body:
                self.visit(child)

            self.pop_scope()

        def visit_AsyncFunctionDef(self, node):
            # Same as FunctionDef
            self.visit_FunctionDef(node)

        def visit_ClassDef(self, node):
            # Class name is defined in outer scope
            self.define(node.name)

            # Class body has its own scope
            self.push_scope()
            for child in node.body:
                self.visit(child)
            self.pop_scope()

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                if not self.is_defined(node.id):
                    self.referenced.add(node.id)
            elif isinstance(node.ctx, ast.Store):
                self.define(node.id)

        def visit_For(self, node):
            # Target is defined before visiting body
            self.visit(node.target)
            self.visit(node.iter)
            for child in node.body:
                self.visit(child)
            for child in node.orelse:
                self.visit(child)

        def visit_comprehension(self, node):
            # The target of a comprehension is defined
            self.visit(node.target)
            self.visit(node.iter)
            for if_ in node.ifs:
                self.visit(if_)

        def visit_ListComp(self, node):
            self.push_scope()
            for gen in node.generators:
                self.visit_comprehension(gen)
            self.visit(node.elt)
            self.pop_scope()

        def visit_SetComp(self, node):
            self.push_scope()
            for gen in node.generators:
                self.visit_comprehension(gen)
            self.visit(node.elt)
            self.pop_scope()

        def visit_DictComp(self, node):
            self.push_scope()
            for gen in node.generators:
                self.visit_comprehension(gen)
            self.visit(node.key)
            self.visit(node.value)
            self.pop_scope()

        def visit_GeneratorExp(self, node):
            self.push_scope()
            for gen in node.generators:
                self.visit_comprehension(gen)
            self.visit(node.elt)
            self.pop_scope()

        def visit_Import(self, node):
            for alias in node.names:
                name = alias.asname or alias.name.split('.')[0]
                self.define(name)

        def visit_ImportFrom(self, node):
            for alias in node.names:
                name = alias.asname or alias.name
                self.define(name)

        def visit_ExceptHandler(self, node):
            if node.name:
                self.define(node.name)
            for child in node.body:
                self.visit(child)

        def visit_With(self, node):
            for item in node.items:
                self.visit(item.context_expr)
                if item.optional_vars:
                    self.visit(item.optional_vars)
            for child in node.body:
                self.visit(child)

        def visit_NamedExpr(self, node):
            # Walrus operator (:=) defines the target
            self.define(node.target.id)
            self.visit(node.value)

    collector = ReferenceCollector()
    collector.visit(tree)

    # Return names that are referenced but not defined locally
    # Filter out builtins - they don't need to be captured
    external = collector.referenced - BUILTIN_NAMES

    return external


def resolve_module_references(names: Set[str], obj: Union[Type, Callable]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Resolve external names to their values from the function/class globals.

    Returns:
        (captured_refs, errors) where:
        - captured_refs: dict of JSON-serializable module-level constants
        - errors: list of error messages for non-serializable references

    Handles three cases:
    1. Module aliases (np, F, torch) -> skip if allowed module
    2. JSON-serializable constants (TOP_K = 10) -> capture
    3. Non-serializable objects -> error
    """
    captured = {}
    errors = []

    # Get globals from the decorated object
    if hasattr(obj, '__globals__'):
        obj_globals = obj.__globals__
    else:
        # For classes, get the module globals
        module = sys.modules.get(obj.__module__)
        obj_globals = getattr(module, '__dict__', {}) if module else {}

    for name in names:
        # Skip if it's a builtin
        if name in BUILTIN_NAMES:
            continue

        # Get the actual value from globals
        if name not in obj_globals:
            # Could be a nested scope variable or truly undefined
            # We'll let Python's runtime handle truly undefined names
            continue

        value = obj_globals[name]

        # Case 1: Module or module alias (np, F, torch, etc.)
        if isinstance(value, types.ModuleType):
            root = value.__name__.split('.')[0]
            if root in ALLOWED_MODULES:
                # Available on server, no need to capture
                continue
            else:
                errors.append(
                    f"Module '{name}' ({value.__name__}) not available on NDIF server. "
                    f"Allowed: {', '.join(sorted(ALLOWED_MODULES))}"
                )
            continue

        # Case 2: Check if it's a @remote-decorated function/class
        if getattr(value, '_remote_validated', False):
            # Will be serialized separately, skip here
            continue

        # Case 3: Try to JSON-serialize
        if is_json_serializable(value):
            captured[name] = value
            continue

        # Case 4: Type references (like int, str, list, etc.)
        if isinstance(value, type):
            if value.__module__ in ('builtins', 'typing'):
                continue
            # Check if it's a @remote class
            if getattr(value, '_remote_validated', False):
                continue
            errors.append(
                f"Reference '{name}' (type '{value.__name__}' from module '{value.__module__}') "
                f"is not @nnsight.remote decorated and cannot be serialized"
            )
            continue

        # Case 5: Functions from allowed modules
        if callable(value):
            if hasattr(value, '__module__'):
                root = value.__module__.split('.')[0] if value.__module__ else ''
                if root in ALLOWED_MODULES:
                    continue

        # Case 6: Non-serializable object
        type_name = type(value).__name__
        errors.append(
            f"Reference '{name}' (type '{type_name}') is not JSON-serializable.\n"
            f"  Options:\n"
            f"    - Make it a class/instance attribute instead\n"
            f"    - Pass it as a function/method argument\n"
            f"    - Use a JSON-serializable type (int, float, str, list, dict, bool, None)"
        )

    return captured, errors


def is_json_serializable(value: Any) -> bool:
    """
    Check if a value can be JSON-serialized.

    Allows: None, bool, int, float, str, list/tuple (of serializable), dict (str keys, serializable values)
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return True

    if isinstance(value, (list, tuple)):
        return all(is_json_serializable(item) for item in value)

    if isinstance(value, dict):
        return all(
            isinstance(k, str) and is_json_serializable(v)
            for k, v in value.items()
        )

    # Try actual JSON serialization as final check
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def validate_ast(tree: ast.AST, name: str) -> List[str]:
    """
    Validate AST for disallowed patterns.

    Checks for:
    - Imports of non-allowed modules
    - Calls to disallowed functions (open, exec, eval, etc.)
    """
    errors = []

    class Validator(ast.NodeVisitor):
        def visit_Import(self, node):
            for alias in node.names:
                module = alias.name.split('.')[0]
                if module not in ALLOWED_MODULES:
                    errors.append(f"Line {node.lineno}: imports '{alias.name}' (not available on NDIF server)")
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            module = (node.module or '').split('.')[0]
            if module not in ALLOWED_MODULES:
                errors.append(f"Line {node.lineno}: imports from '{node.module}' (not available on NDIF server)")
            self.generic_visit(node)

        def visit_Call(self, node):
            # Check for disallowed function calls
            if isinstance(node.func, ast.Name):
                if node.func.id in DISALLOWED_CALLS:
                    errors.append(f"Line {node.lineno}: calls '{node.func.id}()' (not allowed in @nnsight.remote code)")

            # Check for disallowed attribute calls like os.system()
            if isinstance(node.func, ast.Attribute):
                chain = self._get_attr_chain(node.func)
                for pattern in DISALLOWED_ATTR_PATTERNS:
                    if len(chain) >= len(pattern):
                        if chain[:len(pattern)] == pattern:
                            errors.append(f"Line {node.lineno}: calls '{'.'.join(chain)}()' (not allowed in @nnsight.remote code)")
                            break

            self.generic_visit(node)

        def _get_attr_chain(self, node) -> Tuple[str, ...]:
            """Get the chain of attribute names (e.g., os.path.join -> ('os', 'path', 'join'))"""
            if isinstance(node, ast.Attribute):
                parent_chain = self._get_attr_chain(node.value)
                return parent_chain + (node.attr,)
            elif isinstance(node, ast.Name):
                return (node.id,)
            else:
                return ()

    Validator().visit(tree)
    return errors


def validate_class(cls: type) -> List[str]:
    """
    Additional validation for classes.

    Checks:
    - Base classes are @nnsight.remote or object
    - No metaclass
    - No __slots__
    """
    errors = []

    # Check base classes
    for base in cls.__bases__:
        if base is object:
            continue
        if not getattr(base, '_remote_validated', False):
            errors.append(
                f"Base class '{base.__name__}' is not @nnsight.remote decorated. "
                f"All base classes must be @nnsight.remote or object."
            )

    # Check for metaclass
    if type(cls) is not type:
        errors.append(
            f"Uses metaclass '{type(cls).__name__}'. "
            f"@nnsight.remote classes cannot use metaclasses."
        )

    # Check for __slots__
    if hasattr(cls, '__slots__') and cls.__slots__:
        errors.append(
            f"Uses __slots__. "
            f"@nnsight.remote classes cannot use __slots__."
        )

    return errors


def format_validation_errors(name: str, errors: List[str]) -> str:
    """Format validation errors into a readable error message."""
    error_lines = '\n'.join(f"  - {e}" for e in errors)
    return (
        f"@nnsight.remote validation failed for '{name}':\n\n"
        f"{error_lines}\n\n"
        f"For remote execution, code must:\n"
        f"  - Only import allowed modules (torch, numpy, math)\n"
        f"  - Use JSON-serializable module-level constants\n"
        f"  - Avoid file I/O, network, and subprocess operations\n"
        f"  - Use @nnsight.remote on all helper functions/classes"
    )


# Re-export for convenient access
__all__ = ['remote', 'RemoteValidationError', 'is_json_serializable', 'ALLOWED_MODULES']
