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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

# Modules that are available on NDIF servers and don't need to be serialized
ALLOWED_MODULES = {
    'torch', 'numpy', 'math', 'builtins',
    'collections', 'itertools', 'functools', 'operator',
    'os', 'pathlib', 'random',  # Safe when restricted (no I/O operations)
}

# Function calls that are never allowed in @remote code
DISALLOWED_CALLS = {
    'open', 'exec', 'eval', 'compile', 'input',
    '__import__',
}

# Attribute chains that indicate dangerous operations (file I/O, network, subprocess, etc.)
# Format: tuple of attribute names forming the chain. Matching is prefix-based.
DISALLOWED_ATTR_PATTERNS = {
    # os module - block process and filesystem operations
    ('os', 'system'), ('os', 'popen'), ('os', 'spawn'), ('os', 'spawnl'),
    ('os', 'spawnle'), ('os', 'spawnlp'), ('os', 'spawnlpe'), ('os', 'spawnv'),
    ('os', 'spawnve'), ('os', 'spawnvp'), ('os', 'spawnvpe'),
    ('os', 'exec'), ('os', 'execl'), ('os', 'execle'), ('os', 'execlp'),
    ('os', 'execlpe'), ('os', 'execv'), ('os', 'execve'), ('os', 'execvp'),
    ('os', 'execvpe'),
    ('os', 'fork'), ('os', 'forkpty'), ('os', 'kill'), ('os', 'killpg'),
    ('os', 'remove'), ('os', 'unlink'), ('os', 'rmdir'), ('os', 'removedirs'),
    ('os', 'rename'), ('os', 'renames'), ('os', 'replace'),
    ('os', 'mkdir'), ('os', 'makedirs'), ('os', 'symlink'), ('os', 'link'),
    ('os', 'chdir'), ('os', 'chroot'), ('os', 'chmod'), ('os', 'chown'),
    ('os', 'lchown'), ('os', 'chflags'), ('os', 'lchflags'),
    ('os', 'open'), ('os', 'fdopen'), ('os', 'read'), ('os', 'write'),
    ('os', 'truncate'), ('os', 'ftruncate'),
    # subprocess module
    ('subprocess', 'run'), ('subprocess', 'call'), ('subprocess', 'Popen'),
    ('subprocess', 'check_call'), ('subprocess', 'check_output'),
    ('subprocess', 'getoutput'), ('subprocess', 'getstatusoutput'),
    # socket module
    ('socket',),
    # urllib module
    ('urllib',),
    # requests module
    ('requests',),
    # pathlib - block I/O operations but allow path manipulation
    ('pathlib', 'Path', 'read_text'), ('pathlib', 'Path', 'write_text'),
    ('pathlib', 'Path', 'read_bytes'), ('pathlib', 'Path', 'write_bytes'),
    ('pathlib', 'Path', 'open'), ('pathlib', 'Path', 'unlink'),
    ('pathlib', 'Path', 'rmdir'), ('pathlib', 'Path', 'rename'),
    ('pathlib', 'Path', 'replace'), ('pathlib', 'Path', 'symlink_to'),
    ('pathlib', 'Path', 'link_to'), ('pathlib', 'Path', 'mkdir'),
    ('pathlib', 'Path', 'touch'), ('pathlib', 'Path', 'chmod'),
    ('pathlib', 'Path', 'lchmod'),
    # shutil module (file operations)
    ('shutil',),
}

# Builtin names that don't need to be captured
BUILTIN_NAMES = set(dir(builtins))

# Trusted base classes from allowed modules (don't require @remote)
# These are well-known classes whose __dict__ contains all necessary state
ALLOWED_BASE_CLASSES = {
    # torch.nn module
    'torch.nn.Module',
    'torch.nn.modules.module.Module',
    # torch.utils.data module
    'torch.utils.data.Dataset',
    'torch.utils.data.IterableDataset',
    'torch.utils.data.dataset.Dataset',
    'torch.utils.data.dataset.IterableDataset',
}


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

    # Step 5: Extract and validate closure variables (for nested functions)
    closure_vars, closure_errors = extract_closure_variables(obj)
    module_refs.update(closure_vars)  # Merge closure vars into module refs

    # Step 6: Validate AST for disallowed patterns
    ast_errors = validate_ast(tree, obj.__name__)

    # Step 7: For classes, additional validation
    class_errors = []
    if isinstance(obj, type):
        class_errors = validate_class(obj)

    # Collect all errors
    all_errors = resolution_errors + closure_errors + ast_errors + class_errors

    if all_errors:
        raise RemoteValidationError(format_validation_errors(obj.__name__, all_errors))

    # Step 8: Store metadata for serialization
    obj._remote_source = dedented_source  # Use dedented source for clean serialization
    obj._remote_module_refs = module_refs
    obj._remote_closure_vars = closure_vars  # Store closure vars separately for clarity
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


class ValueClassification:
    """Result of classifying a value for remote execution."""
    CAPTURE = "capture"      # Value should be captured (JSON-serializable)
    SKIP = "skip"            # Value available on server, no action needed
    ERROR = "error"          # Value cannot be serialized, validation error


def classify_reference_value(name: str, value: Any) -> Tuple[str, Optional[Any], Optional[str]]:
    """
    Classify a referenced value for remote execution.

    This is the shared logic for both module-level references and closure variables.

    Args:
        name: The variable name (for error messages)
        value: The actual value to classify

    Returns:
        (classification, captured_value, error_detail) where:
        - classification is one of ValueClassification.CAPTURE/SKIP/ERROR
        - captured_value is the value to capture (only if CAPTURE)
        - error_detail is additional context for errors (only if ERROR)
    """
    # Case 1: Module or module alias (np, F, torch, etc.)
    if isinstance(value, types.ModuleType):
        root = value.__name__.split('.')[0]
        if root in ALLOWED_MODULES:
            return ValueClassification.SKIP, None, None
        else:
            return ValueClassification.ERROR, None, f"module '{value.__name__}' not available on NDIF server"

    # Case 2: @remote-decorated function/class (will be serialized separately)
    if getattr(value, '_remote_validated', False):
        return ValueClassification.SKIP, None, None

    # Case 3: JSON-serializable constants
    if is_json_serializable(value):
        return ValueClassification.CAPTURE, value, None

    # Case 4: Type references from builtins/typing
    if isinstance(value, type):
        if value.__module__ in ('builtins', 'typing'):
            return ValueClassification.SKIP, None, None
        # Non-builtin type that's not @remote
        return ValueClassification.ERROR, None, f"type '{value.__name__}' from module '{value.__module__}' is not @nnsight.remote decorated"

    # Case 5: Functions/callables from allowed modules
    if callable(value) and hasattr(value, '__module__'):
        root = value.__module__.split('.')[0] if value.__module__ else ''
        if root in ALLOWED_MODULES:
            return ValueClassification.SKIP, None, None

    # Case 6: Non-serializable value
    type_name = type(value).__name__
    return ValueClassification.ERROR, None, f"type '{type_name}' is not JSON-serializable"


def resolve_module_references(names: Set[str], obj: Union[Type, Callable]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Resolve external names to their values from the function/class globals.

    Returns:
        (captured_refs, errors) where:
        - captured_refs: dict of JSON-serializable module-level constants
        - errors: list of error messages for non-serializable references
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
        classification, captured_value, error_detail = classify_reference_value(name, value)

        if classification == ValueClassification.CAPTURE:
            captured[name] = captured_value
        elif classification == ValueClassification.ERROR:
            errors.append(
                f"Reference '{name}' ({error_detail}).\n"
                f"  Options:\n"
                f"    - Make it a class/instance attribute instead\n"
                f"    - Pass it as a function/method argument\n"
                f"    - Use a JSON-serializable type (int, float, str, list, dict, bool, None)"
            )

    return captured, errors


def extract_closure_variables(obj: Union[Type, Callable]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Extract and validate closure variables from a function or class methods.

    When a function is defined inside another function and captures variables
    from the enclosing scope, those variables are stored in __closure__ and
    their names in __code__.co_freevars.

    For classes, we check all methods for closures since methods like
    __init_subclass__ may capture variables from the defining scope.

    Args:
        obj: The function or class to extract closure variables from

    Returns:
        (captured_vars, errors) where:
        - captured_vars: dict of JSON-serializable closure variables
        - errors: list of error messages for non-serializable closure variables
    """
    captured = {}
    errors = []

    # For classes, extract closures from all methods
    if isinstance(obj, type):
        for attr_name in dir(obj):
            # Skip inherited methods from object/type
            if attr_name.startswith('__') and attr_name.endswith('__'):
                if attr_name not in ('__init__', '__init_subclass__', '__new__', '__call__'):
                    continue
            try:
                attr = getattr(obj, attr_name)
            except AttributeError:
                continue

            # Get the underlying function from methods
            func = None
            if isinstance(attr, types.FunctionType):
                func = attr
            elif hasattr(attr, '__func__'):
                func = attr.__func__

            if func is not None:
                method_captured, method_errors = _extract_closure_from_function(func, attr_name)
                captured.update(method_captured)
                errors.extend(method_errors)

        return captured, errors

    # For functions, extract directly
    return _extract_closure_from_function(obj, obj.__name__ if hasattr(obj, '__name__') else '<function>')


def _extract_closure_from_function(func: Callable, context_name: str) -> Tuple[Dict[str, Any], List[str]]:
    """
    Extract closure variables from a single function.

    Args:
        func: The function to extract closure variables from
        context_name: Name of the function/method for error messages

    Returns:
        (captured_vars, errors)
    """
    captured = {}
    errors = []

    # Check if the function has closure variables
    if not hasattr(func, '__closure__') or func.__closure__ is None:
        return captured, errors

    if not hasattr(func, '__code__') or not hasattr(func.__code__, 'co_freevars'):
        return captured, errors

    # Get the names and values of closure variables
    freevars = func.__code__.co_freevars
    closure_cells = func.__closure__

    if len(freevars) != len(closure_cells):
        return captured, errors  # Shouldn't happen, but be safe

    for name, cell in zip(freevars, closure_cells):
        try:
            value = cell.cell_contents
        except ValueError:
            # Cell is empty (variable was deleted or never assigned)
            continue

        # Skip if it's a builtin
        if name in BUILTIN_NAMES:
            continue

        # Skip __class__ (implicit closure from super() calls)
        if name == '__class__':
            continue

        # Use shared classification logic
        classification, captured_value, error_detail = classify_reference_value(name, value)

        if classification == ValueClassification.CAPTURE:
            captured[name] = captured_value
        elif classification == ValueClassification.ERROR:
            errors.append(
                f"In '{context_name}': closure variable '{name}' ({error_detail}).\n"
                f"  Options:\n"
                f"    - Pass it as a function argument instead\n"
                f"    - Use a JSON-serializable type (int, float, str, list, dict, bool, None)\n"
                f"    - Mark it with @nnsight.remote if it's a custom function/class"
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
            """Get the chain of attribute names (e.g., os.path.join -> ('os', 'path', 'join'))

            Also handles cases like pathlib.Path("x").read_text() where we need to
            follow through Call nodes to get ('pathlib', 'Path', 'read_text').
            """
            if isinstance(node, ast.Attribute):
                parent_chain = self._get_attr_chain(node.value)
                return parent_chain + (node.attr,)
            elif isinstance(node, ast.Name):
                return (node.id,)
            elif isinstance(node, ast.Call):
                # Follow through call nodes, e.g., Path("x").read_text()
                # Get the chain from the call's func (the thing being called)
                return self._get_attr_chain(node.func)
            else:
                return ()

    Validator().visit(tree)
    return errors


def validate_class(cls: type) -> List[str]:
    """
    Additional validation for classes.

    Checks:
    - Base classes are @nnsight.remote, object, or in ALLOWED_BASE_CLASSES
    - No metaclass (except for allowed base classes like nn.Module)
    - No __slots__
    """
    errors = []

    # Check base classes
    for base in cls.__bases__:
        if base is object:
            continue
        # Check if base is in allowed base classes
        base_fullname = f"{base.__module__}.{base.__name__}"
        if base_fullname in ALLOWED_BASE_CLASSES:
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

    # Note: __slots__ classes are now supported via special serialization handling
    # in serialize_instance_state()

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


# =============================================================================
# Lambda source extraction
# =============================================================================

class LambdaExtractionError(Exception):
    """Raised when lambda source cannot be reliably extracted."""
    pass


def extract_lambda_source(func: Callable) -> str:
    """
    Extract the exact source of a lambda function, even when multiple
    lambdas appear on the same line.

    Uses AST parsing + bytecode matching to disambiguate lambdas.

    Args:
        func: A lambda function

    Returns:
        The source code of just the lambda expression (e.g., "lambda x: x + 1")

    Raises:
        LambdaExtractionError: If the lambda source cannot be reliably extracted
    """
    # Verify it's a lambda
    if not callable(func) or func.__name__ != '<lambda>':
        raise LambdaExtractionError(
            f"Expected a lambda function, got {type(func).__name__}"
        )

    # Get the full source line(s)
    try:
        full_source = inspect.getsource(func)
    except OSError as e:
        raise LambdaExtractionError(
            f"Could not get source for lambda. "
            f"Lambda must be defined in a .py file, not interactively. "
            f"Consider using a named function instead."
        ) from e

    # Check for multi-line lambda (problematic)
    lines = full_source.strip().split('\n')
    if len(lines) > 1:
        # Try to detect if this is a multi-line lambda expression
        # Multi-line lambdas are rare and problematic
        joined = ' '.join(line.strip() for line in lines)
        try:
            tree = ast.parse(joined)
        except SyntaxError:
            raise LambdaExtractionError(
                f"Multi-line lambda expressions are not supported for remote execution. "
                f"Please convert to a named function:\n\n"
                f"  # Instead of:\n"
                f"  f = (\n"
                f"      lambda x:\n"
                f"          x * 2\n"
                f"  )\n\n"
                f"  # Use:\n"
                f"  @nnsight.remote\n"
                f"  def f(x):\n"
                f"      return x * 2"
            )
        full_source = joined

    # Parse to find all lambda nodes
    try:
        tree = ast.parse(full_source.strip())
    except SyntaxError as e:
        raise LambdaExtractionError(
            f"Could not parse lambda source: {e}"
        ) from e

    # Collect all lambda nodes with their positions
    lambdas = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            if not hasattr(node, 'end_col_offset'):
                raise LambdaExtractionError(
                    f"Lambda extraction requires Python 3.8+. "
                    f"Please convert to a named function."
                )
            text = full_source.strip()[node.col_offset:node.end_col_offset]
            lambdas.append(text)

    if len(lambdas) == 0:
        raise LambdaExtractionError(
            f"No lambda found in source. This may be a nested inner lambda, "
            f"which is not supported. Please convert to a named function."
        )

    if len(lambdas) == 1:
        # Easy case: only one lambda on the line
        return lambdas[0]

    # Multiple lambdas on same line - disambiguate by bytecode
    target_bytecode = func.__code__.co_code

    for lambda_text in lambdas:
        try:
            compiled = compile(lambda_text, '<lambda>', 'eval')
            # The compiled code has the lambda as a constant
            if compiled.co_consts and hasattr(compiled.co_consts[0], 'co_code'):
                lambda_code = compiled.co_consts[0]
                if lambda_code.co_code == target_bytecode:
                    return lambda_text
        except Exception:
            continue

    # Could not match - might be identical lambdas or a closure issue
    raise LambdaExtractionError(
        f"Could not disambiguate lambda from {len(lambdas)} lambdas on the same line. "
        f"This can happen with:\n"
        f"  - Identical lambdas: `f1, f2 = lambda x: x, lambda x: x`\n"
        f"  - Lambdas with different closure bindings\n\n"
        f"Please convert to named functions:\n\n"
        f"  @nnsight.remote\n"
        f"  def my_func(x):\n"
        f"      return x"
    )


def is_lambda(func: Any) -> bool:
    """Check if an object is a lambda function."""
    return callable(func) and getattr(func, '__name__', None) == '<lambda>'


def validate_lambda_for_remote(func: Callable) -> Tuple[str, List[str]]:
    """
    Validate a lambda function for remote execution and extract its source.

    Args:
        func: A lambda function to validate

    Returns:
        (source, errors) tuple where source is the extracted lambda source
        and errors is a list of validation error messages

    Note:
        This performs the same validation as @nnsight.remote but for lambdas.
    """
    errors = []

    # Try to extract source
    try:
        source = extract_lambda_source(func)
    except LambdaExtractionError as e:
        return "", [str(e)]

    # Parse and validate AST
    try:
        # Wrap in expression for parsing
        tree = ast.parse(source, mode='eval')
    except SyntaxError as e:
        return source, [f"Could not parse lambda: {e}"]

    # Validate AST for disallowed patterns
    ast_errors = validate_ast(tree, '<lambda>')
    errors.extend(ast_errors)

    return source, errors


# Re-export for convenient access
__all__ = [
    'remote', 'RemoteValidationError', 'is_json_serializable', 'ALLOWED_MODULES',
    'extract_lambda_source', 'LambdaExtractionError', 'is_lambda', 'validate_lambda_for_remote',
]
