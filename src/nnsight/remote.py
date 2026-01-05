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
import os
import sys
import types
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

# =============================================================================
# Server-available modules (single source of truth)
# =============================================================================
# This section defines which Python modules are available on NDIF servers.
# These modules don't need their source serialized because they're pre-installed.
#
# When validating @remote code, references to these modules are marked as "skip"
# (available on server) rather than "capture" (needs serialization).
#
# When reconstructing code server-side, imports from these modules are allowed
# and resolved using the server's installed packages.
#
# IMPORTANT: This is the single source of truth for allowed modules. The same
# constants are used by:
# - @remote decorator validation (remote.py)
# - Source serialization (serialization_source.py)
# - Restricted execution (restricted_execution.py)

# Python standard library modules (safe subset)
STDLIB_MODULES = {
    # Core builtins and typing
    'builtins', 'abc', 'typing', 'types',
    # Collections and functional
    'collections', 'functools', 'itertools', 'operator',
    # Data structures and serialization
    'enum', 'dataclasses', 'copy', 'pickle', 'json',
    # Math and numbers
    'math', 'numbers', 'random',
    # String processing
    're', 'string', 'textwrap',
    # Utilities
    'warnings', 'contextlib', 'weakref', 'inspect',
    # I/O (restricted on server - no actual file operations)
    'io', 'os', 'sys', 'pathlib',
}

# Machine learning libraries available on server
ML_LIBRARY_MODULES = {
    # Core ML
    'torch', 'numpy', 'scipy', 'sklearn',
    # Hugging Face ecosystem
    'transformers', 'huggingface_hub', 'tokenizers', 'safetensors', 'accelerate',
    # Other common libraries
    'einops',
}

# Combined set of all server-available modules
SERVER_AVAILABLE_MODULES = STDLIB_MODULES | ML_LIBRARY_MODULES

# Backwards compatibility alias
ALLOWED_MODULES = SERVER_AVAILABLE_MODULES


def is_server_available_module(module_name: str) -> bool:
    """Check if a module is available on NDIF servers.

    Args:
        module_name: Full module name (e.g., 'torch.nn.functional')

    Returns:
        True if the module's root package is available on server
    """
    if not module_name:
        return False
    root = module_name.split('.')[0]
    return root in SERVER_AVAILABLE_MODULES or 'nnsight' in module_name

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

    # If already validated (e.g., during deserialization round-trip), just return
    # For classes, we must check __dict__ directly to avoid inheriting from parent class.
    # A child class should be validated separately even if its parent was already validated.
    if isinstance(obj, type):
        if obj.__dict__.get('_remote_validated', False):
            return obj
    elif getattr(obj, '_remote_validated', False):
        return obj

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

    # Step 1: Verify source is available and get file/line info for error messages
    try:
        source = inspect.getsource(obj)
        # Use relative path for cleaner error messages (matches Python traceback style)
        source_file = os.path.relpath(inspect.getfile(obj))
        _, start_line = inspect.getsourcelines(obj)
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
    module_refs, resolution_errors = resolve_module_references(external_names, obj, source_file, start_line)

    # Step 5: Extract and validate closure variables (for nested functions)
    closure_vars, closure_errors = extract_closure_variables(obj, source_file, start_line)
    module_refs.update(closure_vars)  # Merge closure vars into module refs

    # Step 6: Validate AST for disallowed patterns
    ast_errors = validate_ast(tree, obj.__name__, source_file, start_line)

    # Step 7: For classes, additional validation
    class_errors = []
    if isinstance(obj, type):
        class_errors = validate_class(obj, source_file, start_line)

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
            # Visit decorators BEFORE defining the function (they're in outer scope)
            for decorator in node.decorator_list:
                self.visit(decorator)

            # Function name is defined in outer scope
            self.define(node.name)

            # Visit default argument values (in outer scope, before parameters)
            for default in node.args.defaults:
                self.visit(default)
            for default in node.args.kw_defaults:
                if default is not None:
                    self.visit(default)

            # Function parameters are defined in inner scope
            self.push_scope()
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                self.define(arg.arg)
                # Visit argument annotations
                if arg.annotation:
                    self.visit(arg.annotation)
            if node.args.vararg:
                self.define(node.args.vararg.arg)
                if node.args.vararg.annotation:
                    self.visit(node.args.vararg.annotation)
            if node.args.kwarg:
                self.define(node.args.kwarg.arg)
                if node.args.kwarg.annotation:
                    self.visit(node.args.kwarg.annotation)

            # Visit return annotation
            if node.returns:
                self.visit(node.returns)

            # Visit function body
            for child in node.body:
                self.visit(child)

            self.pop_scope()

        def visit_AsyncFunctionDef(self, node):
            # Same as FunctionDef
            self.visit_FunctionDef(node)

        def visit_ClassDef(self, node):
            # Visit decorators BEFORE defining the class (they're in outer scope)
            for decorator in node.decorator_list:
                self.visit(decorator)

            # Visit base classes and keywords (in outer scope)
            for base in node.bases:
                self.visit(base)
            for keyword in node.keywords:
                self.visit(keyword.value)

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

        def visit_Lambda(self, node):
            # Lambda parameters are in their own scope
            self.push_scope()
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                self.define(arg.arg)
            if node.args.vararg:
                self.define(node.args.vararg.arg)
            if node.args.kwarg:
                self.define(node.args.kwarg.arg)
            # Visit lambda body
            self.visit(node.body)
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
    # Note: We don't filter builtins here because they might be overridden
    # at module level. resolve_module_references will check if the actual
    # value differs from the builtin.
    external = collector.referenced

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
        if is_server_available_module(value.__name__):
            return ValueClassification.SKIP, None, None
        else:
            return ValueClassification.ERROR, None, f"module '{value.__name__}' not available on NDIF server"

    # Case 2: @remote-decorated function/class (will be serialized separately)
    if getattr(value, '_remote_validated', False):
        return ValueClassification.SKIP, None, None

    # Case 2b: The @remote decorator itself or other nnsight internals
    if callable(value) and hasattr(value, '__module__'):
        if value.__module__ and 'nnsight' in value.__module__:
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
        if is_server_available_module(value.__module__ or ''):
            return ValueClassification.SKIP, None, None

    # Case 6: Non-serializable value
    type_name = type(value).__name__
    return ValueClassification.ERROR, None, f"type '{type_name}' is not JSON-serializable"


def resolve_module_references(names: Set[str], obj: Union[Type, Callable], source_file: str = None, start_line: int = None) -> Tuple[Dict[str, Any], List[str]]:
    """
    Resolve external names to their values from the function/class globals.

    Args:
        names: Set of external names to resolve
        obj: The function or class being decorated
        source_file: The source file path (for error messages)
        start_line: The starting line number in the source file (for error messages)

    Returns:
        (captured_refs, errors) where:
        - captured_refs: dict of JSON-serializable module-level constants
        - errors: list of error messages for non-serializable references
    """
    captured = {}
    errors = []

    # Format location prefix for error messages
    if source_file and start_line is not None:
        location = f"{source_file}:{start_line}"
    elif source_file:
        location = source_file
    elif start_line is not None:
        location = f"Line {start_line}"
    else:
        location = None

    # Get globals from the decorated object
    if hasattr(obj, '__globals__'):
        module_globals = obj.__globals__
    else:
        # For classes, get the module globals
        module = sys.modules.get(obj.__module__)
        module_globals = getattr(module, '__dict__', {}) if module else {}

    for name in names:
        # Get the actual value from globals
        if name not in module_globals:
            # Name not in globals - skip if it's a builtin (will use default)
            if name in BUILTIN_NAMES:
                continue
            # Could be a nested scope variable or truly undefined
            # We'll let Python's runtime handle truly undefined names
            continue

        value = module_globals[name]

        # Skip if it's a builtin AND the value is the same as the builtin
        # (if overridden with a different value, we need to capture it)
        if name in BUILTIN_NAMES:
            builtin_value = getattr(builtins, name, None)
            if value is builtin_value:
                continue

        classification, captured_value, error_detail = classify_reference_value(name, value)

        if classification == ValueClassification.CAPTURE:
            captured[name] = captured_value
        elif classification == ValueClassification.ERROR:
            prefix = f"{location}: " if location else ""
            errors.append(
                f"{prefix}reference '{name}' ({error_detail}).\n"
                f"  Options:\n"
                f"    - Make it a class/instance attribute instead\n"
                f"    - Pass it as a function/method argument\n"
                f"    - Use a JSON-serializable type (int, float, str, list, dict, bool, None)"
            )

    return captured, errors


def extract_closure_variables(obj: Union[Type, Callable], source_file: str = None, start_line: int = None) -> Tuple[Dict[str, Any], List[str]]:
    """
    Extract and validate closure variables from a function or class methods.

    When a function is defined inside another function and captures variables
    from the enclosing scope, those variables are stored in __closure__ and
    their names in __code__.co_freevars.

    For classes, we check all methods for closures since methods like
    __init_subclass__ may capture variables from the defining scope.

    Args:
        obj: The function or class to extract closure variables from
        source_file: The source file path (for error messages)
        start_line: The starting line number in the source file (for error messages)

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
                method_captured, method_errors = _extract_closure_from_function(func, attr_name, source_file, start_line)
                captured.update(method_captured)
                errors.extend(method_errors)

        return captured, errors

    # For functions, extract directly
    return _extract_closure_from_function(obj, obj.__name__ if hasattr(obj, '__name__') else '<function>', source_file, start_line)


def _extract_closure_from_function(func: Callable, context_name: str, source_file: str = None, start_line: int = None) -> Tuple[Dict[str, Any], List[str]]:
    """
    Extract closure variables from a single function.

    Args:
        func: The function to extract closure variables from
        context_name: Name of the function/method for error messages
        source_file: The source file path (for error messages)
        start_line: The starting line number in the source file (for error messages)

    Returns:
        (captured_vars, errors)
    """
    captured = {}
    errors = []

    # Format location prefix for error messages
    if source_file and start_line is not None:
        location = f"{source_file}:{start_line}"
    elif source_file:
        location = source_file
    elif start_line is not None:
        location = f"Line {start_line}"
    else:
        location = None

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

        # Skip if it's a builtin AND the value is the same as the builtin
        # (if overridden with a different value, we need to capture it)
        if name in BUILTIN_NAMES:
            builtin_value = getattr(builtins, name, None)
            if value is builtin_value:
                continue

        # Skip __class__ (implicit closure from super() calls)
        if name == '__class__':
            continue

        # Use shared classification logic
        classification, captured_value, error_detail = classify_reference_value(name, value)

        if classification == ValueClassification.CAPTURE:
            captured[name] = captured_value
        elif classification == ValueClassification.ERROR:
            prefix = f"{location}: " if location else ""
            errors.append(
                f"{prefix}in '{context_name}': closure variable '{name}' ({error_detail}).\n"
                f"  Options:\n"
                f"    - Pass it as a function argument instead\n"
                f"    - Use a JSON-serializable type (int, float, str, list, dict, bool, None)\n"
                f"    - Mark it with @nnsight.remote if it's a custom function/class"
            )

    return captured, errors


def is_json_serializable(value: Any, _seen: set = None) -> bool:
    """
    Check if a value can be JSON-serialized.

    Allows: None, bool, int, float, str, list/tuple (of serializable), dict (str keys, serializable values)

    Handles circular references by tracking seen object ids.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return True

    # Track seen objects to handle circular references
    if _seen is None:
        _seen = set()

    obj_id = id(value)
    if obj_id in _seen:
        # Circular reference - not JSON serializable (would cause infinite recursion)
        return False
    _seen.add(obj_id)

    if isinstance(value, (list, tuple)):
        return all(is_json_serializable(item, _seen) for item in value)

    if isinstance(value, dict):
        return all(
            isinstance(k, str) and is_json_serializable(v, _seen)
            for k, v in value.items()
        )

    # Try actual JSON serialization as final check
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError, RecursionError):
        return False


def validate_ast(tree: ast.AST, name: str, source_file: str = None, start_line: int = None) -> List[str]:
    """
    Validate AST for disallowed patterns that would disqualify the code from
    being remoted.

    This is used by the @remote decorator at import time to catch unsafe code
    early, and by auto_discover functions at serialization time to validate
    third-party classes.

    Checks for:
    - Imports of non-allowed modules (not available on NDIF server)
    - Relative imports (from . import X) which can't work on server
    - Calls to disallowed functions (open, exec, eval, compile, input, __import__)
    - Calls to disallowed attribute chains (os.system, subprocess.run, etc.)

    Args:
        tree: The parsed AST of the source code to validate
        name: The name of the function/class being validated (for error messages)
        source_file: The source file path (for error messages)
        start_line: The starting line number in the source file (for error messages)

    Returns:
        A list of error strings describing each validation failure. An empty list
        means the code passed validation and is safe for remote execution.
    """
    errors = []

    def format_location(lineno: int) -> str:
        """Format file:line prefix for error messages."""
        if start_line is not None:
            actual_line = start_line + lineno - 1
        else:
            actual_line = lineno
        if source_file:
            return f"{source_file}:{actual_line}"
        return f"Line {actual_line}"

    class Validator(ast.NodeVisitor):
        def visit_Import(self, node):
            for alias in node.names:
                if not is_server_available_module(alias.name):
                    errors.append(f"{format_location(node.lineno)}: imports '{alias.name}' (not available on NDIF server)")
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            # Relative imports (from . import X, from ..foo import Y) can't work on server
            if node.level > 0:
                dots = '.' * node.level
                module_part = node.module or ''
                errors.append(f"{format_location(node.lineno)}: relative import 'from {dots}{module_part}' not supported in @nnsight.remote code")
            elif not is_server_available_module(node.module or ''):
                errors.append(f"{format_location(node.lineno)}: imports from '{node.module}' (not available on NDIF server)")
            self.generic_visit(node)

        def visit_Call(self, node):
            # Check for disallowed function calls
            if isinstance(node.func, ast.Name):
                if node.func.id in DISALLOWED_CALLS:
                    errors.append(f"{format_location(node.lineno)}: calls '{node.func.id}()' (not allowed in @nnsight.remote code)")

            # Check for disallowed attribute calls like os.system()
            if isinstance(node.func, ast.Attribute):
                chain = self._get_attr_chain(node.func)
                for pattern in DISALLOWED_ATTR_PATTERNS:
                    if len(chain) >= len(pattern):
                        if chain[:len(pattern)] == pattern:
                            errors.append(f"{format_location(node.lineno)}: calls '{'.'.join(chain)}()' (not allowed in @nnsight.remote code)")
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


def validate_class(cls: type, source_file: str = None, start_line: int = None) -> List[str]:
    """
    Additional validation for classes.

    Checks:
    - Base classes are @nnsight.remote, object, or in ALLOWED_BASE_CLASSES
    - No metaclass (except for allowed base classes like nn.Module)

    Args:
        cls: The class to validate
        source_file: The source file path (for error messages)
        start_line: The starting line number in the source file (for error messages)
    """
    errors = []

    # Format location prefix for error messages
    if source_file and start_line is not None:
        location = f"{source_file}:{start_line}"
    elif source_file:
        location = source_file
    elif start_line is not None:
        location = f"Line {start_line}"
    else:
        location = None

    # Check base classes
    prefix = f"{location}: " if location else ""
    for base in cls.__bases__:
        if base is object:
            continue
        # Check if base is in allowed base classes
        base_fullname = f"{base.__module__}.{base.__name__}"
        if base_fullname in ALLOWED_BASE_CLASSES:
            continue
        if not getattr(base, '_remote_validated', False):
            errors.append(
                f"{prefix}base class '{base.__name__}' is not @nnsight.remote decorated. "
                f"All base classes must be @nnsight.remote or object."
            )

    # Check for metaclass
    if type(cls) is not type:
        errors.append(
            f"{prefix}uses metaclass '{type(cls).__name__}'. "
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
# Functions for extracting the source code of lambda functions for serialization.
#
# Lambdas are challenging because:
# 1. Multiple lambdas can appear on the same source line
# 2. inspect.getsource() returns the whole line, not just the lambda
# 3. Lambdas defined in interactive sessions (REPL, notebooks) may not have source
#
# The extraction strategy:
# 1. Get the full source line with inspect.getsource()
# 2. Parse the line to find all lambda AST nodes
# 3. If multiple lambdas exist, disambiguate using bytecode matching
# 4. Return just the lambda expression text (e.g., "lambda x: x + 1")
#
# For complex lambdas or those with non-serializable closures, users should
# convert to @remote decorated named functions.

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


def is_remote_object(obj: Any) -> bool:
    """Check if obj is a @nnsight.remote function/class or instance thereof."""
    # Check if it's a decorated function or class
    if callable(obj) and getattr(obj, '_remote_validated', False):
        return True

    # Check if it's an instance of a decorated class
    if getattr(type(obj), '_remote_validated', False):
        return True

    return False


def remote_noop(obj: Union[Type, Callable] = None, *, version: str = None, library: str = None) -> Union[Type, Callable]:
    """
    No-op version of @remote decorator for use during deserialization.

    When code is deserialized and exec'd on the server, any @remote decorators
    in that code would normally try to re-validate and extract source. But:
    1. The code was already validated client-side
    2. Source extraction won't work on exec'd code

    This no-op decorator is used in the deserialization namespace to skip
    re-validation. It just marks the object as validated without any checks.

    IMPORTANT: This should ONLY be used in the deserialization namespace.
    Regular users should use @remote which performs full validation.
    """
    def apply_noop(obj):
        obj._remote_validated = True
        obj._remote_source = None  # Source already transmitted
        obj._remote_module_refs = {}
        obj._remote_closure_vars = {}
        obj._remote_library = library
        obj._remote_version = version
        return obj

    # Support both @remote_noop and @remote_noop(version="...", library="...")
    if obj is None:
        return apply_noop
    else:
        return apply_noop(obj)


# Re-export for convenient access
__all__ = [
    'remote', 'remote_noop', 'RemoteValidationError', 'is_json_serializable',
    # Module constants (single source of truth for server-available modules)
    'STDLIB_MODULES', 'ML_LIBRARY_MODULES', 'SERVER_AVAILABLE_MODULES',
    'ALLOWED_MODULES',  # Backwards compatibility alias
    'is_server_available_module',
    # Remote object utilities
    'is_remote_object',
    # Lambda utilities
    'extract_lambda_source', 'LambdaExtractionError', 'is_lambda', 'validate_lambda_for_remote',
]
