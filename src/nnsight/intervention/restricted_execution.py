"""
Restricted execution environment for NDIF server-side code execution.

This module uses RestrictedPython to add runtime guards that:
1. Block known sandbox escape routes (dunder attributes, dangerous functions)
2. Log suspicious activity for security auditing
3. Provide defense-in-depth alongside OS-level sandboxing

Usage:
    from nnsight.intervention.restricted_execution import execute_restricted

    namespace = execute_restricted(
        code="result = model.forward(x)",
        globals_dict={'model': model, 'x': input_tensor},
        user_id="researcher_123",
        job_id="job_456",
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Set

# RestrictedPython imports
try:
    from RestrictedPython import compile_restricted
    from RestrictedPython.Guards import (
        guarded_iter_unpack_sequence,
        safer_getattr,
        guarded_unpack_sequence,
    )
    from RestrictedPython.Eval import default_guarded_getitem
    from RestrictedPython import safe_builtins
    RESTRICTED_PYTHON_AVAILABLE = True
except ImportError:
    RESTRICTED_PYTHON_AVAILABLE = False
    compile_restricted = None
    safe_builtins = None


# =============================================================================
# Security Audit Logger
# =============================================================================

# Configure audit logger - server should route this to security monitoring
audit_logger = logging.getLogger("nnsight.security.audit")


class SecurityAuditError(Exception):
    """Raised when suspicious activity is detected and blocked."""
    pass


# =============================================================================
# Suspicious Patterns
# =============================================================================

# Attributes that indicate sandbox escape attempts
SUSPICIOUS_ATTRS: Set[str] = {
    # Process execution
    'system', 'popen', 'spawn', 'spawnl', 'spawnle', 'spawnlp', 'spawnlpe',
    'spawnv', 'spawnve', 'spawnvp', 'spawnvpe',
    'fork', 'forkpty', 'kill', 'killpg',
    'execl', 'execle', 'execlp', 'execlpe', 'execv', 'execve', 'execvp', 'execvpe',

    # Frame inspection (can leak variables, escape scope)
    'f_back', 'f_locals', 'f_globals', 'f_code', 'f_builtins',
    'f_lineno', 'f_lasti', 'f_trace',

    # Generator/coroutine frame access
    'gi_frame', 'gi_code', 'gi_yieldfrom',
    'cr_frame', 'cr_code', 'cr_origin', 'cr_await',
    'ag_frame', 'ag_code', 'ag_await',

    # Code object manipulation (can modify function behavior)
    '__code__', '__globals__', '__builtins__', '__closure__',
    '__self__', '__func__', '__kwdefaults__',

    # File operations
    'read', 'write', 'open', 'close', 'seek', 'tell', 'truncate',
    'readline', 'readlines', 'writelines',
}

# Modules that should never be imported
BLOCKED_MODULES: Set[str] = {
    'subprocess', 'socket', 'shutil', 'ctypes', 'multiprocessing',
    'threading', 'concurrent', 'asyncio',
    'http', 'urllib', 'requests', 'ftplib', 'smtplib', 'telnetlib',
    'pickle', 'shelve', 'marshal', 'importlib',
    'code', 'codeop', 'pty', 'tty', 'termios',
    'resource', 'syslog', 'grp', 'pwd', 'spwd', 'crypt',
    'tempfile', 'glob', 'fnmatch', 'linecache', 'tokenize',
}

# Functions that should be blocked even if available
BLOCKED_FUNCTIONS: Set[str] = {
    'eval', 'exec', 'compile', '__import__',
    'breakpoint', 'input',
}


# =============================================================================
# Guard Functions with Audit Logging
# =============================================================================

def create_guarded_getattr(user_id: str, job_id: str, audit: bool = True) -> Callable:
    """
    Create a guarded getattr that logs suspicious access.

    This wraps RestrictedPython's safer_getattr to add audit logging
    for security-relevant attribute access attempts.

    Args:
        user_id: ID of the user running the code
        job_id: ID of the specific job/trace
        audit: If True (server-side), log attempts. If False (local validation), skip logging.

    Returns:
        A _getattr_ function for RestrictedPython
    """
    def _getattr_(obj: Any, name: str, default: Any = None) -> Any:
        # Check for suspicious attribute access
        if name in SUSPICIOUS_ATTRS:
            if audit:
                audit_logger.warning(
                    f"SUSPICIOUS_GETATTR | user={user_id} | job={job_id} | "
                    f"obj_type={type(obj).__name__} | attr={name}"
                )
            raise SecurityAuditError(
                f"Access to '{name}' is not allowed in remote code."
            )

        # Delegate to RestrictedPython's safer_getattr
        if RESTRICTED_PYTHON_AVAILABLE:
            return safer_getattr(obj, name, default)
        else:
            return getattr(obj, name, default)

    return _getattr_


def create_guarded_getitem(user_id: str, job_id: str, audit: bool = True) -> Callable:
    """
    Create a guarded getitem that logs suspicious key access.

    Args:
        user_id: ID of the user running the code
        job_id: ID of the specific job/trace
        audit: If True (server-side), log attempts. If False (local validation), skip logging.

    Returns:
        A _getitem_ function for RestrictedPython
    """
    def _getitem_(obj: Any, key: Any) -> Any:
        # Check for suspicious string keys that look like attribute access
        if isinstance(key, str) and key in SUSPICIOUS_ATTRS:
            if audit:
                audit_logger.warning(
                    f"SUSPICIOUS_GETITEM | user={user_id} | job={job_id} | "
                    f"obj_type={type(obj).__name__} | key={key}"
                )
            raise SecurityAuditError(
                f"Access to '{key}' via indexing is not allowed in remote code."
            )

        # Delegate to RestrictedPython's default guard
        if RESTRICTED_PYTHON_AVAILABLE:
            return default_guarded_getitem(obj, key)
        else:
            return obj[key]

    return _getitem_


def create_guarded_import(
    user_id: str,
    job_id: str,
    allowed_modules: Set[str],
    audit: bool = True
) -> Callable:
    """
    Create a guarded import that only allows specific modules.

    Args:
        user_id: ID of the user running the code
        job_id: ID of the specific job/trace
        allowed_modules: Set of module names that are allowed
        audit: If True (server-side), log attempts. If False (local validation), skip logging.

    Returns:
        An __import__ function for the restricted namespace
    """
    def _import_(
        name: str,
        globals: Optional[Dict] = None,
        locals: Optional[Dict] = None,
        fromlist: tuple = (),
        level: int = 0
    ) -> Any:
        root_module = name.split('.')[0]

        if root_module in BLOCKED_MODULES:
            if audit:
                audit_logger.warning(
                    f"BLOCKED_IMPORT | user={user_id} | job={job_id} | "
                    f"module={name}"
                )
            raise SecurityAuditError(
                f"Import of '{name}' is not allowed in remote code."
            )

        if root_module not in allowed_modules:
            if audit:
                audit_logger.warning(
                    f"UNAUTHORIZED_IMPORT | user={user_id} | job={job_id} | "
                    f"module={name}"
                )
            raise SecurityAuditError(
                f"Import of '{name}' is not in the allowed list for remote code."
            )

        return __import__(name, globals, locals, fromlist, level)

    return _import_


def create_guarded_builtins(user_id: str, job_id: str, audit: bool = True) -> Callable:
    """
    Create wrapped versions of potentially dangerous builtins.

    Args:
        user_id: ID of the user running the code
        job_id: ID of the specific job/trace
        audit: If True (server-side), log attempts. If False (local validation), skip logging.

    Returns:
        A function that wraps a builtin with logging
    """
    def wrap_builtin(name: str, func: Callable) -> Callable:
        def logged_func(*args, **kwargs):
            if audit:
                audit_logger.warning(
                    f"SUSPICIOUS_CALL | user={user_id} | job={job_id} | "
                    f"func={name}"
                )
            raise SecurityAuditError(
                f"Call to '{name}()' is not allowed in remote code."
            )
        return logged_func

    return wrap_builtin


# =============================================================================
# Restricted Namespace Builder
# =============================================================================

# Default allowed modules for NDIF
DEFAULT_ALLOWED_MODULES: Set[str] = {
    'torch', 'numpy', 'math', 'random', 'collections', 'functools',
    'itertools', 'operator', 'typing', 'dataclasses', 'enum',
    'copy', 'json', 're', 'string', 'textwrap',
}


def create_restricted_builtins(user_id: str, job_id: str, audit: bool = True) -> Dict[str, Any]:
    """
    Create a restricted builtins dict with security guards.

    Args:
        user_id: ID of the user running the code
        job_id: ID of the specific job/trace
        audit: If True (server-side), log violations to security audit log

    Returns:
        Dict suitable for use as __builtins__ in exec()
    """
    if RESTRICTED_PYTHON_AVAILABLE:
        builtins = dict(safe_builtins)
    else:
        # Fallback: start with empty and add safe ones
        builtins = {}

    # Add our guarded getattr
    guarded_getattr = create_guarded_getattr(user_id, job_id, audit=audit)
    builtins['getattr'] = guarded_getattr

    # Block dangerous functions with logging
    wrap_builtin = create_guarded_builtins(user_id, job_id, audit=audit)
    for func_name in BLOCKED_FUNCTIONS:
        if func_name in builtins:
            builtins[func_name] = wrap_builtin(func_name, builtins[func_name])

    # Ensure safe builtins are present
    safe_additions = {
        'True': True, 'False': False, 'None': None,
        'int': int, 'float': float, 'str': str, 'bool': bool,
        'bytes': bytes, 'bytearray': bytearray,
        'list': list, 'dict': dict, 'tuple': tuple, 'set': set, 'frozenset': frozenset,
        'len': len, 'range': range, 'enumerate': enumerate, 'zip': zip,
        'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
        'pow': pow, 'divmod': divmod,
        'sorted': sorted, 'reversed': reversed,
        'map': map, 'filter': filter, 'all': all, 'any': any,
        'isinstance': isinstance, 'issubclass': issubclass,
        'hasattr': hasattr, 'setattr': setattr, 'delattr': delattr,
        'callable': callable, 'type': type,
        'repr': repr, 'ascii': ascii, 'chr': chr, 'ord': ord,
        'hex': hex, 'oct': oct, 'bin': bin,
        'format': format, 'hash': hash, 'id': id,
        'slice': slice, 'object': object,
        'staticmethod': staticmethod, 'classmethod': classmethod, 'property': property,
        'super': super,
        'Exception': Exception, 'BaseException': BaseException,
        'ValueError': ValueError, 'TypeError': TypeError, 'KeyError': KeyError,
        'IndexError': IndexError, 'AttributeError': AttributeError,
        'RuntimeError': RuntimeError, 'StopIteration': StopIteration,
        'AssertionError': AssertionError, 'NotImplementedError': NotImplementedError,
        'ZeroDivisionError': ZeroDivisionError, 'OverflowError': OverflowError,
        'MemoryError': MemoryError, 'RecursionError': RecursionError,
        'print': print,  # Allow print for debugging (could redirect to logger)
    }

    for name, value in safe_additions.items():
        if name not in builtins:
            builtins[name] = value

    return builtins


def create_restricted_globals(
    user_id: str,
    job_id: str,
    base_globals: Optional[Dict[str, Any]] = None,
    allowed_modules: Optional[Set[str]] = None,
    audit: bool = True,
) -> Dict[str, Any]:
    """
    Create a restricted globals dict for executing user code.

    Args:
        user_id: ID of the user running the code
        job_id: ID of the specific job/trace
        base_globals: Additional globals to include (e.g., model, torch)
        allowed_modules: Set of module names allowed for import
        audit: If True (server-side), log violations to security audit log

    Returns:
        Dict suitable for use as globals in exec()
    """
    if allowed_modules is None:
        allowed_modules = DEFAULT_ALLOWED_MODULES

    globals_dict: Dict[str, Any] = {}

    # Add RestrictedPython guards
    guarded_getattr = create_guarded_getattr(user_id, job_id, audit=audit)
    guarded_getitem = create_guarded_getitem(user_id, job_id, audit=audit)
    guarded_import = create_guarded_import(user_id, job_id, allowed_modules, audit=audit)

    globals_dict['_getattr_'] = guarded_getattr
    globals_dict['_getitem_'] = guarded_getitem
    globals_dict['__import__'] = guarded_import

    # Add iteration guards (required by RestrictedPython)
    globals_dict['_getiter_'] = iter
    if RESTRICTED_PYTHON_AVAILABLE:
        globals_dict['_iter_unpack_sequence_'] = guarded_iter_unpack_sequence
        globals_dict['_unpack_sequence_'] = guarded_unpack_sequence
    else:
        globals_dict['_iter_unpack_sequence_'] = lambda x, y: x
        globals_dict['_unpack_sequence_'] = lambda x, y: x

    # Add write guard (for augmented assignment)
    globals_dict['_write_'] = lambda x: x

    # Add restricted builtins
    restricted_builtins = create_restricted_builtins(user_id, job_id, audit=audit)
    # Add guarded import to builtins (required for import statements)
    restricted_builtins['__import__'] = guarded_import
    globals_dict['__builtins__'] = restricted_builtins

    # Add base globals (model, pre-imported modules, etc.)
    if base_globals:
        for key, value in base_globals.items():
            if key != '__builtins__':  # Don't override our restricted builtins
                globals_dict[key] = value

    return globals_dict


# =============================================================================
# Static Import Analysis
# =============================================================================

import ast


def check_imports_in_code(
    code: str,
    allowed_modules: Set[str],
    user_id: str = "unknown",
    job_id: str = "unknown",
    audit: bool = False,
) -> None:
    """
    Statically analyze code for blocked import statements.

    Scans the AST for import and from...import statements and raises
    SecurityAuditError if any blocked modules are found.

    Args:
        code: Source code to analyze
        allowed_modules: Set of module names that are allowed
        user_id: ID of the user (for audit logging)
        job_id: ID of the job (for audit logging)
        audit: If True (server-side), log attempts to security audit log

    Raises:
        SecurityAuditError: If blocked imports are found
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Let the actual compiler report syntax errors
        return

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split('.')[0]
                if module_name in BLOCKED_MODULES:
                    if audit:
                        audit_logger.warning(
                            f"STATIC_BLOCKED_IMPORT | user={user_id} | job={job_id} | "
                            f"module={alias.name}"
                        )
                    raise SecurityAuditError(
                        f"Import of '{alias.name}' is not allowed in remote code."
                    )
                if module_name not in allowed_modules:
                    if audit:
                        audit_logger.warning(
                            f"STATIC_UNAUTHORIZED_IMPORT | user={user_id} | job={job_id} | "
                            f"module={alias.name}"
                        )
                    raise SecurityAuditError(
                        f"Import of '{alias.name}' is not in the allowed list for remote code."
                    )

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module.split('.')[0]
                if module_name in BLOCKED_MODULES:
                    if audit:
                        audit_logger.warning(
                            f"STATIC_BLOCKED_IMPORT | user={user_id} | job={job_id} | "
                            f"module={node.module}"
                        )
                    raise SecurityAuditError(
                        f"Import from '{node.module}' is not allowed in remote code."
                    )
                if module_name not in allowed_modules:
                    if audit:
                        audit_logger.warning(
                            f"STATIC_UNAUTHORIZED_IMPORT | user={user_id} | job={job_id} | "
                            f"module={node.module}"
                        )
                    raise SecurityAuditError(
                        f"Import from '{node.module}' is not in the allowed list for remote code."
                    )


# =============================================================================
# Main Execution Function
# =============================================================================

def compile_user_code(
    code: str,
    filename: str = "<user_code>",
    mode: str = "exec",
    allowed_modules: Optional[Set[str]] = None,
    user_id: str = "unknown",
    job_id: str = "unknown",
    audit: bool = False,
) -> Any:
    """
    Compile user code with RestrictedPython and static import analysis.

    This performs compile-time checks that block:
    - Dunder attribute access (__class__, __bases__, etc.)
    - Blocked/unauthorized import statements
    - Certain dangerous patterns

    Args:
        code: Source code to compile
        filename: Filename for error messages
        mode: Compilation mode ('exec', 'eval', 'single')
        allowed_modules: Set of module names allowed for import
        user_id: ID of the user (for audit logging)
        job_id: ID of the job (for audit logging)
        audit: If True (server-side), log violations to security audit log

    Returns:
        Compiled code object

    Raises:
        SyntaxError: If code contains blocked patterns
        SecurityAuditError: If code contains blocked imports
    """
    if allowed_modules is None:
        allowed_modules = DEFAULT_ALLOWED_MODULES

    # Static import analysis
    check_imports_in_code(code, allowed_modules, user_id, job_id, audit=audit)

    if RESTRICTED_PYTHON_AVAILABLE:
        return compile_restricted(code, filename, mode)
    else:
        # Fallback to regular compile (less safe)
        audit_logger.warning(
            f"FALLBACK_COMPILE | RestrictedPython not available, using regular compile"
        )
        return compile(code, filename, mode)


def execute_restricted(
    code: str,
    globals_dict: Dict[str, Any],
    user_id: str,
    job_id: str,
    filename: str = "<user_code>",
    allowed_modules: Optional[Set[str]] = None,
    audit: bool = True,
) -> Dict[str, Any]:
    """
    Execute user code in a restricted environment with audit logging.

    This is the main entry point for server-side code execution.

    Args:
        code: Source code to execute
        globals_dict: Base globals (model, torch, numpy, etc.)
        user_id: ID of the user running the code
        job_id: ID of the specific job/trace
        filename: Filename for error messages
        allowed_modules: Set of module names allowed for import
        audit: If True (server-side), log violations to security audit log

    Returns:
        The namespace after execution (contains results)

    Raises:
        SyntaxError: If code contains compile-time blocked patterns
        SecurityAuditError: If code attempts blocked operations (static or runtime)
    """
    if allowed_modules is None:
        allowed_modules = DEFAULT_ALLOWED_MODULES

    # Compile with restrictions and static import analysis
    byte_code = compile_user_code(
        code, filename,
        allowed_modules=allowed_modules,
        user_id=user_id,
        job_id=job_id,
        audit=audit,
    )

    # Create restricted namespace
    namespace = create_restricted_globals(
        user_id=user_id,
        job_id=job_id,
        base_globals=globals_dict,
        allowed_modules=allowed_modules,
        audit=audit,
    )

    # Execute
    exec(byte_code, namespace)

    return namespace


# =============================================================================
# Utility Functions
# =============================================================================

def is_restricted_python_available() -> bool:
    """Check if RestrictedPython is installed."""
    return RESTRICTED_PYTHON_AVAILABLE


def get_audit_logger() -> logging.Logger:
    """Get the security audit logger for configuration."""
    return audit_logger
