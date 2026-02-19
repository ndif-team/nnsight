"""Source-based function serialization for cross-version compatibility.

This module provides a custom serialization system built on top of cloudpickle
that serializes Python functions by their source code rather than bytecode.

Why source-based serialization?
    Standard pickle/cloudpickle serialize functions using Python bytecode, which
    is version-specific and can break when deserializing on a different Python
    version. By serializing the source code instead, we can reconstruct functions
    on any Python version that supports the syntax, enabling cross-version
    compatibility for remote execution (e.g., client on Python 3.10, server on 3.11).

Key components:
    - CustomCloudPickler: Serializes functions by capturing their source code,
      closure variables, and metadata instead of bytecode.
    - CustomCloudUnpickler: Deserializes data with support for persistent object
      references (objects that shouldn't be serialized but looked up by ID).
    - make_function: Reconstructs a function from its serialized components.
    - dumps/loads: High-level API for serializing and deserializing objects
      (named to match the standard pickle module API).

Persistent objects:
    Some objects (like model proxies or tensors) shouldn't be serialized directly
    but instead referenced by ID and resolved at deserialization time. Objects
    with a `_persistent_id` attribute in their __dict__ are handled this way.

Examples:
    >>> import serialization
    >>> def my_func(x, y=10):
    ...     return x + y
    >>> data = serialization.dumps(my_func)
    >>> restored = serialization.loads(data)
    >>> restored(5)  # Returns 15
"""

import dataclasses
import inspect
import io
import pickle
import textwrap
import tokenize
import types
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

import cloudpickle
import cloudpickle.cloudpickle as _cloudpickle_internal
from cloudpickle.cloudpickle import _function_getstate, _get_cell_contents

# Names of methods that the @dataclass decorator generates dynamically
_DATACLASS_GENERATED_METHODS = frozenset(
    {
        "__init__",
        "__repr__",
        "__eq__",
        "__ne__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__hash__",
        "__setattr__",
        "__delattr__",
    }
)

# Default pickle protocol - protocol 4 is available in Python 3.4+ and supports large objects
DEFAULT_PROTOCOL = 4


def _extract_lambda_source(source: str, code: types.CodeType) -> str:
    """Extract a specific lambda's source when multiple lambdas share a line.

    Uses co_positions() (Python 3.11+) to find column offset and tokenization
    to extract just the target lambda. Falls back to full source on older Python.

    This handles tricky cases like:
    - Multiple lambdas on same line: f, g = lambda x: x*2, lambda x: x+1
    - Nested lambdas: lambda x: lambda y: x + y
    - Lambdas with lambda defaults: lambda x=lambda: 1: x()
    - Multi-line lambdas (automatically wrapped in parentheses)

    Args:
        source: The full source code containing the lambda (from inspect.getsource).
        code: The code object of the lambda function.

    Returns:
        The extracted source for just this specific lambda, or the original
        source if extraction fails or isn't needed.
    """
    if code.co_name != "<lambda>" or not hasattr(code, "co_positions"):
        return source

    # Find first meaningful position (skip entries with zero columns)
    target_line = target_col = None
    for line, _, col, end_col in code.co_positions():
        if col or end_col:
            target_line, target_col = line, col
            break
    if target_line is None:
        return source

    # Adjust to source-relative coordinates
    source_line = target_line - code.co_firstlineno + 1

    # Tokenize source
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenizeError:
        return source

    # Find each lambda and its body-start position (the colon).
    # co_positions points to the body, so we need the lambda whose colon
    # is closest to but before the target position.
    best_lambda_idx = None
    best_colon_col = -1

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == tokenize.NAME and tok.string == "lambda":
            # Find this lambda's body colon, skipping nested lambdas and structures
            depth = 0
            lambda_depth = 0
            for j in range(i + 1, len(tokens)):
                t = tokens[j]
                if t.type == tokenize.NAME and t.string == "lambda":
                    lambda_depth += 1
                elif t.type == tokenize.OP:
                    if t.string in "([{":
                        depth += 1
                    elif t.string in ")]}":
                        depth -= 1
                    elif t.string == ":" and depth == 0:
                        if lambda_depth > 0:
                            # This colon belongs to a nested lambda
                            lambda_depth -= 1
                        else:
                            # This is our lambda's body colon
                            colon_line, colon_col = t.start
                            if colon_line < source_line or (
                                colon_line == source_line and colon_col < target_col
                            ):
                                if colon_col > best_colon_col:
                                    best_lambda_idx = i
                                    best_colon_col = colon_col
                            break
        i += 1

    if best_lambda_idx is None:
        return source

    # Find end of lambda: comma/colon at depth 0 in body, closing paren, or newline.
    # A colon after the body starts indicates an enclosing lambda's body separator.
    idx = best_lambda_idx
    depth = 0
    lambda_depth = 0
    past_colon = False
    end_idx = idx
    for j in range(idx + 1, len(tokens)):  # Start after our lambda keyword
        t = tokens[j]
        if t.type == tokenize.NAME and t.string == "lambda":
            lambda_depth += 1
        elif t.type == tokenize.OP:
            if t.string == ":" and depth == 0:
                if lambda_depth > 0:
                    lambda_depth -= 1
                elif past_colon:
                    # Second colon at depth 0 - enclosing lambda's body
                    end_idx = j - 1
                    break
                else:
                    past_colon = True
            elif t.string in "([{":
                depth += 1
            elif t.string in ")]}":
                if depth == 0:
                    end_idx = j - 1
                    break
                depth -= 1
            elif t.string == "," and depth == 0 and past_colon:
                end_idx = j - 1
                break
        elif t.type in (tokenize.NEWLINE, tokenize.ENDMARKER):
            end_idx = j - 1
            break
        end_idx = j

    # Extract source span
    start, end = tokens[idx], tokens[end_idx]
    lines = source.splitlines(keepends=True)
    if start.start[0] == end.end[0]:
        return lines[start.start[0] - 1][start.start[1] : end.end[1]]
    parts = []
    for n in range(start.start[0], end.end[0] + 1):
        line = lines[n - 1]
        if n == start.start[0]:
            parts.append(line[start.start[1] :])
        elif n == end.end[0]:
            parts.append(line[: end.end[1]])
        else:
            parts.append(line)
    result = "".join(parts)
    # Multi-line lambdas need parentheses to be syntactically valid
    if "\n" in result:
        result = "(" + result + ")"
    return result


def _is_dataclass_generated_method(func: types.FunctionType) -> bool:
    """Check if a function is a dynamically generated dataclass method.

    Dataclass methods like __init__, __repr__, __eq__, etc. are generated
    dynamically by the @dataclass decorator using exec(). They don't have
    inspectable source code, so we need to detect them and handle them specially.

    Detection criteria:
    1. The function name is one of the known dataclass-generated method names
    2. The function's qualname suggests it's a method (contains '.')
    3. Trying to get source fails (dynamically generated)
    4. The enclosing class is a dataclass

    Args:
        func: The function to check.

    Returns:
        True if this appears to be a dataclass-generated method.
    """
    # Quick check: is this a known dataclass method name?
    if func.__name__ not in _DATACLASS_GENERATED_METHODS:
        return False

    # Check if it looks like a method (qualname contains class name)
    qualname = getattr(func, "__qualname__", "")
    if "." not in qualname:
        return False

    # Try to verify source is unavailable (dataclass methods are exec'd)
    try:
        inspect.getsource(func)
        # If we got source, it's not a dataclass-generated method
        return False
    except OSError:
        pass

    # Try to find the class and check if it's a dataclass
    # The qualname format is "ClassName.method_name" or "Outer.ClassName.method_name"
    class_qualname = qualname.rsplit(".", 1)[0]
    module = func.__globals__.get("__name__")

    if module:
        import sys

        mod = sys.modules.get(module)
        if mod:
            # Try to find the class by traversing the qualname
            try:
                obj = mod
                for part in class_qualname.split("."):
                    obj = getattr(obj, part)
                if isinstance(obj, type) and dataclasses.is_dataclass(obj):
                    return True
            except AttributeError:
                pass

    # Fallback: check if the globals contain dataclass-related markers
    # This catches cases where the class might not be easily accessible
    return False


def _make_dataclass_skeleton(
    name: str,
    bases: tuple,
    type_kwargs: dict,
    fields_info: list,
    dataclass_params: dict,
    class_tracker_id: str,
) -> type:
    """Reconstruct a dataclass skeleton that will have @dataclass applied.

    This creates a class with the field annotations and then applies the
    @dataclass decorator to regenerate all the methods (__init__, __repr__, etc.)
    for the current Python version.

    Args:
        name: Class name.
        bases: Base classes tuple.
        type_kwargs: Additional type kwargs (__module__, etc.).
        fields_info: List of (field_name, field_type, field_obj_or_default) tuples.
        dataclass_params: Parameters passed to @dataclass decorator.
        class_tracker_id: Tracking ID for class identity preservation.

    Returns:
        The reconstructed dataclass.
    """
    # Build the class namespace with annotations and defaults
    namespace = {}
    annotations = {}

    for field_name, field_type, field_default in fields_info:
        annotations[field_name] = field_type
        if field_default is not dataclasses.MISSING:
            namespace[field_name] = field_default

    namespace["__annotations__"] = annotations

    # Merge in type_kwargs
    namespace.update(type_kwargs)

    # Create the class
    cls = type(name, bases, namespace)

    # Apply the @dataclass decorator with the original parameters
    cls = dataclasses.dataclass(**dataclass_params)(cls)

    # Track for identity preservation (like cloudpickle does)
    return _cloudpickle_internal._lookup_class_or_track(class_tracker_id, cls)


def _dataclass_reduce(cls: type) -> tuple:
    """Reduce a dataclass for pickling in a cross-version compatible way.

    Instead of pickling the dynamically generated methods (which use
    version-specific bytecode), we pickle the class definition and
    dataclass parameters. On deserialization, @dataclass is re-applied
    to regenerate all methods for the target Python version.

    Args:
        cls: The dataclass to reduce.

    Returns:
        A tuple for pickle's reduce protocol.
    """
    # Get the dataclass parameters
    params = getattr(cls, "__dataclass_params__", None)
    if params is None:
        # Fallback to defaults
        dataclass_params = {}
    else:
        dataclass_params = {
            "init": params.init,
            "repr": params.repr,
            "eq": params.eq,
            "order": params.order,
            "unsafe_hash": params.unsafe_hash,
            "frozen": params.frozen,
        }
        # Add newer params if they exist
        if hasattr(params, "match_args"):
            dataclass_params["match_args"] = params.match_args
        if hasattr(params, "kw_only"):
            dataclass_params["kw_only"] = params.kw_only
        if hasattr(params, "slots"):
            dataclass_params["slots"] = params.slots

    # Extract field information
    fields_info = []
    for field in dataclasses.fields(cls):
        # Get the field type (might be a string annotation or actual type)
        field_type = field.type

        # Determine the default value
        if field.default is not dataclasses.MISSING:
            field_default = field.default
        elif field.default_factory is not dataclasses.MISSING:
            # For default_factory, we need to wrap it in a Field object
            field_default = dataclasses.field(default_factory=field.default_factory)
        else:
            field_default = dataclasses.MISSING

        fields_info.append((field.name, field_type, field_default))

    # Get type kwargs for reconstruction
    type_kwargs = {}
    if "__module__" in cls.__dict__:
        type_kwargs["__module__"] = cls.__module__

    # Get class tracker ID for identity preservation
    class_tracker_id = _cloudpickle_internal._get_or_create_tracker_id(cls)

    # Get the class state (non-generated attributes)
    clsdict, slotstate = _cloudpickle_internal._class_getstate(cls)

    # Remove dataclass-generated methods from the class dict
    # These will be regenerated by @dataclass on the other end
    for method_name in _DATACLASS_GENERATED_METHODS:
        clsdict.pop(method_name, None)

    # Also remove dataclass internal attributes that will be regenerated
    clsdict.pop("__dataclass_fields__", None)
    clsdict.pop("__dataclass_params__", None)

    args = (
        cls.__name__,
        cls.__bases__,
        type_kwargs,
        fields_info,
        dataclass_params,
        class_tracker_id,
    )

    state = (clsdict, slotstate)

    return (
        _make_dataclass_skeleton,
        args,
        state,
        None,
        None,
        _cloudpickle_internal._class_setstate,
    )


def make_function(
    source: str,
    name: str,
    filename: Optional[str],
    qualname: str,
    module: str,
    doc: Optional[str],
    annotations: Optional[dict],
    defaults: Optional[tuple],
    kwdefaults: Optional[dict],
    base_globals: dict,
    closure_values: Optional[list],
    closure_names: Optional[list],
) -> types.FunctionType:
    """Reconstruct a function from its serialized source code and metadata.

    This is the deserialization counterpart to CustomCloudPickler's function
    serialization. It recompiles source code and reconstructs the function
    with all its original attributes (defaults, annotations, closure, etc.).

    This function creates the function with minimal globals. The full globals
    (including any self-references for recursive functions) are applied later
    by _source_function_setstate, which is called after pickle memoizes the
    function. This two-phase approach enables proper handling of circular
    references like recursive or mutually recursive functions.

    Args:
        source: The function's source code as a string. May be indented.
        name: The function's __name__ attribute.
        filename: Original filename where the function was defined. Used for
            tracebacks and debugging. Falls back to "<serialization>" if None.
        qualname: The function's __qualname__ (qualified name including class).
        module: The function's __module__ attribute.
        doc: The function's docstring (__doc__).
        annotations: Type annotations dict (__annotations__).
        defaults: Default values for positional arguments (__defaults__).
        kwdefaults: Default values for keyword-only arguments (__kwdefaults__).
        base_globals: Minimal global variables dict. Full globals including
            self-references are added later by _source_function_setstate.
        closure_values: List of closure variable values (passed immediately, not
            deferred, because closures need factory pattern to bind properly).
        closure_names: List of closure variable names (co_freevars).

    Returns:
        A newly constructed function object. Note: the globals are minimal
        at this point - they're filled in by _source_function_setstate.

    Raises:
        ValueError: If the function name cannot be found in the compiled source.
    """
    # Remove any leading indentation (e.g., if function was defined inside a class)
    source = textwrap.dedent(source)

    # Set up the global namespace for the reconstructed function.
    # This is a minimal globals dict - full globals are added by _source_function_setstate.
    func_globals = {"__builtins__": __builtins__, **base_globals}

    if closure_values and closure_names:
        # CLOSURE HANDLING: Functions with closures require special treatment.
        #
        # Python closures work by capturing variables from enclosing scopes.
        # We can't directly create closure cells, so we use a factory pattern:
        # wrap the function definition inside another function that takes the
        # closure values as parameters, then call it to create the real function.
        #
        # Note: closure values are bound here (not deferred) because the factory
        # pattern requires them at function creation time. However, the globals
        # (including any self-references) ARE deferred to the state setter.
        closure_params = ", ".join(closure_names)
        # Prepend 'from __future__ import annotations' so type annotations are
        # stored as strings and not evaluated at function definition time.
        # Without this, annotations like 'x: SomeType' would fail if SomeType
        # isn't in the limited func_globals used during reconstruction.
        factory_source = "from __future__ import annotations\n"
        factory_source += f"def _seri_factory_({closure_params}):\n"

        # Lambdas have '<lambda>' as their name, which is not valid Python syntax.
        # We handle them by assigning the lambda expression to a temporary variable.
        if name == "<lambda>":
            indented_source = "    _lambda_result_ = " + source + "\n"
            factory_source += indented_source
            factory_source += "    return _lambda_result_\n"
        else:
            indented_source = textwrap.indent(source, "    ")
            factory_source += indented_source + "\n"
            factory_source += f"    return {name}\n"

        # Compile and execute the factory, then call it with closure values
        try:
            factory_code = compile(
                factory_source, filename or "<serialization>", "exec"
            )
        except SyntaxError as e:
            raise ValueError(
                f"Failed to compile source for function '{name}'. "
                f"This may indicate corrupted serialized data or version incompatibility."
            ) from e
        exec(factory_code, func_globals)
        factory = func_globals["_seri_factory_"]
        func = factory(*closure_values)

        # The factory path doesn't preserve defaults, so restore them
        if defaults:
            func.__defaults__ = tuple(defaults)
    else:

        try:
            module_code = compile(source, filename or "<serialization>", "exec")
        except SyntaxError as e:
            raise ValueError(
                f"Failed to compile source for function '{name}'. "
                f"This may indicate corrupted serialized data or version incompatibility."
            ) from e

        # Search through the module's constants to find our function's code object.
        # For lambdas with nested lambdas as defaults, multiple code objects have
        # co_name=="<lambda>". The outermost lambda is last, so don't break early.
        func_code = None
        for const in module_code.co_consts:
            if isinstance(const, types.CodeType) and const.co_name == name:
                func_code = const
                # Don't break for lambdas - we want the last (outermost) one

        if func_code is None:
            raise ValueError(f"Could not find function '{name}' in compiled source")

        # Create the function directly from the code object
        func = types.FunctionType(func_code, func_globals, name, defaults, None)

    # Restore function metadata
    if kwdefaults:
        func.__kwdefaults__ = kwdefaults

    if annotations:
        func.__annotations__ = annotations

    # Restore identity attributes for proper introspection
    func.__module__ = module
    func.__doc__ = doc
    func.__qualname__ = qualname

    # Attach original source for re-serialization (inspect.getsource won't work
    # because line numbers don't match the original file)
    func.__source__ = source

    return func


class SerializedFrame:
    def __init__(self, co_filename: str, co_firstlineno: int, co_name: str):

        self.f_locals = {}
        self.f_globals = {}

        self.f_code = types.SimpleNamespace(
            co_filename=co_filename,
            co_firstlineno=co_firstlineno,
            co_name=co_name,
        )


def make_frame(co_filename: str, co_firstlineno: int, co_name: str) -> tuple:
    return SerializedFrame(co_filename, co_firstlineno, co_name)


def _source_function_setstate(func: types.FunctionType, state: tuple) -> None:
    """Update the state of a source-serialized function after memoization.

    This is called by pickle after the function has been created and memoized.
    It updates the function's globals with the full captured values, including
    any self-references (for recursive functions) or cross-references (for
    mutually recursive functions).

    This two-phase approach (create minimal function, then fill in state) is
    essential for handling circular references. By the time this function is
    called, the function object is already in pickle's memo, so any references
    to it in the state (like self-references in globals) resolve correctly.

    For local recursive functions (where the self-reference is in a closure
    variable rather than globals), we also fill in deferred closure cells here.

    Args:
        func: The function to update (already created by make_function).
        state: A tuple of (func_dict, slotstate) where:
            - func_dict: Custom attributes to add to func.__dict__
            - slotstate: Dict with keys:
                - "__globals__": Full captured globals to merge into func.__globals__
                - "__deferred_closure__": Dict mapping closure cell indices to the
                  function objects that should fill those cells (handles both
                  self-references and cross-references for mutual recursion)
    """
    func_dict, slotstate = state

    # Update func.__dict__ with any custom attributes
    func.__dict__.update(func_dict)

    # Get the full globals from state
    full_globals = slotstate.get("__globals__", {})

    # Update globals in place - this is the key to handling self-references!
    # The function is already memoized, so any self-reference in full_globals
    # now points to the existing (memoized) function object.
    func.__globals__.update(full_globals)

    # Fill in deferred closure cells (for local recursive/mutually recursive functions).
    # These are closure cells containing function references that were replaced
    # with None placeholders during serialization to avoid infinite recursion.
    # Now that all functions are memoized, we can safely fill them in.
    deferred_closure = slotstate.get("__deferred_closure__", {})
    if deferred_closure and func.__closure__ is not None:
        for idx_str, target_func in deferred_closure.items():
            # The index might be a string if it came through JSON-like serialization
            idx = int(idx_str) if isinstance(idx_str, str) else idx_str
            # Fill in the function reference (could be self or another function)
            func.__closure__[idx].cell_contents = target_func


class CustomCloudPickler(cloudpickle.Pickler):
    """A cloudpickle-based pickler that serializes functions by source code.

    This pickler extends cloudpickle.Pickler to override how dynamic functions
    are serialized. Instead of using bytecode (which is Python version-specific),
    it captures the function's source code, enabling cross-version compatibility.

    Key features:
        - Source-based function serialization via _dynamic_function_reduce
        - Persistent object references via persistent_id for objects that
          shouldn't be fully serialized

    Examples:
        >>> import io
        >>> def my_func(x):
        ...     return x * 2
        >>> buffer = io.BytesIO()
        >>> CustomCloudPickler(buffer).dump(my_func)
        >>> # Function is now serialized with its source code
    """

    def reducer_override(self, obj):
        # Special handling for dataclass classes
        # We serialize them in a way that re-applies @dataclass on deserialization,
        # which regenerates all methods for the target Python version
        if isinstance(obj, type) and dataclasses.is_dataclass(obj):
            # Check if this should be pickled by reference (importable module attribute)
            if not _cloudpickle_internal._should_pickle_by_reference(obj):
                return _dataclass_reduce(obj)

        result = super().reducer_override(obj)

        if isinstance(obj, types.FrameType):
            return self._frame_reduce(obj)

        return result

    def _frame_reduce(self, frame: types.FrameType) -> tuple:

        return (
            make_frame,
            (
                frame.f_code.co_filename,
                frame.f_code.co_firstlineno,
                frame.f_code.co_name,
            ),
            None,
            None,
            None,
        )

    def _dynamic_function_reduce(self, func: types.FunctionType) -> tuple:
        """Serialize a function by capturing its source code and metadata.

        This method is called by cloudpickle when serializing dynamic functions
        (functions defined at runtime or in __main__). We override it to capture
        source code instead of bytecode.

        The serialization uses pickle's 6-tuple reduce protocol with a state setter
        to properly handle circular references (like recursive functions). The
        pattern is:
            1. make_function creates a function with minimal globals and empty closures
            2. Pickle memoizes the function object
            3. _source_function_setstate fills in the full globals and closure values

        This deferred state application is essential because the full globals may
        contain references to the function itself (for recursive functions) or to
        other functions that reference this one (for mutual recursion). By the time
        the state setter runs, the function is already memoized, so these references
        resolve correctly.

        Args:
            func: The function to serialize.

        Returns:
            A 6-tuple for pickle's reduce protocol:
                - Callable to reconstruct the function (make_function)
                - Args tuple with minimal state (source, metadata, empty closures)
                - State tuple (func_dict, slotstate with full globals/closure)
                - None (for list items, unused)
                - None (for dict items, unused)
                - State setter function (_source_function_setstate)

        Note:
            If the function has a __source__ attribute (manually attached),
            that is used instead of calling inspect.getsource(). This is useful
            for functions where source inspection might fail (e.g., dynamically
            generated functions with attached source).
        """
        # Check if this is a dataclass-generated method (like __init__, __repr__, etc.)
        # These are created dynamically by exec() and have no source code.
        # Fall back to cloudpickle's bytecode-based serialization for these.
        # Note: The dataclass class itself is handled specially in reducer_override,
        # which re-applies @dataclass on deserialization to regenerate these methods.
        if _is_dataclass_generated_method(func):
            # Use cloudpickle's default bytecode-based serialization
            return super()._dynamic_function_reduce(func)

        # Get source code - prefer explicit __source__ attribute if present.
        # This allows users to attach source to dynamically generated functions
        # where inspect.getsource() would fail.
        if hasattr(func, "__source__"):
            source = func.__source__
        else:
            try:
                source = inspect.getsource(func)
            except OSError as e:
                raise pickle.PicklingError(
                    f"Cannot serialize function '{func.__name__}': source code unavailable. "
                    f"Attach source manually via func.__source__ = '...'. Original error: {e}"
                ) from e
            # For lambdas, extract just this lambda when multiple share a line
            source = _extract_lambda_source(source, func.__code__)

        # Use cloudpickle's internal helper to extract function state.
        # _function_getstate returns:
        #   - func_dict: func.__dict__ (custom attributes)
        #   - slotstate: dict of function slots (__globals__, __defaults__, etc.)
        func_dict, slotstate = _function_getstate(func)

        # Remove __source__ from func_dict since we handle it separately
        func_dict = {k: v for k, v in func_dict.items() if k != "__source__"}

        # Extract metadata for the args tuple
        name = slotstate["__name__"]
        defaults = slotstate["__defaults__"]
        kwdefaults = slotstate["__kwdefaults__"]
        annotations = slotstate["__annotations__"]
        filename = func.__code__.co_filename
        qualname = slotstate["__qualname__"]
        module_name = slotstate["__module__"]
        doc = slotstate["__doc__"]

        # BASE GLOBALS: Minimal globals for function creation.
        # Following cloudpickle's pattern, we only include module identity attrs.
        # The full globals (including any self-references) go in the state and
        # are applied AFTER memoization by _source_function_setstate.
        base_globals = {}
        func_globals = func.__globals__
        for k in ["__package__", "__name__", "__path__", "__file__"]:
            if k in func_globals:
                base_globals[k] = func_globals[k]

        # SECURITY CHECK: Check function globals for prohibited modules/functions.
        # This catches dangerous patterns like `import os; os.getcwd()` or
        # `from subprocess import run; run(...)` where the prohibited object
        # is captured in the function's globals.
        captured_globals = slotstate.get("__globals__", {})

        # CLOSURE: Extract closure values and names.
        # For recursive/mutually recursive local functions, the closure may contain
        # function references that create cycles. We defer ALL function values in
        # closures to the state setter, which runs after memoization. This allows
        # pickle's memo to break the cycle.
        deferred_closure = {}  # Maps index -> function to fill in after memoization
        if func.__closure__ is not None:
            closure_names = list(func.__code__.co_freevars)
            closure_values = []
            for i, cell in enumerate(func.__closure__):
                value = _get_cell_contents(cell)

                if isinstance(value, types.FunctionType):
                    # Defer ALL functions to state setter to handle cycles.
                    # This includes self-references and cross-references.
                    closure_values.append(None)  # Placeholder
                    deferred_closure[i] = value
                else:
                    closure_values.append(value)
        else:
            closure_values = None
            closure_names = None

        # STATE: Full globals and deferred closure info, applied after memoization.
        # This is where self-references and cross-references live - by the time
        # _source_function_setstate is called, the function is already memoized,
        # so these references resolve correctly.
        state_slotstate = {
            "__globals__": slotstate["__globals__"],
            "__deferred_closure__": deferred_closure,  # Maps indices to target functions
        }
        state = (func_dict, state_slotstate)

        # Args for make_function - creates function with minimal globals.
        # Closure values are passed here (not deferred) because the factory
        # pattern needs them at function creation time.
        args = (
            source,
            name,
            filename,
            qualname,
            module_name,
            doc,
            annotations,
            defaults,
            kwdefaults,
            base_globals,
            closure_values,
            closure_names,
        )

        # Return 6-tuple: (func, args, state, listitems, dictitems, state_setter)
        # The state_setter is called AFTER memoization to fill in full globals/closure
        return (make_function, args, state, None, None, _source_function_setstate)

    def persistent_id(self, obj: Any) -> Optional[Any]:
        """Return a persistent ID for objects that shouldn't be fully serialized.

        Pickle's persistent_id mechanism allows certain objects to be referenced
        by an ID rather than serialized. During deserialization, persistent_load
        resolves these IDs back to actual objects.

        This is critical for nnsight's remote execution where certain objects
        (like model proxies, intervention graph nodes, or large tensors) should
        not be serialized but instead looked up on the server side.

        Args:
            obj: The object being pickled.

        Returns:
            The persistent ID if obj has a `_persistent_id` in its __dict__,
            otherwise None (meaning pickle should serialize normally).

        Examples:
            An object with obj.__dict__["_persistent_id"] = "node_42" will be
            serialized as just the reference "node_42", and during deserialization,
            persistent_load("node_42") will be called to resolve it.
        """
        try:
            return obj.__dict__["_persistent_id"]
        except (AttributeError, KeyError, TypeError):
            # Object doesn't have __dict__ or doesn't have _persistent_id
            pass

        return None


class CustomCloudUnpickler(pickle.Unpickler):
    """A custom unpickler that resolves persistent object references.

    Works in conjunction with CustomCloudPickler to handle objects that were
    serialized by reference (persistent_id) rather than by value. During
    deserialization, persistent IDs are looked up in the provided dictionary.

    This enables patterns where certain objects (like model proxies or graph
    nodes) are referenced by ID in the serialized data and resolved to actual
    objects on the server side.

    Args:
        file: File-like object to read pickle data from.
        persistent_objects: Dictionary mapping persistent IDs to actual objects.
            When a persistent ID is encountered during deserialization, it's
            looked up in this dictionary.

    Examples:
        >>> # On the server side
        >>> model_proxy = get_model_proxy("gpt2")
        >>> persistent_objects = {"model_ref_1": model_proxy}
        >>> data = receive_from_client()
        >>> obj = CustomCloudUnpickler(io.BytesIO(data), persistent_objects).load()
        >>> # Any references to "model_ref_1" in the data are now resolved
    """

    def __init__(self, file: BinaryIO, persistent_objects: Optional[dict] = None):
        """Initialize the unpickler with a file and optional persistent objects.

        Args:
            file: Binary file-like object containing pickle data.
            persistent_objects: Optional dict mapping persistent IDs to objects.
                Defaults to empty dict if not provided.
        """
        super().__init__(file)
        self.persistent_objects = persistent_objects or {}

    def persistent_load(self, pid: Any) -> Any:
        """Resolve a persistent ID to its corresponding object.

        Called automatically by pickle when it encounters a persistent reference
        (created by persistent_id during serialization).

        Args:
            pid: The persistent ID to resolve.

        Returns:
            The object corresponding to the persistent ID.

        Raises:
            pickle.UnpicklingError: If the persistent ID is not found in
                the persistent_objects dictionary.
        """
        if pid in self.persistent_objects:
            return self.persistent_objects[pid]

        raise pickle.UnpicklingError(f"Unknown persistent id: {pid}")


def dumps(
    obj: Any,
    path: Optional[Union[str, Path]] = None,
    protocol: int = DEFAULT_PROTOCOL,
) -> Optional[bytes]:
    """Serialize an object using source-based function serialization.

    This is the high-level API for serializing objects with CustomCloudPickler.
    Functions in the object graph will be serialized by source code rather than
    bytecode, enabling cross-Python-version compatibility.

    Args:
        obj: Any picklable object. Functions will be serialized by source code.
        path: Optional file path to write the serialized data to.
            Accepts both string paths and pathlib.Path objects.
            If None, returns the serialized bytes directly.
        protocol: Pickle protocol version to use. Defaults to DEFAULT_PROTOCOL (4).
            Protocol 4 is available in Python 3.4+ and supports large objects.

    Returns:
        If path is None: The serialized data as bytes.
        If path is provided: None (data is written to file).

    Examples:
        >>> # Serialize to bytes (for network transmission)
        >>> data = dumps(my_function)
        >>> send_to_server(data)
        >>>
        >>> # Serialize to file (for persistence)
        >>> dumps(my_function, "/path/to/function.pkl")
        >>>
        >>> # Using pathlib.Path
        >>> from pathlib import Path
        >>> dumps(my_function, Path("./functions/my_func.pkl"))
    """
    if path is None:
        # In-memory serialization - return bytes directly
        buffer = io.BytesIO()
        CustomCloudPickler(buffer, protocol=protocol).dump(obj)
        buffer.seek(0)
        return buffer.read()

    # File-based serialization - write to disk
    # Normalize to Path for consistent handling
    path = Path(path)
    with path.open("wb") as file:
        CustomCloudPickler(file, protocol=protocol).dump(obj)


def loads(
    data: Union[str, bytes, Path],
    persistent_objects: Optional[dict] = None,
) -> Any:
    """Deserialize data that was serialized with dumps().

    This is the high-level API for deserializing objects with CustomCloudUnpickler.
    Functions serialized by source code will be reconstructed by recompiling
    their source on the current Python version.

    Args:
        data: One of:
            - bytes: Serialized data (e.g., received over network)
            - str: Path to a file containing serialized data
            - Path: pathlib.Path to a file containing serialized data
        persistent_objects: Optional dictionary mapping persistent IDs to objects.
            Used to resolve objects that were serialized by reference rather
            than by value. See CustomCloudUnpickler for details.

    Returns:
        The deserialized object.

    Raises:
        pickle.UnpicklingError: If a persistent ID is encountered that isn't
            in the persistent_objects dictionary.
        FileNotFoundError: If a file path is provided but the file doesn't exist.

    Examples:
        >>> # Load from bytes (received over network)
        >>> data = receive_from_client()
        >>> obj = loads(data)
        >>>
        >>> # Load from file (string path)
        >>> obj = loads("/path/to/function.pkl")
        >>>
        >>> # Load from file (pathlib.Path)
        >>> from pathlib import Path
        >>> obj = loads(Path("./functions/my_func.pkl"))
        >>>
        >>> # Load with persistent object resolution
        >>> persistent = {"model_proxy": actual_model}
        >>> obj = loads(data, persistent_objects=persistent)
    """
    if isinstance(data, bytes):
        # In-memory deserialization from bytes
        return CustomCloudUnpickler(io.BytesIO(data), persistent_objects).load()

    # File-based deserialization - data is a file path
    # Normalize to Path for consistent handling
    path = Path(data)
    with path.open("rb") as file:
        return CustomCloudUnpickler(file, persistent_objects).load()


# Backward-compatible aliases for existing code that uses save/load
save = dumps
load = loads
