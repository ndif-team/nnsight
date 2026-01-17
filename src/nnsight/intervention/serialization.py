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

Example:
    >>> import serialization
    >>> def my_func(x, y=10):
    ...     return x + y
    >>> data = serialization.dumps(my_func)
    >>> restored = serialization.loads(data)
    >>> restored(5)  # Returns 15
"""

import inspect
import io
import pickle
import textwrap
import types
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

import cloudpickle
from cloudpickle.cloudpickle import _function_getstate, _get_cell_contents

# Default pickle protocol - protocol 4 is available in Python 3.4+ and supports large objects
DEFAULT_PROTOCOL = 4


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
    globals_dict: dict,
    closure: Optional[list],
    closure_names: Optional[list],
) -> types.FunctionType:
    """Reconstruct a function from its serialized source code and metadata.

    This is the deserialization counterpart to CustomCloudPickler's function
    serialization. It recompiles source code and reconstructs the function
    with all its original attributes (defaults, annotations, closure, etc.).

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
        globals_dict: Global variables the function needs access to. These are
            captured during serialization and restored here.
        closure: List of closure variable values (the actual values, not cells).
        closure_names: List of closure variable names (co_freevars). Must match
            the order and length of `closure`.

    Returns:
        A newly constructed function object equivalent to the original.

    Raises:
        ValueError: If the function name cannot be found in the compiled source.

    Note:
        For functions with closures, we use a factory function pattern to properly
        bind closure variables. This is necessary because Python's closure mechanism
        requires variables to be captured from an enclosing scope - we can't just
        assign them directly to a code object.
    """
    # Remove any leading indentation (e.g., if function was defined inside a class)
    source = textwrap.dedent(source)

    # Set up the global namespace for the reconstructed function.
    # We need __builtins__ for the function to access built-in functions.
    func_globals = {"__builtins__": __builtins__, **globals_dict}

    if closure and closure_names:
        # CLOSURE HANDLING: Functions with closures require special treatment.
        #
        # Python closures work by capturing variables from enclosing scopes.
        # We can't directly create closure cells, so we use a factory pattern:
        # wrap the function definition inside another function that takes the
        # closure values as parameters, then call it to create the real function.
        #
        # Example: if original was:
        #     def outer():
        #         x = 10
        #         def inner(y):
        #             return x + y
        #         return inner
        #
        # We generate:
        #     def _seri_factory_(x):
        #         def inner(y):
        #             return x + y
        #         return inner
        #     func = _seri_factory_(10)
        closure_params = ", ".join(closure_names)
        factory_source = f"def _seri_factory_({closure_params}):\n"
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
        func = factory(*closure)
    else:
        # NO CLOSURE: Simpler path - compile source and extract the code object.
        #
        # We compile the entire source as a module, then find the specific
        # function's code object in the module's constants.
        try:
            module_code = compile(source, filename or "<serialization>", "exec")
        except SyntaxError as e:
            raise ValueError(
                f"Failed to compile source for function '{name}'. "
                f"This may indicate corrupted serialized data or version incompatibility."
            ) from e

        # Search through the module's constants to find our function's code object
        func_code = None
        for const in module_code.co_consts:
            if isinstance(const, types.CodeType) and const.co_name == name:
                func_code = const
                break

        if func_code is None:
            raise ValueError(f"Could not find function '{name}' in compiled source")

        # Create the function directly from the code object
        func = types.FunctionType(func_code, func_globals, name, defaults, None)

    # RESTORE FUNCTION METADATA
    #
    # The factory path above doesn't preserve defaults (they become regular
    # parameters in the factory), so we need to restore them explicitly.
    if closure and closure_names and defaults:
        func.__defaults__ = tuple(defaults)

    if kwdefaults:
        func.__kwdefaults__ = kwdefaults

    if annotations:
        func.__annotations__ = annotations

    # Restore identity attributes for proper introspection
    func.__module__ = module
    func.__doc__ = doc
    func.__qualname__ = qualname

    return func


class CustomCloudPickler(cloudpickle.Pickler):
    """A cloudpickle-based pickler that serializes functions by source code.

    This pickler extends cloudpickle.Pickler to override how dynamic functions
    are serialized. Instead of using bytecode (which is Python version-specific),
    it captures the function's source code, enabling cross-version compatibility.

    Key features:
        - Source-based function serialization via _dynamic_function_reduce
        - Persistent object references via persistent_id for objects that
          shouldn't be fully serialized

    Example:
        >>> import io
        >>> def my_func(x):
        ...     return x * 2
        >>> buffer = io.BytesIO()
        >>> CustomCloudPickler(buffer).dump(my_func)
        >>> # Function is now serialized with its source code
    """

    def _dynamic_function_reduce(self, func: types.FunctionType) -> tuple:
        """Serialize a function by capturing its source code and metadata.

        This method is called by cloudpickle when serializing dynamic functions
        (functions defined at runtime or in __main__). We override it to capture
        source code instead of bytecode.

        Args:
            func: The function to serialize.

        Returns:
            A 5-tuple for pickle's reduce protocol:
                - Callable to reconstruct the function (make_function)
                - Args tuple for that callable
                - State dict (func.__dict__ minus __source__)
                - None (for list items, unused)
                - None (for dict items, unused)

        Note:
            If the function has a __source__ attribute (manually attached),
            that is used instead of calling inspect.getsource(). This is useful
            for functions where source inspection might fail (e.g., dynamically
            generated functions with attached source).
        """
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

        # Use cloudpickle's internal helper to extract function state.
        # _function_getstate returns:
        #   - state: func.__dict__ (custom attributes)
        #   - slotstate: dict of function slots (__globals__, __defaults__, etc.)
        state, slotstate = _function_getstate(func)

        # Extract all the metadata we need to reconstruct the function
        func_globals_captured = slotstate["__globals__"]
        defaults = slotstate["__defaults__"]
        kwdefaults = slotstate["__kwdefaults__"]
        annotations = slotstate["__annotations__"]
        name = slotstate["__name__"]
        filename = func.__code__.co_filename
        qualname = slotstate["__qualname__"]
        module = slotstate["__module__"]
        doc = slotstate["__doc__"]
        closure = slotstate["__closure__"]

        # Convert closure cells to their actual values.
        # Closure cells are wrapper objects; we need the contained values.
        if closure:
            closure = [_get_cell_contents(cell) for cell in closure]
        else:
            closure = None

        # Get the names of closure variables from the code object.
        # These are needed to properly rebind the closure during deserialization.
        closure_names = (
            list(func.__code__.co_freevars) if func.__code__.co_freevars else None
        )

        # Preserve custom attributes from func.__dict__, but exclude __source__
        # since we've already captured that separately.
        func_dict = {k: v for k, v in state.items() if k != "__source__"}

        # Return the reduce tuple for pickle protocol.
        # When unpickling, pickle will call: make_function(*args)
        # then update the result's __dict__ with func_dict.
        args = (
            source,
            name,
            filename,
            qualname,
            module,
            doc,
            annotations,
            defaults,
            kwdefaults,
            func_globals_captured,
            closure,
            closure_names,
        )
        return (make_function, args, func_dict, None, None)

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

        Example:
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

    Example:
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

    Example:
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

    Example:
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
