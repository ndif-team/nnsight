"""
Source-based serialization for nnsight remote execution.

This module provides serialization that uses source code + JSON instead of
cloudpickle bytecode. This enables:
- Python version independence (3.10 client can work with 3.12 server)
- Third-party libraries decorated with @nnsight.remote work without server installation
- Clear, early errors instead of mysterious runtime failures
- Auto-discovery of dependencies (classes used by LanguageModel subclasses)
"""

from __future__ import annotations

import ast
import base64
import inspect
import json
import sys
import textwrap
import types
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..intervention.tracing.base import Tracer

from ..remote import (
    is_json_serializable, ALLOWED_MODULES, ALLOWED_BASE_CLASSES,
    SERVER_AVAILABLE_MODULES, is_server_available_module, is_remote_object,
    is_lambda, extract_lambda_source, LambdaExtractionError, validate_lambda_for_remote,
    find_external_references, resolve_module_references, validate_ast,
)


class SourceSerializationError(Exception):
    """Raised when source-based serialization fails."""
    pass


# =============================================================================
# Serialization Format Constants
# =============================================================================
# Marker keys used in the serialized JSON format for type identification.
# These constants define the wire format for source-based serialization:
# - Each marker key (e.g., "__tensor__", "__nn_module__") identifies a special
#   type in the JSON payload that requires reconstruction on the server.
# - The format is designed to be self-describing so the server can reconstruct
#   Python objects (tensors, nn.Modules, @remote instances) from JSON.
# - IMPORTANT: Changing these values would break backwards compatibility with
#   existing serialized payloads and server deployments.

SERIALIZATION_VERSION = "2.2"

# Type markers in serialized data
TENSOR_MARKER = "__tensor__"
NN_PARAMETER_MARKER = "__nn_parameter__"
NN_MODULE_MARKER = "__nn_module__"
DICT_MARKER = "__dict__"
LIST_MARKER = "__list__"
TUPLE_MARKER = "__tuple__"
SET_MARKER = "__set__"
ENUM_MARKER = "__enum__"
REF_MARKER = "__ref__"
ID_MARKER = "__id__"
MODEL_REF_MARKER = "__model_ref__"
REMOTE_REF_MARKER = "__remote_ref__"
REMOTE_TYPE_MARKER = "__remote_type__"
CLASS_MARKER = "__class__"  # User-defined class instances (source transmitted separately)
CALLABLE_REF_MARKER = "__callable_ref__"
TYPE_REF_MARKER = "__type_ref__"
WEAKREF_MARKER = "__weakref__"
SERVER_PROVIDED_MARKER = "__server_provided__"
ENUM_FALLBACK_MARKER = "__enum_fallback__"


# =============================================================================
# Forbidden Serialization Lists
# =============================================================================
# Some objects should never be serialized for remote execution, even if they
# technically could be. These objects either:
# 1. Represent OS/system resources that cannot be transferred (sockets, locks)
# 2. Have huge dependency graphs that would generate confusing warnings (pandas)
# 3. Leak into scope from test frameworks or infrastructure (pytest fixtures)
#
# We check these EARLY to provide clear, actionable error messages instead of
# confusing failures after extensive processing.

# Module prefixes to skip during auto-discovery
# Objects from these modules will be immediately rejected with a clear message
FORBIDDEN_MODULE_PREFIXES: frozenset = frozenset({
    # Test frameworks - these leak into scope during testing
    '_pytest',
    'pytest',
    'unittest',

    # OS/System resources - represent state that cannot be transferred
    'socket',           # Network sockets (has Python wrapper over C extension)
    'multiprocessing',  # Process resources, queues, locks
    'asyncio',          # Async event loops, tasks
    'concurrent',       # Thread pools, futures
    'queue',            # Thread communication queues
    'subprocess',       # OS subprocesses

    # Database connections - represent network/session state
    'sqlite3',
    'sqlalchemy',
    'pymongo',
    'redis',
    'psycopg',          # PostgreSQL
    'psycopg2',
    'mysql',
    'pymysql',

    # Logging - handlers often have file references
    'logging',
})

# Specific class names (fully qualified) with custom error messages
# Format: "module.ClassName": "Error message with suggestions"
FORBIDDEN_CLASSES: Dict[str, str] = {
    # Pandas - massive dependency explosion, convert to tensor instead
    'pandas.core.frame.DataFrame': (
        "pandas.DataFrame cannot be serialized for remote execution.\n"
        "DataFrames have complex internal state that cannot be transferred.\n"
        "\n"
        "Convert to a tensor before the trace:\n"
        "  tensor_data = torch.tensor(df.values)\n"
        "Or extract specific columns:\n"
        "  values = df['column'].tolist()"
    ),
    'pandas.core.series.Series': (
        "pandas.Series cannot be serialized for remote execution.\n"
        "\n"
        "Convert to a tensor or list before the trace:\n"
        "  tensor_data = torch.tensor(series.values)\n"
        "  values = series.tolist()"
    ),

    # Matplotlib - contains rendering state and callbacks
    'matplotlib.figure.Figure': (
        "matplotlib.Figure cannot be serialized for remote execution.\n"
        "Figures contain rendering state and callbacks that cannot be transferred.\n"
        "\n"
        "If you need to pass image data, save to bytes first:\n"
        "  import io\n"
        "  buf = io.BytesIO()\n"
        "  fig.savefig(buf, format='png')\n"
        "  image_bytes = buf.getvalue()"
    ),
    'matplotlib.axes._axes.Axes': (
        "matplotlib.Axes cannot be serialized for remote execution.\n"
        "Axes contain rendering state bound to a Figure.\n"
        "\n"
        "Access the data you need before the trace instead."
    ),
    'matplotlib.axes._subplots.Axes': (
        "matplotlib.Axes cannot be serialized for remote execution.\n"
        "Axes contain rendering state bound to a Figure.\n"
        "\n"
        "Access the data you need before the trace instead."
    ),
    'matplotlib.axes._subplots.AxesSubplot': (
        "matplotlib.Axes cannot be serialized for remote execution.\n"
        "Axes contain rendering state bound to a Figure.\n"
        "\n"
        "Access the data you need before the trace instead."
    ),

    # PIL/Pillow - convert to tensor
    'PIL.Image.Image': (
        "PIL.Image cannot be serialized for remote execution.\n"
        "\n"
        "Convert to a tensor first:\n"
        "  from torchvision import transforms\n"
        "  tensor = transforms.ToTensor()(image)"
    ),

    # Scipy sparse matrices - convert to dense or use specific format
    'scipy.sparse._csr.csr_matrix': (
        "scipy.sparse.csr_matrix cannot be serialized for remote execution.\n"
        "\n"
        "Convert to a dense tensor:\n"
        "  tensor = torch.tensor(sparse_matrix.toarray())\n"
        "Or extract the CSR components if you need sparse format."
    ),
    'scipy.sparse._csc.csc_matrix': (
        "scipy.sparse.csc_matrix cannot be serialized for remote execution.\n"
        "\n"
        "Convert to a dense tensor:\n"
        "  tensor = torch.tensor(sparse_matrix.toarray())"
    ),
}


def _get_full_class_name(obj: Any) -> str:
    """Get the fully qualified class name of an object."""
    cls = type(obj) if not isinstance(obj, type) else obj
    module = getattr(cls, '__module__', '')
    name = getattr(cls, '__qualname__', cls.__name__)
    return f"{module}.{name}" if module else name


def is_forbidden_for_serialization(obj: Any) -> Tuple[bool, Optional[str]]:
    """
    Check if an object is forbidden from serialization.

    This check happens EARLY, before any auto-discovery attempt, to provide
    clear error messages without generating confusing warnings.

    Args:
        obj: The object to check

    Returns:
        (is_forbidden, error_message) tuple. If is_forbidden is True,
        error_message contains a helpful explanation and suggestions.
    """
    # Get the class and module info
    cls = type(obj) if not isinstance(obj, type) else obj
    module = getattr(cls, '__module__', '') or ''
    full_name = _get_full_class_name(obj)

    # Check against forbidden class names first (most specific)
    if full_name in FORBIDDEN_CLASSES:
        return True, FORBIDDEN_CLASSES[full_name]

    # Check against forbidden module prefixes
    for prefix in FORBIDDEN_MODULE_PREFIXES:
        if module.startswith(prefix) or module.startswith(f"_{prefix}"):
            # Generate a helpful message based on the category
            if prefix in ('_pytest', 'pytest', 'unittest'):
                msg = (
                    f"'{type(obj).__name__}' from {module} cannot be serialized.\n"
                    f"This appears to be a test framework object that leaked into scope.\n"
                    f"\n"
                    f"This usually happens when pytest fixtures are captured in the trace.\n"
                    f"Make sure you're not accidentally referencing test infrastructure."
                )
            elif prefix in ('socket', 'multiprocessing', 'asyncio', 'concurrent', 'queue', 'subprocess'):
                msg = (
                    f"'{type(obj).__name__}' from {module} cannot be serialized.\n"
                    f"This object represents OS/system resources that cannot be transferred.\n"
                    f"\n"
                    f"Options:\n"
                    f"  - Create the resource inside the trace block\n"
                    f"  - Pass configuration data and create the resource on the server"
                )
            elif prefix in ('sqlite3', 'sqlalchemy', 'pymongo', 'redis', 'psycopg', 'psycopg2', 'mysql', 'pymysql'):
                msg = (
                    f"'{type(obj).__name__}' from {module} cannot be serialized.\n"
                    f"Database connections represent network state that cannot be transferred.\n"
                    f"\n"
                    f"Options:\n"
                    f"  - Query the data before the trace and pass results as tensors/lists\n"
                    f"  - Create a new connection inside the trace block"
                )
            elif prefix == 'logging':
                msg = (
                    f"'{type(obj).__name__}' from {module} cannot be serialized.\n"
                    f"Logging handlers contain file/stream references that cannot be transferred.\n"
                    f"\n"
                    f"Create logging handlers inside the trace block if needed."
                )
            else:
                msg = (
                    f"'{type(obj).__name__}' from {module} cannot be serialized.\n"
                    f"Objects from this module are not supported for remote execution."
                )
            return True, msg

    return False, None


def check_forbidden_or_raise(name: str, obj: Any) -> None:
    """
    Check if an object is forbidden and raise SourceSerializationError if so.

    Args:
        name: Variable name (for error message)
        obj: The object to check

    Raises:
        SourceSerializationError: If the object is forbidden
    """
    is_forbidden, message = is_forbidden_for_serialization(obj)
    if is_forbidden:
        raise SourceSerializationError(
            f"Cannot serialize '{name}' for remote execution:\n\n{message}"
        )


# =============================================================================
# Disambiguation Helpers
# =============================================================================
# These functions support the hybrid disambiguation strategy described in
# disambiguation-design.md. They detect problematic patterns like nonlocal
# closures and mutable class attributes that cannot be properly serialized.

class _NonlocalFinder(ast.NodeVisitor):
    """AST visitor to find nonlocal statements in a function body."""

    def __init__(self):
        self.nonlocal_names = set()

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.nonlocal_names.update(node.names)
        self.generic_visit(node)


def _has_nonlocal_closure(func: Any) -> Tuple[bool, Set[str]]:
    """
    Check if a function uses nonlocal to capture mutable closure state.

    This pattern cannot be correctly serialized because nonlocal requires
    the captured variable to exist in an enclosing function's scope, not
    in globals.

    Args:
        func: The function to check

    Returns:
        Tuple of (has_nonlocal, set of nonlocal variable names)
    """
    try:
        source = inspect.getsource(func)
        tree = ast.parse(textwrap.dedent(source))
    except (OSError, TypeError, SyntaxError):
        return False, set()

    finder = _NonlocalFinder()
    finder.visit(tree)

    return len(finder.nonlocal_names) > 0, finder.nonlocal_names


def _check_mutable_class_attributes(cls: type) -> List[Tuple[str, str]]:
    """
    Check for mutable class-level attributes that won't survive round-trip.

    Class attributes are reset to their source-code values during deserialization.
    If they've been mutated (e.g., a list that items were appended to), those
    mutations will be lost.

    Args:
        cls: The class to check

    Returns:
        List of (attr_name, attr_type) tuples for mutable class attributes
    """
    mutable_attrs = []

    # Get class-level attributes (not instance attributes)
    for name, value in vars(cls).items():
        # Skip private/magic attributes and methods
        if name.startswith('_') or callable(value):
            continue

        # Check for mutable types
        if isinstance(value, (list, dict, set)):
            type_name = type(value).__name__
            mutable_attrs.append((name, type_name))

    return mutable_attrs


# =============================================================================
# Remote Metadata Extraction
# =============================================================================
# Helper functions to extract metadata from @remote decorated objects.
# This metadata (source code, module references, closure variables) is attached
# to objects by the @remote decorator at import time and is used during
# serialization to send the object's definition to the server.

def _get_remote_metadata(cls_or_func: Any) -> Dict[str, Any]:
    """
    Extract metadata from a @remote decorated class or function.

    This is a helper to avoid duplicating the metadata extraction logic
    in both auto_discover_class() and extract_remote_object().

    Args:
        cls_or_func: A @remote decorated class or function

    Returns:
        Dict with source, module_refs, closure_vars, type, library, version
    """
    # Get file/line info
    try:
        source_file = inspect.getfile(cls_or_func)
        _, start_line = inspect.getsourcelines(cls_or_func)
    except (OSError, TypeError):
        source_file = "<unknown>"
        start_line = 1

    return {
        "source": {
            "code": getattr(cls_or_func, '_remote_source', ''),
            "file": source_file,
            "line": start_line,
        },
        "module_refs": getattr(cls_or_func, '_remote_module_refs', {}),
        "closure_vars": getattr(cls_or_func, '_remote_closure_vars', {}),
        "type": "class" if isinstance(cls_or_func, type) else "function",
        "instances": {},
        "library": getattr(cls_or_func, '_remote_library', None),
        "version": getattr(cls_or_func, '_remote_version', None),
    }


# =============================================================================
# Auto-Discovery of Dependencies
# =============================================================================
# Functions for automatically discovering classes that need to be serialized,
# even when they are not explicitly decorated with @remote. This enables
# third-party libraries (like nnterp) to work with NDIF without modification.
#
# Auto-discovery works by:
# 1. Detecting when a value's class has available source code (via inspect)
# 2. Extracting the source and finding external references (like @remote does)
# 3. Recursively discovering base classes and type dependencies
# 4. Including discovered classes in the serialized payload for server-side
#    reconstruction
#
# This is triggered at serialization time (not import time) when we encounter
# instances of classes that aren't @remote decorated but have available source.

# Cache of auto-discovered classes to avoid re-processing
_auto_discovered_cache: Dict[type, Dict[str, Any]] = {}


def can_auto_discover(cls: type) -> bool:
    """
    Check if a class can be auto-discovered for remote serialization.

    A class can be auto-discovered if:
    1. It has available source code (via inspect.getsource)
    2. It's not from a core allowed module (torch, numpy, etc.)
    3. It's not already @remote decorated
    """
    # Already decorated
    if getattr(cls, '_remote_validated', False):
        return True  # Use existing metadata

    # Check module - skip core allowed modules
    module = getattr(cls, '__module__', '')
    if module:
        # Skip server-available modules (torch, numpy, etc.)
        # But allow nnsight.modeling subclasses (they need serialization)
        if is_server_available_module(module):
            if not ('nnsight' in module and 'modeling' in module):
                return False

    # Check if source is available
    try:
        inspect.getsource(cls)
        return True
    except (OSError, TypeError):
        return False


def auto_discover_class(cls: type, discovered: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Auto-discover a class for remote serialization.

    This does the same work as @remote but at serialization time,
    allowing third-party classes (like nnterp's LayerAccessor) to be
    automatically serialized without requiring decoration.

    Args:
        cls: The class to discover
        discovered: Dict to accumulate discovered classes (for recursion)

    Returns:
        Dict with source, module_refs, and type info
    """
    if discovered is None:
        discovered = {}

    cls_name = cls.__name__
    cls_module = cls.__module__
    cls_qualified = f"{cls_module}.{cls_name}"

    # Check cache first
    if cls in _auto_discovered_cache:
        cached = _auto_discovered_cache[cls]
        if cls_qualified not in discovered:
            discovered[cls_qualified] = cached
        return cached

    # Already discovered this exact class
    if cls_qualified in discovered:
        return discovered[cls_qualified]

    # Already @remote decorated - use existing metadata
    if getattr(cls, '_remote_validated', False):
        result = _get_remote_metadata(cls)
        result['qualified_name'] = cls_qualified
        _auto_discovered_cache[cls] = result
        discovered[cls_qualified] = result
        return result

    # Get file/line info first (for error messages)
    try:
        source_file = inspect.getfile(cls)
        _, start_line = inspect.getsourcelines(cls)
        location = f"{source_file}:{start_line}"
    except (OSError, TypeError):
        source_file = None
        start_line = None
        location = None

    # Extract source
    try:
        source = inspect.getsource(cls)
        dedented_source = textwrap.dedent(source)
    except (OSError, TypeError) as e:
        prefix = f"{location}: " if location else ""
        raise SourceSerializationError(
            f"{prefix}cannot auto-discover class '{cls_name}': source not available. {e}"
        )

    # Parse AST and find external references
    try:
        tree = ast.parse(dedented_source)
    except SyntaxError as e:
        prefix = f"{location}: " if location else ""
        raise SourceSerializationError(
            f"{prefix}cannot parse source for '{cls_name}': {e}"
        )

    # Find external references
    external_names = find_external_references(tree, cls)

    # Get globals to resolve references
    module_globals = {}
    if hasattr(cls, '__module__'):
        module = sys.modules.get(cls.__module__)
        module_globals = getattr(module, '__dict__', {}) if module else {}

    # Resolve module references, but also track type references for auto-discovery
    module_refs, resolution_errors = resolve_module_references(external_names, cls, source_file, start_line)

    # Find type references that need auto-discovery
    types_to_discover = []
    filtered_errors = []
    for error in resolution_errors:
        # Check if error is about a type reference we can auto-discover
        # Error format: "Reference 'X' (type 'X' from module 'Y') is not @nnsight.remote..."
        if "is not @nnsight.remote decorated" in error:
            # Try to find the type in globals
            for name in external_names:
                if name in module_globals:
                    value = module_globals[name]
                    if isinstance(value, type) and can_auto_discover(value):
                        types_to_discover.append(value)
                        break
            else:
                filtered_errors.append(error)
        else:
            filtered_errors.append(error)

    # Validate AST (warnings only for auto-discovered - don't block)
    ast_errors = validate_ast(tree, cls_name, source_file, start_line)

    # For auto-discovered classes, we're more lenient - just warn
    if filtered_errors or ast_errors:
        import warnings
        all_errors = filtered_errors + ast_errors
        if all_errors:
            warnings.warn(
                f"Auto-discovered class '{cls_name}' has potential issues:\n" +
                "\n".join(f"  - {e}" for e in all_errors[:3])
            )

    # Check for mutable class attributes (see disambiguation-design.md)
    # These are reset to source values on deserialization, so mutations are lost
    mutable_attrs = _check_mutable_class_attributes(cls)
    if mutable_attrs:
        import warnings
        attrs_str = ', '.join(f"'{name}' (type: {type_name})" for name, type_name in mutable_attrs)
        warnings.warn(
            f"Class '{cls_name}' has mutable class attribute(s): {attrs_str}.\n"
            f"Class attributes are reset to source values during remote execution.\n"
            f"Current values will not be preserved. Consider using module-level state instead."
        )

    # Auto-detect library/version
    library = None
    version = None
    if hasattr(cls, '__module__') and cls.__module__:
        root_package = cls.__module__.split('.')[0]
        try:
            from importlib.metadata import version as get_version
            version = get_version(root_package)
            library = root_package
        except Exception:
            pass

    result = {
        "source": {
            "code": dedented_source,
            "file": source_file,
            "line": start_line,
        },
        "module_refs": module_refs,
        "closure_vars": {},
        "type": "class",
        "instances": {},
        "library": library,
        "version": version,
        "qualified_name": cls_qualified,  # For collision detection
    }

    # Cache and record using qualified name
    _auto_discovered_cache[cls] = result
    discovered[cls_qualified] = result

    # Recursively discover base classes (except object and allowed bases)
    for base in cls.__bases__:
        if base is object:
            continue
        base_fullname = f"{base.__module__}.{base.__name__}"
        if base_fullname in ALLOWED_BASE_CLASSES:
            continue
        if can_auto_discover(base):
            auto_discover_class(base, discovered)

    # Recursively discover type references found in the source
    for dep_type in types_to_discover:
        dep_qualified = f"{dep_type.__module__}.{dep_type.__name__}"
        if dep_qualified not in discovered:
            auto_discover_class(dep_type, discovered)

    return result


def is_auto_discoverable_instance(value: Any) -> bool:
    """Check if a value is an instance of an auto-discoverable class."""
    if value is None:
        return False
    cls = type(value)
    # Skip basic types
    if cls.__module__ == 'builtins':
        return False
    return can_auto_discover(cls)


# =============================================================================
# Tensor Serialization
# =============================================================================
# Functions for serializing PyTorch tensors and NumPy arrays to JSON-compatible
# format. Tensors are converted to base64-encoded bytes with dtype/shape metadata.
#
# Special tensor types are handled:
# - Sparse tensors: Preserved in COO format to maintain memory efficiency
# - Quantized tensors: Preserved with scale/zero_point to maintain precision
# - GPU/MPS tensors: Moved to CPU before serialization
# - bfloat16 tensors: View-cast to int16 (same bits) since NumPy lacks bfloat16
#
# Note: Per-tensor compression is disabled because the full JSON payload is
# compressed at the transport layer, making per-tensor compression redundant.

def is_tensor(value: Any) -> bool:
    """
    Check if value is a tensor (torch.Tensor or numpy.ndarray).

    Uses string-based type checking to avoid import dependencies.
    """
    type_name = type(value).__name__
    module = getattr(type(value), '__module__', '')

    # torch.Tensor
    if type_name == 'Tensor' and 'torch' in module:
        return True

    # numpy.ndarray
    if type_name == 'ndarray' and 'numpy' in module:
        return True

    return False


def _can_deep_serialize(value: Any, _seen: set = None) -> bool:
    """
    Check if a value (including nested collections) can be serialized.

    This is more permissive than is_json_serializable - it allows:
    - JSON primitives (None, bool, int, float, str)
    - Tensors (torch.Tensor, numpy.ndarray)
    - Collections containing the above (list, tuple, dict, set)
    - @remote decorated objects
    - Auto-discoverable instances

    Returns True if the value can be serialized via serialize_value.
    """
    # Handle circular references
    if _seen is None:
        _seen = set()

    # Primitives are always serializable
    if value is None or isinstance(value, (bool, int, float, str)):
        return True

    # Check for circular reference
    obj_id = id(value)
    if obj_id in _seen:
        return True  # We handle circular refs via memo
    _seen.add(obj_id)

    # Tensors are serializable
    if is_tensor(value):
        return True

    # @remote objects are serializable
    if is_remote_object(value):
        return True

    # Auto-discoverable instances are serializable
    if is_auto_discoverable_instance(value):
        return True

    # Collections - recursively check all elements
    if isinstance(value, (list, tuple)):
        return all(_can_deep_serialize(item, _seen) for item in value)

    if isinstance(value, dict):
        return all(
            isinstance(k, str) and _can_deep_serialize(v, _seen)
            for k, v in value.items()
        )

    if isinstance(value, set):
        return all(_can_deep_serialize(item, _seen) for item in value)

    # Enum values are serializable
    from enum import Enum
    if isinstance(value, Enum):
        return True

    # nn.Module instances are serializable (check by type name to avoid circular dependency)
    type_name = type(value).__name__
    module = getattr(type(value), '__module__', '')
    if 'torch.nn' in module or (hasattr(value, 'forward') and hasattr(value, 'parameters')):
        return True

    return False


def serialize_tensor(value: Any) -> Dict[str, Any]:
    """
    Serialize a tensor to a JSON-compatible dict.

    Strategy:
    - Convert to numpy bytes (handling sparse, quantized, device-specific tensors)
    - Base64 encode for JSON transport
    - Note: Per-tensor compression is disabled since the full message is compressed

    Handles special cases:
    - Sparse tensors: preserved in COO format (memory efficient)
    - Quantized tensors: preserved with quantization parameters (smaller, exact)
    - GPU/MPS tensors: moved to CPU first

    Returns:
        Dict with keys:
        - __tensor__: base64-encoded bytes
        - dtype: string dtype (e.g., "float32")
        - shape: list of dimensions
        - quantization: (optional) dict with scale, zero_point, qtype for quantized tensors
        - sparse: (optional) dict with indices, dense_shape for sparse COO tensors

    Wire format example:
        {"__tensor__": "base64...", "dtype": "float32", "shape": [768]}

    Wire format example (quantized - preserves exact int8 values, 4x smaller):
        {"__tensor__": "base64...", "dtype": "int8", "shape": [768],
         "quantization": {"scale": 0.1, "zero_point": 0, "qtype": "qint8"}}

    Wire format example (sparse COO - memory efficient for large sparse tensors):
        {"__tensor__": "base64...", "dtype": "float32", "shape": [nnz],
         "sparse": {"indices": "base64...", "indices_shape": [2, nnz], "dense_shape": [1000, 1000]}}
    """
    import numpy as np

    quantization_info = None
    sparse_info = None

    original_dtype = None  # Track special dtypes like bfloat16

    # Convert torch.Tensor to numpy if needed
    if hasattr(value, 'detach'):
        # It's a torch tensor
        t = value.detach()

        # Reject nested tensors (complex structure, not supported)
        if hasattr(t, 'is_nested') and t.is_nested:
            raise SourceSerializationError(
                "Nested tensors cannot be serialized. "
                "Convert to a list of regular tensors first."
            )

        # Handle sparse tensors - preserve sparsity!
        if hasattr(t, 'is_sparse') and t.is_sparse:
            t = t.coalesce()  # Ensure indices are unique and sorted
            indices = t._indices().cpu().numpy()
            values = t._values().cpu().numpy()

            # Encode indices (no per-tensor compression; message is compressed globally)
            indices_bytes = indices.tobytes()
            indices_data = base64.b64encode(indices_bytes).decode('ascii')

            sparse_info = {
                "indices": indices_data,
                "indices_dtype": str(indices.dtype),
                "indices_shape": list(indices.shape),
                "dense_shape": list(t.shape),
            }
            np_array = values

        # Handle quantized tensors - preserve quantization!
        elif hasattr(t, 'is_quantized') and t.is_quantized:
            # Extract quantization parameters
            quantization_info = {
                "scale": float(t.q_scale()),
                "zero_point": int(t.q_zero_point()),
                "qtype": str(t.dtype).replace("torch.", ""),  # e.g., "qint8"
            }
            # Get the underlying integer representation
            np_array = t.int_repr().cpu().numpy()
        else:
            # Move to CPU and convert
            # Handle bfloat16: view as int16 (same bit pattern, has numpy dtype)
            import torch as _torch
            original_dtype = str(t.dtype).replace("torch.", "")
            if t.dtype == _torch.bfloat16:
                np_array = t.view(_torch.int16).cpu().numpy()
            else:
                np_array = t.cpu().numpy()
                original_dtype = None  # Only set for special dtypes
    else:
        # Already numpy
        np_array = value
        original_dtype = None

    # Get raw bytes and base64 encode (no per-tensor compression; message is compressed globally)
    raw_bytes = np_array.tobytes()
    b64_data = base64.b64encode(raw_bytes).decode('ascii')

    result = {
        TENSOR_MARKER: b64_data,
        "dtype": str(np_array.dtype),
        "shape": list(np_array.shape),
    }

    # Add quantization info if present
    if quantization_info is not None:
        result["quantization"] = quantization_info

    # Add sparse info if present
    if sparse_info is not None:
        result["sparse"] = sparse_info

    # Add original dtype if different from numpy dtype (e.g., bfloat16)
    if original_dtype is not None:
        result["torch_dtype"] = original_dtype

    return result


def deserialize_tensor(data: Dict[str, Any], as_torch: bool = True) -> Any:
    """
    Deserialize a tensor from JSON payload.

    Args:
        data: Dict with __tensor__, dtype, shape, compressed keys
              Optional "quantization" dict for quantized tensors
              Optional "sparse" dict for sparse COO tensors
        as_torch: If True, return torch.Tensor; else return numpy.ndarray

    Returns:
        Reconstructed tensor (quantized/sparse if original was)
    """
    import numpy as np

    # Decode base64 for values
    raw_bytes = base64.b64decode(data[TENSOR_MARKER])

    # Reconstruct numpy array (values for sparse, or full tensor for dense)
    dtype = np.dtype(data["dtype"])
    shape = tuple(data["shape"])
    np_array = np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)

    # Convert to torch if requested
    if as_torch:
        import torch
        values_tensor = torch.from_numpy(np_array.copy())  # copy() to own the memory

        # Restore original torch dtype if different from numpy dtype (e.g., bfloat16)
        if "torch_dtype" in data:
            dtype_map = {
                "bfloat16": torch.bfloat16,
            }
            target_dtype = dtype_map.get(data["torch_dtype"])
            if target_dtype is not None:
                values_tensor = values_tensor.view(target_dtype)

        # Reconstruct sparse tensor if original was sparse
        if "sparse" in data:
            s = data["sparse"]
            # Decode indices
            indices_bytes = base64.b64decode(s["indices"])
            indices_dtype = np.dtype(s["indices_dtype"])
            indices_shape = tuple(s["indices_shape"])
            indices_np = np.frombuffer(indices_bytes, dtype=indices_dtype).reshape(indices_shape)
            indices_tensor = torch.from_numpy(indices_np.copy())

            # Create sparse COO tensor
            dense_shape = tuple(s["dense_shape"])
            tensor = torch.sparse_coo_tensor(indices_tensor, values_tensor, dense_shape)
            return tensor.coalesce()

        # Re-quantize if original was quantized
        if "quantization" in data:
            q = data["quantization"]
            # Map qtype string to torch dtype
            qtype_map = {
                "qint8": torch.qint8,
                "quint8": torch.quint8,
                "qint32": torch.qint32,
            }
            qtype = qtype_map.get(q["qtype"], torch.qint8)
            # Re-quantize using the original parameters
            values_tensor = torch._make_per_tensor_quantized_tensor(
                values_tensor, scale=q["scale"], zero_point=q["zero_point"]
            )

        return values_tensor

    return np_array


def serialize_source_based(
    tracer: "Tracer",
    strict_remote: bool = False,
    max_upload_mb: float = 10.0,
) -> bytes:
    """
    Serialize a tracer for remote execution using source + JSON.

    This is the new serialization format that doesn't require matching
    Python versions between client and server.

    Args:
        tracer: The tracer object containing source and frame information
        strict_remote: If True, require explicit @remote decorations for all
                user-defined functions/classes. If False (default), auto-discover
                classes/functions with available source code.
        max_upload_mb: Maximum upload payload size in MB before warning. Default is
                       10 MB. Set to 0 to disable warnings.

    Returns:
        JSON-encoded bytes ready for transmission

    Raises:
        SourceSerializationError: If any variable cannot be serialized
    """
    source = tracer.info.source
    frame = tracer.info.frame
    frame_locals = frame.f_locals if frame else {}

    # Find the traced model from frame locals (for identity-based model ref detection)
    traced_model = None
    for value in frame_locals.values():
        if is_model_reference(value):
            traced_model = value
            break

    # Extract file/line metadata for error mapping
    source_metadata = extract_source_metadata(tracer)

    # Extract variables and remote objects
    variables, remote_objects, model_refs = extract_all(
        frame_locals, traced_model=traced_model, strict_remote=strict_remote
    )

    payload = {
        "version": SERIALIZATION_VERSION,
        "source": source_metadata,  # Now includes file/line info
        "variables": variables,
        "remote_objects": remote_objects,
        "model_refs": model_refs,
    }

    # Check if the model is a LanguageModel subclass that needs its source sent
    if traced_model is not None and is_languagemodel_subclass(traced_model):
        try:
            model_subclass_data = serialize_model_subclass(traced_model)
            payload["model_subclass"] = model_subclass_data
        except SourceSerializationError:
            # If we can't auto-discover the subclass, fall back to just using model_key
            # The server will create a plain LanguageModel with the rename dict
            pass

    # Serialize to JSON bytes
    result = json.dumps(payload).encode('utf-8')

    # Check upload payload size against threshold
    if max_upload_mb > 0:
        payload_mb = len(result) / (1024 * 1024)
        if payload_mb > max_upload_mb:
            import warnings
            warnings.warn(
                f"Upload payload size ({payload_mb:.2f} MB) exceeds threshold "
                f"({max_upload_mb} MB). Large uploads may cause slow transmission "
                f"or NDIF submission failures. Consider reducing tensor sizes or "
                f"computing values server-side. Use max_upload_mb=0 to disable.",
                UserWarning,
                stacklevel=3,  # Point to caller's caller (trace block)
            )

    return result


def extract_source_metadata(tracer: "Tracer") -> Dict[str, Any]:
    """
    Extract source code with file/line metadata for server-side error mapping.

    Args:
        tracer: The tracer containing source and frame information

    Returns:
        Dict containing:
        - code: The source code string
        - file: Original filename
        - line: Starting line number in the original file
        - remote_objects: List of remote object sources with their metadata
    """
    source = tracer.info.source
    frame = tracer.info.frame

    # Get file and line info from frame
    filename = "<unknown>"
    start_line = 1

    if frame:
        filename = frame.f_code.co_filename or "<unknown>"
        # The line number where the trace block starts
        start_line = frame.f_lineno

    # Build the main source block with metadata
    result = {
        "code": source if isinstance(source, str) else "".join(source),
        "file": filename,
        "line": start_line,
    }

    return result


# =============================================================================
# Auto-Discovery Helper Functions
# =============================================================================
# These functions handle auto-discovery of classes and functions that don't have
# @remote decoration but have available source code. This enables third-party
# libraries to work without requiring explicit decoration.

def _can_auto_discover_function(func: Any) -> bool:
    """
    Check if a function can be auto-discovered for remote serialization.

    A function can be auto-discovered if:
    1. It has available source code (via inspect.getsource)
    2. It's not from a core allowed module (torch, numpy, etc.)
    3. It's not already @remote decorated
    """
    # Already decorated
    if getattr(func, '_remote_validated', False):
        return True  # Use existing metadata

    # Skip lambdas (handled separately)
    if is_lambda(func):
        return False

    # Check module - skip core allowed modules
    module = getattr(func, '__module__', '')
    if module and is_server_available_module(module):
        return False

    # Check if source is available
    try:
        inspect.getsource(func)
        return True
    except (OSError, TypeError):
        return False


def _auto_discover_function(func: Any) -> Dict[str, Any]:
    """
    Auto-discover a function for remote serialization.

    Similar to auto_discover_class but for standalone functions.
    Uses fully qualified names for disambiguation.
    """
    func_name = func.__name__
    func_module = getattr(func, '__module__', '__main__')
    func_qualified = f"{func_module}.{func_name}"

    # Check for nonlocal closures (forbidden - see disambiguation-design.md)
    has_nonlocal, nonlocal_names = _has_nonlocal_closure(func)
    if has_nonlocal:
        names_str = ', '.join(sorted(nonlocal_names))
        raise SourceSerializationError(
            f"Function '{func_name}' captures mutable closure variable(s) "
            f"'{names_str}' via nonlocal. This cannot be serialized for remote execution.\n"
            f"\n"
            f"Workaround: Refactor to use a class with instance state:\n"
            f"    class Counter:\n"
            f"        def __init__(self):\n"
            f"            self.count = 0\n"
            f"        def increment(self):\n"
            f"            self.count += 1\n"
            f"            return self.count"
        )

    # Get file/line info
    try:
        source_file = inspect.getfile(func)
        source_lines, start_line = inspect.getsourcelines(func)
    except (OSError, TypeError):
        source_file = "<unknown>"
        start_line = 1

    # Get source
    try:
        source = inspect.getsource(func)
        dedented_source = textwrap.dedent(source)
    except (OSError, TypeError) as e:
        raise SourceSerializationError(
            f"Cannot auto-discover function '{func_name}': source not available. {e}"
        )

    # Parse AST and find external references
    try:
        tree = ast.parse(dedented_source)
    except SyntaxError as e:
        raise SourceSerializationError(
            f"Cannot parse source for function '{func_name}': {e}"
        )

    # Find external references
    external_names = find_external_references(tree, func)

    # Get module globals
    module_globals = {}
    if hasattr(func, '__module__'):
        module = sys.modules.get(func.__module__)
        module_globals = getattr(module, '__dict__', {}) if module else {}

    # Resolve module references
    module_refs, resolution_errors = resolve_module_references(
        external_names, func, source_file, start_line
    )

    # Get closure variables (JSON-serializable only)
    closure_vars = {}
    if func.__closure__ and hasattr(func.__code__, 'co_freevars'):
        for name, cell in zip(func.__code__.co_freevars, func.__closure__):
            try:
                value = cell.cell_contents
            except ValueError:
                continue

            if is_json_serializable(value):
                closure_vars[name] = value
            elif isinstance(value, types.ModuleType) and is_server_available_module(value.__name__):
                continue  # Skip - available on server
            elif is_remote_object(value):
                continue  # Skip - serialized separately

    # Auto-detect library/version
    library = None
    version = None
    if hasattr(func, '__module__') and func.__module__:
        root_package = func.__module__.split('.')[0]
        try:
            from importlib.metadata import version as get_version
            version = get_version(root_package)
            library = root_package
        except Exception:
            pass

    # Track __globals__ identity for namespace grouping (see disambiguation-design.md)
    globals_id = None
    if hasattr(func, '__globals__'):
        globals_id = id(func.__globals__)

    return {
        "source": {
            "code": dedented_source,
            "file": source_file,
            "line": start_line,
        },
        "module_refs": module_refs,
        "closure_vars": closure_vars,
        "type": "function",
        "library": library,
        "version": version,
        "qualified_name": func_qualified,
        "globals_id": globals_id,  # For namespace group detection
    }


def extract_auto_discovered_function(var_name: str, func: Any, result: Dict[str, Any]) -> None:
    """
    Extract an auto-discovered function for serialization.

    Uses fully qualified names as keys to avoid collisions between
    functions with the same short name from different modules.
    """
    func_module = getattr(func, '__module__', '__main__')
    func_qualified = f"{func_module}.{func.__name__}"

    if func_qualified not in result:
        result[func_qualified] = _auto_discover_function(func)


def extract_auto_discovered_type(var_name: str, cls: type, result: Dict[str, Any]) -> None:
    """
    Extract an auto-discovered class type (not instance) for serialization.

    Uses fully qualified names as keys to avoid collisions between
    classes with the same short name from different modules.
    """
    # Use qualified name as key to prevent collisions
    cls_qualified = f"{cls.__module__}.{cls.__name__}"

    if cls_qualified not in result:
        discovered = {}
        auto_discover_class(cls, discovered)
        result.update(discovered)


def extract_auto_discovered_object(
    var_name: str,
    value: Any,
    result: Dict[str, Any],
    traced_model: Any = None
) -> None:
    """
    Extract an auto-discovered object instance for serialization.

    Similar to extract_remote_object but for non-@remote classes.
    """
    cls = type(value)
    # Use qualified name as key to prevent collisions between classes
    # with the same short name from different modules
    cls_qualified = f"{cls.__module__}.{cls.__name__}"

    # Auto-discover the class if not already in result
    if cls_qualified not in result:
        discovered = {}
        auto_discover_class(cls, discovered)
        result.update(discovered)

    # Serialize instance state
    instance_state = serialize_instance_state(value, discovered_classes=result, traced_model=traced_model)
    if "instances" not in result[cls_qualified]:
        result[cls_qualified]["instances"] = {}
    result[cls_qualified]["instances"][str(id(value))] = {
        "var_name": var_name,
        "state": instance_state
    }


def extract_all(
    locals_dict: Dict[str, Any],
    traced_model: Any = None,
    strict_remote: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """
    Extract all serializable data from locals.

    Args:
        locals_dict: Dictionary of local variables
        traced_model: The model being traced (for identity-based model ref detection)
        strict_remote: If True, require explicit @remote decorations. If False (default),
                       auto-discover classes/functions with available source code.

    Returns:
        (variables, remote_objects, model_refs) where:
        - variables: JSON-serializable simple values
        - remote_objects: @nnsight.remote functions/classes/instances (or auto-discovered)
        - model_refs: list of variable names that reference the model
    """
    variables = {}
    remote_objects = {}
    model_refs = []

    for name, value in locals_dict.items():
        # Skip dunder names
        if name.startswith('__'):
            continue

        # Skip nnsight internal objects
        if name.startswith('_nnsight') or name.startswith('nnsight'):
            continue

        # Check for model references (Envoy or NNsight types)
        if is_model_reference(value):
            model_refs.append(name)
            continue

        # Check for @nnsight.remote decorated functions/classes or instances
        if is_remote_object(value):
            extract_remote_object(name, value, remote_objects, traced_model=traced_model)
            continue

        # Check for lambda functions (extract and validate)
        if is_lambda(value):
            extract_lambda_object(name, value, remote_objects)
            continue

        # Check for tensors (torch.Tensor or numpy.ndarray)
        if is_tensor(value):
            variables[name] = serialize_tensor(value)
            continue

        # Try JSON serialization for simple values
        if is_json_serializable(value):
            variables[name] = value
            continue

        # Handle collections (list, tuple, dict, set) that might contain tensors
        # These need special serialization via serialize_value
        if isinstance(value, (list, tuple, dict, set)):
            if _can_deep_serialize(value):
                memo = {}
                discovered = {}
                variables[name] = serialize_value(value, name, memo, discovered, traced_model)
                # If any classes were auto-discovered during serialization, add them
                remote_objects.update(discovered)
                continue

        # Skip module references (they'll be available on server)
        if isinstance(value, types.ModuleType):
            if is_server_available_module(value.__name__):
                continue
            raise SourceSerializationError(
                f"Variable '{name}' references module '{value.__name__}' "
                f"which is not available on NDIF server."
            )

        # Skip functions from allowed modules
        if callable(value) and hasattr(value, '__module__'):
            if is_server_available_module(value.__module__ or ''):
                continue

        # Skip type references from builtins
        if isinstance(value, type) and value.__module__ == 'builtins':
            continue

        # EARLY CHECK: Reject forbidden objects before any auto-discovery attempt
        # This catches common mistakes (pandas, matplotlib, sockets) with clear messages
        check_forbidden_or_raise(name, value)

        # Auto-discovery mode (strict_remote=False): try to auto-discover classes/functions
        if not strict_remote:
            # Try to auto-discover instances of classes with available source
            if is_auto_discoverable_instance(value):
                extract_auto_discovered_object(name, value, remote_objects, traced_model=traced_model)
                continue

            # Try to auto-discover class types (not instances)
            if isinstance(value, type) and can_auto_discover(value):
                extract_auto_discovered_type(name, value, remote_objects)
                continue

            # Try to auto-discover regular functions with available source
            if callable(value) and not isinstance(value, type):
                if _can_auto_discover_function(value):
                    extract_auto_discovered_function(name, value, remote_objects)
                    continue

        # If we get here, we can't serialize this value
        if strict_remote:
            raise SourceSerializationError(
                f"Variable '{name}' of type '{type(value).__name__}' "
                f"cannot be serialized for source-based remote execution.\n"
                f"Options:\n"
                f"  - Use JSON-serializable type (int, float, str, list, dict)\n"
                f"  - Mark functions/classes with @nnsight.remote\n"
                f"  - Compute the value inside the trace block"
            )
        else:
            raise SourceSerializationError(
                f"Variable '{name}' of type '{type(value).__name__}' "
                f"cannot be serialized for source-based remote execution.\n"
                f"The source code for this object is not available for auto-discovery.\n"
                f"Options:\n"
                f"  - Use JSON-serializable type (int, float, str, list, dict)\n"
                f"  - Mark functions/classes with @nnsight.remote\n"
                f"  - Compute the value inside the trace block"
            )

    return variables, remote_objects, model_refs


def is_model_reference(value: Any) -> bool:
    """Check if value is a reference to the model (Envoy or NNsight types).

    Note: This is a type-based heuristic used in extract_all for frame locals.
    For precise model identity checking during serialization, use is_the_traced_model().
    """
    # Check by type name to avoid circular imports
    type_name = type(value).__name__
    if type_name in ('Envoy', 'NNsight', 'LanguageModel'):
        return True

    # Check module path
    module = getattr(type(value), '__module__', '')
    if module and 'nnsight' in module:
        # Check if it's an intervention/tracing type
        if 'envoy' in module.lower() or 'modeling' in module.lower():
            return True

    return False


# =============================================================================
# Model Subclass Handling
# =============================================================================
# Functions for serializing LanguageModel subclasses so they can be reconstructed
# on the NDIF server. This enables third-party model wrappers (like nnterp's
# StandardizedTransformer) to work on servers that don't have the library installed.
#
# When a user traces a LanguageModel subclass:
# 1. is_languagemodel_subclass() detects it's not a base nnsight class
# 2. serialize_model_subclass() auto-discovers the class hierarchy
# 3. The class source, dependencies, and instance state are serialized
# 4. reconstruct_model_subclass() rebuilds the class on the server
#
# This differs from regular @remote classes because:
# - The model is the "context" for the trace, not just a variable
# - Server-provided attributes (_module, _tokenizer) must be preserved
# - The model_key is used to load the HuggingFace model on the server

def is_languagemodel_subclass(value: Any) -> bool:
    """Check if value is an instance of a LanguageModel SUBCLASS (not LanguageModel itself).

    This is used to detect third-party model wrappers like nnterp's StandardizedTransformer
    that need their class source sent to the server.
    """
    cls = type(value)
    cls_name = cls.__name__

    # Not a subclass if it's exactly LanguageModel or a base nnsight class
    if cls_name in ('LanguageModel', 'TransformersModel', 'HuggingFaceModel', 'NNsight', 'Envoy'):
        return False

    # Check if it inherits from a nnsight modeling class
    module = getattr(cls, '__module__', '')

    # Walk the MRO to see if any base is a nnsight modeling class
    for base in cls.__mro__[1:]:  # Skip the class itself
        base_module = getattr(base, '__module__', '')
        if base_module and 'nnsight' in base_module and 'modeling' in base_module:
            return True

    return False


def is_nnsight_class(cls: type) -> bool:
    """Check if a class is part of nnsight (and thus available on server)."""
    module = getattr(cls, '__module__', '')
    return module and 'nnsight' in module


def is_server_available_class(cls: type) -> bool:
    """Check if a class is from a module that's available on the server."""
    module = getattr(cls, '__module__', '')
    if not module:
        return True  # Unknown module, assume available
    return is_server_available_module(module)


def auto_discover_model_subclass(cls: type, discovered: Dict[str, Any]) -> None:
    """
    Auto-discover a LanguageModel subclass for remote serialization.

    Unlike auto_discover_class, this function:
    - Only discovers the subclass itself (not nnsight base classes)
    - Discovers non-nnsight dependencies (like nnterp helper classes)
    - Stops recursion at nnsight/server-available classes

    Args:
        cls: The LanguageModel subclass to discover
        discovered: Dict to accumulate discovered classes
    """
    import inspect

    cls_name = cls.__name__

    # Skip if already discovered (or in progress)
    if cls_name in discovered:
        return

    # Skip classes that are already available on the server
    if is_server_available_class(cls):
        return

    # Mark as in progress to prevent cycles
    discovered[cls_name] = None

    # Get the source
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        # Can't get source, skip this class
        return

    # Get file/line metadata
    try:
        source_file = inspect.getfile(cls)
        source_lines, start_line = inspect.getsourcelines(cls)
    except (OSError, TypeError):
        source_file = "<unknown>"
        start_line = 1

    # Find module-level references in the source
    module_refs = {}
    external_names = set()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None

    if tree:
        # Find names used in the class that aren't defined locally
        # Uses the unified ReferenceCollector which properly handles scopes,
        # decorators, default arguments, and annotations
        external_names = find_external_references(tree, cls)

    # Resolve external names from the class's module
    # Track server-available imports so they can be resolved during reconstruction
    server_imports = {}

    cls_module = sys.modules.get(cls.__module__)
    if cls_module:
        for name in external_names:
            if hasattr(cls_module, name):
                value = getattr(cls_module, name)

                # Record server-available types with their module path
                if isinstance(value, type) and is_server_available_class(value):
                    type_module = getattr(value, '__module__', '')
                    type_name = value.__name__
                    if type_module:
                        server_imports[name] = {"type": "class", "module": type_module, "name": type_name}
                    continue

                # Record server-available functions/decorators (like dataclass, abstractmethod)
                if callable(value) and hasattr(value, '__module__'):
                    func_module = getattr(value, '__module__', '')
                    if func_module and is_server_available_module(func_module):
                        func_name = getattr(value, '__name__', name)
                        server_imports[name] = {"type": "callable", "module": func_module, "name": func_name}
                        continue

                # Record server-available modules
                if isinstance(value, types.ModuleType):
                    mod_name = value.__name__
                    if is_server_available_module(mod_name):
                        server_imports[name] = {"type": "module", "module": mod_name}
                        continue

                # Capture JSON-serializable values
                if value is None or isinstance(value, (bool, int, float, str)):
                    module_refs[name] = value
                elif isinstance(value, (list, tuple, dict)):
                    try:
                        json.dumps(value)
                        module_refs[name] = value
                    except (TypeError, ValueError):
                        pass

                # Discover non-server-available type dependencies
                if isinstance(value, type) and not is_server_available_class(value):
                    if can_auto_discover(value):
                        auto_discover_model_subclass(value, discovered)

    # Add this class to discovered
    discovered[cls_name] = {
        "source": {
            "code": source,
            "file": source_file,
            "line": start_line,
        },
        "module_refs": module_refs,
        "server_imports": server_imports,
        "type": "class",
    }

    # Discover non-server-available base classes
    for base in cls.__bases__:
        if base is object:
            continue
        if not is_server_available_class(base) and can_auto_discover(base):
            auto_discover_model_subclass(base, discovered)


def serialize_model_subclass(model: Any, discovered_classes: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Serialize a LanguageModel subclass for remote reconstruction.

    This enables third-party model wrappers (like nnterp's StandardizedTransformer)
    to work on servers that don't have the library installed.

    Args:
        model: The LanguageModel subclass instance
        discovered_classes: Dict to accumulate auto-discovered classes

    Returns:
        Dict containing:
        - class_name: Name of the subclass
        - source: Auto-discovered class source with dependencies
        - state: Serialized non-server-provided instance state
        - model_key: The model key for server-side model creation
    """
    if discovered_classes is None:
        discovered_classes = {}

    cls = type(model)
    cls_name = cls.__name__

    # Check if class can be auto-discovered
    if not can_auto_discover(cls):
        raise SourceSerializationError(
            f"LanguageModel subclass '{cls_name}' cannot be auto-discovered for remote execution. "
            f"Ensure the class source is available (in a .py file, not dynamically generated)."
        )

    # Auto-discover ONLY the subclass itself (not nnsight base classes)
    # nnsight classes are already available on the server
    auto_discover_model_subclass(cls, discovered_classes)

    # Get server-provided attributes from the class hierarchy
    server_provided = getattr(cls, '_server_provided', frozenset())

    # Serialize non-server-provided instance state
    custom_state = {}
    for key, value in model.__dict__.items():
        if key in server_provided:
            # Mark as server-provided (will be substituted on server)
            custom_state[key] = {SERVER_PROVIDED_MARKER: True}
        elif key.startswith('_'):
            # Skip private attributes (likely internal state)
            custom_state[key] = {SERVER_PROVIDED_MARKER: True}
        else:
            # Try to serialize the value
            try:
                # Check if JSON-serializable
                if value is None or isinstance(value, (bool, int, float, str)):
                    custom_state[key] = value
                elif isinstance(value, (list, tuple)):
                    json.dumps(value)  # Validate
                    custom_state[key] = list(value) if isinstance(value, tuple) else value
                elif isinstance(value, dict):
                    json.dumps(value)  # Validate
                    custom_state[key] = value
                else:
                    # Non-serializable, mark as server-provided
                    custom_state[key] = {SERVER_PROVIDED_MARKER: True}
            except (TypeError, ValueError):
                custom_state[key] = {SERVER_PROVIDED_MARKER: True}

    # Get the model key for server-side model creation
    model_key = None
    if hasattr(model, '_remoteable_model_key'):
        model_key = model._remoteable_model_key()

    return {
        "class_name": cls_name,
        "discovered_classes": discovered_classes,
        "state": custom_state,
        "model_key": model_key,
    }


def reconstruct_model_subclass(
    subclass_data: Dict[str, Any],
    server_model: Any,
    namespace: dict,
    exec_func: callable,
) -> Any:
    """
    Reconstruct a LanguageModel subclass on the server.

    This enables third-party model wrappers (like nnterp's StandardizedTransformer)
    to work on servers that don't have the library installed.

    Args:
        subclass_data: Dict containing class_name, discovered_classes, state, model_key
        server_model: The server's base LanguageModel instance
        namespace: Current execution namespace
        exec_func: Function to execute code with signature (code, ns, source_file, start_line)

    Returns:
        Reconstructed model as an instance of the subclass
    """
    class_name = subclass_data.get('class_name')
    discovered_classes = subclass_data.get('discovered_classes', {})
    custom_state = subclass_data.get('state', {})

    # First, execute all discovered class definitions
    # Sort by dependency order if needed (base classes first)
    # For simplicity, we execute them in the order they were discovered
    for cls_name, cls_data in discovered_classes.items():
        source_data = cls_data.get('source', '')
        if isinstance(source_data, dict):
            source_code = source_data.get('code', '')
            source_file = source_data.get('file', '<unknown>')
            start_line = source_data.get('line', 1)
        else:
            source_code = source_data
            source_file = '<unknown>'
            start_line = 1

        # Add module references (JSON-serializable values) to namespace
        module_refs = cls_data.get('module_refs', {})
        for ref_name, ref_value in module_refs.items():
            if ref_name not in namespace:
                namespace[ref_name] = ref_value

        # Resolve server-available imports (types and modules)
        server_imports = cls_data.get('server_imports', {})
        for ref_name, import_info in server_imports.items():
            if ref_name in namespace:
                continue
            try:
                import importlib
                import_type = import_info.get('type')
                mod_name = import_info.get('module', '')
                obj_name = import_info.get('name', '')

                if import_type == 'module':
                    # Import the module
                    if mod_name:
                        namespace[ref_name] = importlib.import_module(mod_name)
                elif import_type in ('class', 'callable'):
                    # Import the class or callable from its module
                    if mod_name and obj_name:
                        mod = importlib.import_module(mod_name)
                        namespace[ref_name] = getattr(mod, obj_name)
            except (ImportError, AttributeError):
                # If import fails, continue (might be handled elsewhere)
                pass

        # Execute the class definition
        if source_code:
            try:
                exec_func(source_code, namespace, source_file, start_line)
            except Exception:
                # If a class fails to define, continue with others
                # (might be a base class that's already available)
                pass

    # Get the target subclass from namespace
    if class_name not in namespace:
        raise ValueError(
            f"LanguageModel subclass '{class_name}' could not be reconstructed. "
            f"Class definition may have failed to execute."
        )

    subclass = namespace[class_name]

    # Create instance without calling __init__
    reconstructed = object.__new__(subclass)

    # Get server-provided attributes from the class hierarchy
    server_provided = getattr(subclass, '_server_provided', frozenset())

    # Copy all attributes from server_model first (as base)
    for key, value in server_model.__dict__.items():
        reconstructed.__dict__[key] = value

    # Override with custom state (non-server-provided attributes)
    for key, value in custom_state.items():
        if isinstance(value, dict) and value.get('__server_provided__'):
            # Keep the server's value (already copied above)
            pass
        else:
            # Use the serialized value
            reconstructed.__dict__[key] = value

    return reconstructed


# =============================================================================
# Remote Object Extraction
# =============================================================================
# Functions for extracting @remote decorated objects and lambdas from the trace
# frame's local variables. These functions are called by extract_all() during
# serialization to identify which objects need their source code sent to the server.
#
# For @remote functions/classes: The source and metadata were captured at import
# time by the @remote decorator. We extract and include them in the payload.
#
# For instances of @remote classes: We serialize their __dict__ state so the
# server can reconstruct the instance after re-creating the class from source.
#
# For lambdas: We extract the source using AST parsing (extract_lambda_source),
# validate it doesn't capture non-serializable closures, and include it in the
# payload. Complex lambdas should be converted to @remote functions.

def is_the_traced_model(value: Any, traced_model: Any) -> bool:
    """Check if value IS the specific model being traced (by identity)."""
    if traced_model is None:
        return False
    return value is traced_model


def extract_remote_object(var_name: str, value: Any, result: Dict[str, Any], traced_model: Any = None) -> None:
    """
    Extract a @nnsight.remote object (function, class, or instance) for serialization.

    Args:
        var_name: The variable name in the trace
        value: The @nnsight.remote decorated object or instance
        result: Dict to add the extracted data to
        traced_model: The model being traced (for identity-based model ref detection)
    """
    # Determine if it's a class/function itself or an instance
    if isinstance(value, type):
        # It's a class
        cls = value
        is_instance = False
    elif isinstance(value, types.FunctionType) and hasattr(value, '_remote_source'):
        # It's a @remote decorated function (not a callable instance like nn.Module)
        cls = value
        is_instance = False
    else:
        # It's an instance (including nn.Module instances which are callable)
        cls = type(value)
        is_instance = True

    # Use qualified name as key to prevent collisions between classes
    # with the same short name from different modules
    cls_qualified = f"{cls.__module__}.{cls.__name__}"

    # Create entry for this class/function if not exists
    if cls_qualified not in result:
        result[cls_qualified] = _get_remote_metadata(cls)

    # For instances, serialize their state
    if is_instance:
        # Pass result as discovered_classes so any new dependencies get added
        instance_state = serialize_instance_state(value, discovered_classes=result, traced_model=traced_model)
        result[cls_qualified]["instances"][str(id(value))] = {
            "var_name": var_name,
            "state": instance_state
        }


def extract_lambda_object(var_name: str, func: Any, result: Dict[str, Any]) -> None:
    """
    Extract a lambda function for serialization.

    Args:
        var_name: The variable name in the trace
        func: The lambda function
        result: Dict to add the extracted data to

    Raises:
        SourceSerializationError: If lambda source cannot be extracted
    """
    import inspect

    # Get file/line metadata first (for error messages)
    try:
        source_file = inspect.getfile(func)
        source_line = func.__code__.co_firstlineno
        location = f"{source_file}:{source_line}"
    except (OSError, TypeError):
        source_file = None
        source_line = None
        location = None

    # Use the lambda extraction function
    source, errors = validate_lambda_for_remote(func)

    if errors:
        # Format errors nicely
        prefix = f"{location}: " if location else ""
        error_text = '\n'.join(f"  - {e}" for e in errors)
        raise SourceSerializationError(
            f"{prefix}lambda '{var_name}' cannot be serialized for remote execution:\n\n"
            f"{error_text}\n\n"
            f"Consider converting to a named function:\n\n"
            f"  @nnsight.remote\n"
            f"  def {var_name}(...):\n"
            f"      return ..."
        )

    # Extract closure variables if present
    closure_vars = {}
    prefix = f"{location}: " if location else ""
    if func.__closure__ and hasattr(func.__code__, 'co_freevars'):
        for name, cell in zip(func.__code__.co_freevars, func.__closure__):
            try:
                value = cell.cell_contents
            except ValueError:
                # Empty cell (variable was deleted)
                continue

            # Skip modules from allowed list (available on server)
            if isinstance(value, types.ModuleType):
                if is_server_available_module(value.__name__):
                    continue
                raise SourceSerializationError(
                    f"{prefix}lambda '{var_name}' captures module '{value.__name__}' "
                    f"which is not available on NDIF server."
                )

            # Skip @nnsight.remote objects (serialized separately)
            if is_remote_object(value):
                continue

            # JSON-serializable values get captured
            if is_json_serializable(value):
                closure_vars[name] = value
                continue

            # Nested lambda in closure
            if is_lambda(value):
                raise SourceSerializationError(
                    f"{prefix}lambda '{var_name}' captures another lambda '{name}' in its closure. "
                    f"Nested lambdas are not supported. Please convert to named functions."
                )

            # Functions from allowed modules (skip - available on server)
            if callable(value) and hasattr(value, '__module__'):
                if is_server_available_module(value.__module__ or ''):
                    continue

            # Non-serializable closure variable - error!
            raise SourceSerializationError(
                f"{prefix}lambda '{var_name}' captures '{name}' of type '{type(value).__name__}' "
                f"which cannot be serialized.\n\n"
                f"Options:\n"
                f"  - Pass '{name}' as an argument instead of capturing it\n"
                f"  - Use a JSON-serializable type (int, float, str, list, dict)\n"
                f"  - Convert to a @nnsight.remote function that takes '{name}' as a parameter"
            )

    # Create a unique key for this lambda (use id to handle multiple lambdas)
    lambda_key = f"__lambda_{id(func)}"

    result[lambda_key] = {
        "source": {
            "code": source,
            "file": source_file,
            "line": source_line,
        },
        "module_refs": {},  # Lambdas don't have module-level refs like decorated functions
        "closure_vars": closure_vars,
        "type": "lambda",
        "var_name": var_name,  # Store the actual variable name for reconstruction
        "library": None,
        "version": None,
    }


# =============================================================================
# Value Serialization
# =============================================================================
# Functions for recursively serializing Python values into JSON-compatible format.
# Used by serialize_instance_state() to serialize @remote instance __dict__ contents.
#
# Each value type has a specific serialization strategy:
# - Primitives (None, bool, int, float, str): Pass through unchanged
# - Tensors/Parameters: serialize_tensor() with base64 encoding
# - nn.Modules: Serialize class path + __dict__ for reconstruction
# - @remote instances: Reference by id (serialized separately)
# - Auto-discoverable instances: Include class source + __dict__
# - Containers (list, dict, tuple, set): Recursive serialization with markers
# - Callables from allowed modules: Store qualified name for server import
# - Enums: Store class/module/member for reconstruction
#
# Deduplication: A memo dict tracks serialized objects by id() to handle:
# - Shared references (same tensor in multiple places)
# - Circular references (object A contains B which contains A)

def is_nn_parameter(value: Any) -> bool:
    """Check if value is a torch.nn.Parameter."""
    type_name = type(value).__name__
    module = getattr(type(value), '__module__', '')
    return type_name == 'Parameter' and 'torch' in module


def is_nn_module(value: Any) -> bool:
    """Check if value is a torch.nn.Module instance."""
    # Check by inheritance to handle all nn.Module subclasses
    try:
        import torch.nn
        return isinstance(value, torch.nn.Module)
    except ImportError:
        return False


def serialize_nn_module(value: Any, key: str, memo: dict, discovered_classes: Dict[str, Any] = None, traced_model: Any = None) -> Dict[str, Any]:
    """
    Serialize a torch built-in nn.Module instance (e.g., torch.nn.Linear).

    Only handles modules from torch.* - user-defined nn.Module subclasses
    go through regular auto-discovery like any other class.
    """
    if discovered_classes is None:
        discovered_classes = {}

    cls = type(value)
    module_path = cls.__module__
    class_name = cls.__name__

    # Only handle built-in torch modules here
    if not module_path.startswith('torch'):
        return None  # Signal to caller to try auto-discovery instead

    # Register in memo before recursing (handles circular refs)
    obj_id = id(value)
    ref_id = f"module_{obj_id}"
    serialized_dict = {}
    result = {
        NN_MODULE_MARKER: f"{module_path}.{class_name}",
        DICT_MARKER: serialized_dict,
        ID_MARKER: ref_id,
    }
    memo[obj_id] = (ref_id, result)

    # Recursively serialize __dict__
    for k, v in value.__dict__.items():
        serialized_dict[k] = serialize_value(v, f"{key}.{k}", memo, discovered_classes, traced_model)

    return result


def serialize_value(value: Any, key: str, memo: dict, discovered_classes: Dict[str, Any] = None, traced_model: Any = None) -> Any:
    """
    Serialize a single value for instance state.

    Args:
        value: The value to serialize
        key: The attribute key (for error messages)
        memo: Dict mapping id(obj) -> (ref_id, serialized) for deduplication
        discovered_classes: Dict to accumulate auto-discovered classes
        traced_model: The model being traced (for identity-based model ref detection)

    Returns:
        Serialized representation of the value
    """
    if discovered_classes is None:
        discovered_classes = {}

    # Primitives don't need deduplication
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    # Check if we've already serialized this object (deduplication)
    obj_id = id(value)
    if obj_id in memo:
        ref_id, _ = memo[obj_id]
        return {REF_MARKER: ref_id}

    # Handle THE traced model (by identity)
    if is_the_traced_model(value, traced_model):
        return {MODEL_REF_MARKER: True}

    # Handle Enum instances (serialize as class + member name)
    from enum import Enum
    if isinstance(value, Enum):
        return {
            ENUM_MARKER: True,
            "class": type(value).__name__,
            "module": type(value).__module__,
            "member": value.name,
        }

    # Handle torch built-in nn.Module instances (e.g., torch.nn.Linear)
    # User-defined nn.Module subclasses fall through to auto-discovery
    if is_nn_module(value):
        result = serialize_nn_module(value, key, memo, discovered_classes, traced_model)
        if result is not None:
            return result
        # Fall through to auto-discovery for user-defined nn.Module subclasses

    # Handle auto-discoverable instances (user-defined classes, including nn.Module subclasses)
    if is_auto_discoverable_instance(value) or is_remote_object(value):
        cls = type(value)
        cls_qualified = f"{cls.__module__}.{cls.__name__}"

        # Auto-discover the class if not already discovered
        if cls_qualified not in discovered_classes:
            auto_discover_class(cls, discovered_classes)

        # Serialize with fully qualified class name
        ref_id = f"auto_{obj_id}"
        serialized_dict = {}
        result = {
            CLASS_MARKER: cls_qualified,
            DICT_MARKER: serialized_dict,
            ID_MARKER: ref_id,
        }
        memo[obj_id] = (ref_id, result)

        # Recursively serialize __dict__
        for k, v in value.__dict__.items():
            serialized_dict[k] = serialize_value(v, f"{key}.{k}", memo, discovered_classes, traced_model)

        return result

    # Handle nn.Parameter (must check before is_tensor since Parameter is a Tensor subclass)
    if is_nn_parameter(value):
        # Register in memo before serializing (for circular refs)
        ref_id = f"param_{obj_id}"
        tensor_data = serialize_tensor(value.data)
        tensor_data[NN_PARAMETER_MARKER] = True
        tensor_data["requires_grad"] = value.requires_grad
        tensor_data[ID_MARKER] = ref_id
        memo[obj_id] = (ref_id, tensor_data)
        return tensor_data

    # Handle tensors
    if is_tensor(value):
        # Register in memo before serializing (for circular refs)
        ref_id = f"tensor_{obj_id}"
        tensor_data = serialize_tensor(value)
        tensor_data[ID_MARKER] = ref_id
        memo[obj_id] = (ref_id, tensor_data)
        return tensor_data

    # Handle nested dicts (like nn.Module's _parameters, _buffers, _modules)
    if isinstance(value, dict):
        # Check if the dict is fully JSON-serializable - if so, return as-is
        if is_json_serializable(value):
            return value
        # Register in memo before recursing (handles circular refs)
        ref_id = f"dict_{obj_id}"
        serialized_dict = {}
        result = {DICT_MARKER: serialized_dict, ID_MARKER: ref_id}
        memo[obj_id] = (ref_id, result)
        for k, v in value.items():
            serialized_dict[k] = serialize_value(v, f"{key}.{k}", memo, discovered_classes, traced_model)
        return result

    # Handle lists/tuples
    if isinstance(value, (list, tuple)):
        # Check if all items are JSON-serializable - if so, return as-is
        if all(is_json_serializable(item) for item in value):
            return list(value) if isinstance(value, tuple) else value
        # Register in memo before recursing
        ref_id = f"list_{obj_id}"
        serialized_list = []
        result = {LIST_MARKER: serialized_list, TUPLE_MARKER: isinstance(value, tuple), ID_MARKER: ref_id}
        memo[obj_id] = (ref_id, result)
        for i, item in enumerate(value):
            serialized_list.append(serialize_value(item, f"{key}[{i}]", memo, discovered_classes, traced_model))
        return result

    # Handle sets
    if isinstance(value, set):
        # Sets can only contain JSON-serializable values
        if all(is_json_serializable(item) for item in value):
            return {SET_MARKER: list(value)}
        raise SourceSerializationError(
            f"Set attribute '{key}' contains non-JSON-serializable values."
        )

    # JSON-serializable values (complex types like nested lists/dicts)
    if is_json_serializable(value):
        return value

    # Functions from allowed modules
    if callable(value):
        callable_ref = get_callable_reference(value)
        if callable_ref:
            return {CALLABLE_REF_MARKER: callable_ref}

    # Weakrefs - internal torch state, serialize as None (will be rebuilt)
    import weakref
    if isinstance(value, weakref.ReferenceType):
        return {WEAKREF_MARKER: True}

    # Type references (like torch.float32)
    if isinstance(value, type):
        module = getattr(value, '__module__', '')
        if is_server_available_module(module) or module == 'builtins':
            return {TYPE_REF_MARKER: f"{module}.{value.__name__}"}

    raise SourceSerializationError(
        f"Instance attribute '{key}' of type '{type(value).__name__}' "
        f"cannot be serialized for remote execution.\n"
        f"Options:\n"
        f"  - Use a JSON-serializable type (int, float, str, list, dict)\n"
        f"  - Mark it with @nnsight.remote if it's a custom class\n"
        f"  - Use functions from allowed modules (torch, numpy, etc.)"
    )


def serialize_instance_state(obj: Any, memo: dict = None, discovered_classes: Dict[str, Any] = None, traced_model: Any = None) -> Dict[str, Any]:
    """
    Serialize a @nnsight.remote instance's state (__dict__ or __slots__).

    Args:
        obj: Instance of a @nnsight.remote class
        memo: Dict mapping id(obj) -> (ref_id, serialized) for deduplication
        discovered_classes: Dict to accumulate auto-discovered classes
        traced_model: The model being traced (for identity-based model ref detection)

    Returns:
        Dict containing the serialized state

    Raises:
        SourceSerializationError: If an attribute cannot be serialized
    """
    if memo is None:
        memo = {}
    if discovered_classes is None:
        discovered_classes = {}

    state = {}

    # Handle __slots__ classes
    cls = type(obj)
    has_slots = False

    # Collect all slots from class hierarchy
    all_slots = []
    for klass in cls.__mro__:
        if hasattr(klass, '__slots__'):
            slots = klass.__slots__
            if isinstance(slots, str):
                slots = [slots]
            all_slots.extend(slots)
            has_slots = True

    if has_slots and all_slots:
        # Serialize slot values
        for slot in all_slots:
            if slot == '__dict__':
                continue  # Skip __dict__ slot, handle separately
            if slot == '__weakref__':
                continue  # Skip weakref slot
            if hasattr(obj, slot):
                try:
                    value = getattr(obj, slot)
                    state[slot] = serialize_value(value, slot, memo, discovered_classes, traced_model)
                except AttributeError:
                    pass  # Slot not set

    # Also serialize __dict__ if present (slots classes can have __dict__ too)
    if hasattr(obj, '__dict__'):
        for key, value in obj.__dict__.items():
            state[key] = serialize_value(value, key, memo, discovered_classes, traced_model)

    return state


def get_callable_reference(value: Any) -> Optional[str]:
    """
    Get a fully-qualified reference string for a callable from allowed modules.

    Args:
        value: A callable object

    Returns:
        Reference string like "torch.nn.functional.relu" or None if not from allowed modules
    """
    if not callable(value):
        return None

    if not hasattr(value, '__module__') or not hasattr(value, '__name__'):
        return None

    module = value.__module__

    # Check if it's from a server-available module
    if not is_server_available_module(module):
        return None

    # Handle special cases for qualified names (like methods)
    if hasattr(value, '__qualname__'):
        qualname = value.__qualname__
        # If qualname differs from __name__, use it for nested classes/functions
        if '.' in qualname and qualname != value.__name__:
            return f"{module}.{qualname}"

    return f"{module}.{value.__name__}"


# =============================================================================
# Top-Level API: Serialization and Deserialization
# =============================================================================
# Main entry points for source-based serialization.
#
# Client-side (serialize_source_based):
#   Called when the trace context exits with a remote backend. (The Tracer.__exit__
#   method invokes RemoteBackend, which serializes the trace for submission to NDIF.)
#   Extracts the trace source code,
#   frame locals, and all dependencies into a JSON payload. The payload includes:
#   - The trace block source code with file/line metadata for error mapping
#   - JSON-serializable variables (int, str, list, dict, tensors)
#   - @remote object definitions (source + module refs + instances)
#   - Model subclass definitions (for LanguageModel subclasses)
#
# Server-side (deserialize_source_based):
#   Called on the NDIF server to reconstruct the execution environment.
#   Re-creates all @remote classes/functions by exec'ing their source, rebuilds
#   instances with their serialized state, and returns a namespace dict ready
#   for executing the user's trace code.

def can_serialize_source_based(tracer: "Tracer") -> Tuple[bool, Optional[str]]:
    """
    Check if a tracer can be serialized using source-based serialization.

    Args:
        tracer: The tracer to check

    Returns:
        (can_serialize, error_message) tuple
    """
    try:
        frame_locals = tracer.info.frame.f_locals if tracer.info.frame else {}
        # Find the traced model from frame locals (for identity-based model ref detection)
        traced_model = None
        for value in frame_locals.values():
            if is_model_reference(value):
                traced_model = value
                break
        extract_all(frame_locals, traced_model=traced_model)
        return True, None
    except SourceSerializationError as e:
        return False, str(e)


# Server-side deserialization

def _exec_with_source_info(
    source_code: str,
    namespace: dict,
    source_file: str = "<unknown>",
    start_line: int = 1,
) -> None:
    """
    Execute source code with proper file/line info for error tracebacks.

    This ensures that when errors occur in user code, the traceback shows
    the original file path and line numbers from the user's source, not
    generic "<string>" with line 1.

    Args:
        source_code: The source code to execute
        namespace: The namespace dict for exec()
        source_file: Original source file path
        start_line: Starting line number in the original file
    """
    tree = ast.parse(source_code)
    ast.increment_lineno(tree, start_line - 1)
    code_obj = compile(tree, source_file, 'exec')
    exec(code_obj, namespace)


def deserialize_source_based(
    payload: bytes,
    model: Any,
    user_id: Optional[str] = None,
    job_id: Optional[str] = None,
    use_restricted: bool = False,
) -> Dict[str, Any]:
    """
    Deserialize a source-based payload and prepare for execution.

    This reconstructs @nnsight.remote classes and functions, their instances,
    and simple variables into a namespace suitable for exec().

    Args:
        payload: JSON-encoded bytes from serialize_source_based()
        model: The model object to inject into the namespace
        user_id: (Optional) User ID for security audit logging
        job_id: (Optional) Job ID for security audit logging
        use_restricted: If True, use RestrictedPython for execution

    Returns:
        Namespace dict ready for code execution
    """
    import torch
    import numpy
    import os
    import pathlib
    import random
    from nnsight.remote import remote_noop

    data = json.loads(payload.decode('utf-8'))

    # Build base namespace with allowed modules
    # Note: We use remote_noop instead of remote because:
    # 1. Code was already validated client-side before transmission
    # 2. Source extraction won't work on exec'd code
    # 3. remote_noop just marks objects as validated without re-checking
    base_modules = {
        'torch': torch,
        'numpy': numpy,
        'np': numpy,
        'os': os,
        'pathlib': pathlib,
        'random': random,
        'model': model,
        'remote': remote_noop,  # No-op for deserialized code (already validated)
    }

    # Set up execution environment
    if use_restricted:
        from .restricted_execution import (
            create_restricted_globals,
            compile_user_code,
            DEFAULT_ALLOWED_MODULES,
        )
        allowed_modules = DEFAULT_ALLOWED_MODULES | {'os', 'pathlib'}
        effective_user_id = user_id or "unknown"
        effective_job_id = job_id or "unknown"
        namespace = create_restricted_globals(
            user_id=effective_user_id,
            job_id=effective_job_id,
            base_globals=base_modules,
            allowed_modules=allowed_modules,
        )

        def exec_func(code, ns, source_file='<user_code>', start_line=1):
            compiled = compile_user_code(
                code,
                filename=source_file,
                allowed_modules=allowed_modules,
                user_id=effective_user_id,
                job_id=effective_job_id,
            )
            exec(compiled, ns)
    else:
        namespace = dict(base_modules)

        def exec_func(code, ns, source_file='<unknown>', start_line=1):
            _exec_with_source_info(code, ns, source_file, start_line)

    # Store source metadata for error mapping
    source_info = data.get('source', {})
    if isinstance(source_info, dict):
        namespace['__nnsight_source_file__'] = source_info.get('file', '<unknown>')
        namespace['__nnsight_source_line__'] = source_info.get('line', 1)
        namespace['__nnsight_source_code__'] = source_info.get('code', '')
    else:
        # Handle legacy format (v2.0) where source is just a string
        namespace['__nnsight_source_file__'] = '<unknown>'
        namespace['__nnsight_source_line__'] = 1
        namespace['__nnsight_source_code__'] = source_info if isinstance(source_info, str) else "".join(source_info)

    # Track reconstructed remote objects for cross-referencing
    reconstructed_instances = {}

    # Handle LanguageModel subclass reconstruction
    # This enables third-party model wrappers (like nnterp's StandardizedTransformer)
    # to work on servers that don't have the library installed
    model_subclass_data = data.get('model_subclass')
    if model_subclass_data:
        model = reconstruct_model_subclass(
            model_subclass_data,
            model,
            namespace,
            exec_func,
        )
        # Update the model reference in namespace
        namespace['model'] = model

    # Reconstruct @nnsight.remote functions and classes
    # Note: Keys are now qualified names (e.g., 'mymodule.MyClass') for disambiguation,
    # but exec creates objects with their short names. We extract the short name for lookup.
    for obj_qualified_name, obj_data in data.get('remote_objects', {}).items():
        # Add captured module-level references
        namespace.update(obj_data.get('module_refs', {}))

        # Add closure variables
        namespace.update(obj_data.get('closure_vars', {}))

        # Extract source code and location info (handle both new and legacy format)
        source_data = obj_data.get('source', '')
        if isinstance(source_data, dict):
            source_code = source_data.get('code', '')
            source_file = source_data.get('file', '<unknown>')
            start_line = source_data.get('line', 1)
        else:
            source_code = source_data
            source_file = '<unknown>'
            start_line = 1

        # Extract short name from qualified name for namespace lookup
        # Qualified name format: 'module.submodule.ClassName' -> 'ClassName'
        # For backwards compatibility, handle both qualified and short names
        if '.' in obj_qualified_name:
            obj_short_name = obj_qualified_name.split('.')[-1]
        else:
            obj_short_name = obj_qualified_name

        # Handle different types
        obj_type = obj_data.get('type', 'function')

        if obj_type == 'lambda':
            # Lambda: wrap in assignment and execute
            var_name = obj_data.get('var_name', obj_short_name)
            lambda_assignment = f"{var_name} = {source_code}"
            exec_func(lambda_assignment, namespace, source_file, start_line)
        else:
            # Function or class: execute definition directly with line numbers
            exec_func(source_code, namespace, source_file, start_line)

            # For classes, reconstruct instances
            if obj_type == 'class':
                cls = namespace[obj_short_name]
                for instance_id, instance_data in obj_data.get('instances', {}).items():
                    # Create instance without calling __init__
                    obj = object.__new__(cls)
                    obj.__dict__ = reconstruct_state(
                        instance_data['state'],
                        namespace,
                        model,
                        reconstructed_instances
                    )
                    var_name = instance_data['var_name']
                    namespace[var_name] = obj
                    reconstructed_instances[instance_id] = obj

    # Add simple variables (deserializing tensors if needed)
    for var_name, var_value in data.get('variables', {}).items():
        if isinstance(var_value, dict) and '__tensor__' in var_value:
            namespace[var_name] = deserialize_tensor(var_value)
        else:
            namespace[var_name] = var_value

    # Handle model refs
    for ref_name in data.get('model_refs', []):
        namespace[ref_name] = model

    return namespace


# =============================================================================
# Value Reconstruction
# =============================================================================
# Functions for recursively reconstructing Python values from serialized JSON.
# Called server-side by deserialize_source_based() to rebuild @remote instance
# state and nested data structures.
#
# Each marker type has a reconstruction strategy:
# - __tensor__: deserialize_tensor() with optional quantization/sparsity
# - __nn_module__: object.__new__(cls) + reconstruct __dict__
# - __auto_instance__: Same as above, class was exec'd from discovered source
# - __dict__/__list__/__tuple__/__set__: Recursive reconstruction
# - __ref__: Return previously reconstructed object (deduplication)
# - __model_ref__: Return the traced model
# - __callable_ref__/__type_ref__: Import from module path
# - __enum__: Import enum class and get member by name
#
# The reconstructed dict tracks objects by their serialized ID_MARKER to handle
# shared references and circular references correctly.

def reconstruct_value(value: Any, namespace: dict, model: Any, reconstructed: dict) -> Any:
    """
    Reconstruct a single serialized value.

    Args:
        value: The serialized value
        namespace: Current namespace
        model: The model object
        reconstructed: Dict mapping ref_id -> reconstructed object for deduplication

    Returns:
        Reconstructed value
    """
    if not isinstance(value, dict):
        return value

    # Reference to previously reconstructed object (deduplication)
    if REF_MARKER in value:
        ref_id = value[REF_MARKER]
        if ref_id in reconstructed:
            return reconstructed[ref_id]
        raise ValueError(f"Reference '{ref_id}' not found in reconstructed objects")

    # Model reference
    if MODEL_REF_MARKER in value:
        return model

    # Remote object reference
    if REMOTE_REF_MARKER in value:
        ref_id = str(value[REMOTE_REF_MARKER])
        if ref_id in reconstructed:
            return reconstructed[ref_id]
        # Placeholder, will be resolved later if needed
        return value

    # User-defined class instance (auto-discovered or @remote)
    if CLASS_MARKER in value:
        cls_qualified = value[CLASS_MARKER]

        # Extract short name from qualified name for namespace lookup
        # Qualified name format: 'module.submodule.ClassName' -> 'ClassName'
        if '.' in cls_qualified:
            cls_short_name = cls_qualified.split('.')[-1]
        else:
            cls_short_name = cls_qualified

        # Get class from namespace (should have been exec'd from source)
        cls = namespace.get(cls_short_name)
        if cls is None:
            raise ValueError(f"Class '{cls_qualified}' not found in namespace")

        # Create instance without __init__
        instance = object.__new__(cls)

        # Register BEFORE recursing (handles circular refs)
        if ID_MARKER in value:
            reconstructed[value[ID_MARKER]] = instance

        # Reconstruct __dict__
        if DICT_MARKER in value:
            for k, v in value[DICT_MARKER].items():
                setattr(instance, k, reconstruct_value(v, namespace, model, reconstructed))

        return instance

    # Callable reference
    if CALLABLE_REF_MARKER in value:
        ref = value[CALLABLE_REF_MARKER]
        parts = ref.rsplit('.', 1)
        if len(parts) == 2:
            mod_name, func_name = parts
            import importlib
            try:
                mod = importlib.import_module(mod_name)
                return getattr(mod, func_name)
            except (ImportError, AttributeError):
                return value
        return value

    # Type reference
    if TYPE_REF_MARKER in value:
        ref = value[TYPE_REF_MARKER]
        parts = ref.rsplit('.', 1)
        if len(parts) == 2:
            mod_name, type_name = parts
            import importlib
            try:
                mod = importlib.import_module(mod_name)
                return getattr(mod, type_name)
            except (ImportError, AttributeError):
                return value
        return value

    # nn.Module (built-in torch modules or @remote modules)
    if NN_MODULE_MARKER in value:
        import importlib
        module_class_path = value[NN_MODULE_MARKER]
        parts = module_class_path.rsplit('.', 1)
        if len(parts) == 2:
            mod_name, class_name = parts

            # First check if class is in namespace (for @remote modules)
            cls = namespace.get(class_name)

            if cls is None:
                # Try to import (for torch built-in modules)
                try:
                    mod = importlib.import_module(mod_name)
                    cls = getattr(mod, class_name)
                except (ImportError, AttributeError) as e:
                    raise ValueError(f"Cannot reconstruct nn.Module '{module_class_path}': {e}")

            # Use object.__new__() + __dict__ approach
            instance = object.__new__(cls)

            # Register BEFORE recursing (handles circular refs)
            if ID_MARKER in value:
                reconstructed[value[ID_MARKER]] = instance

            # Reconstruct __dict__
            if DICT_MARKER in value:
                reconstructed_dict = {}
                for k, v in value[DICT_MARKER].items():
                    reconstructed_dict[k] = reconstruct_value(v, namespace, model, reconstructed)
                instance.__dict__.update(reconstructed_dict)

            return instance
        return value

    # nn.Parameter
    if TENSOR_MARKER in value and value.get(NN_PARAMETER_MARKER):
        import torch
        tensor = deserialize_tensor(value)
        param = torch.nn.Parameter(tensor, requires_grad=value.get('requires_grad', True))
        # Register for deduplication
        if ID_MARKER in value:
            reconstructed[value[ID_MARKER]] = param
        return param

    # Regular tensor
    if TENSOR_MARKER in value:
        tensor = deserialize_tensor(value)
        # Register for deduplication
        if ID_MARKER in value:
            reconstructed[value[ID_MARKER]] = tensor
        return tensor

    # Nested dict
    if DICT_MARKER in value:
        result = {}
        # Register BEFORE recursing (handles circular refs)
        if ID_MARKER in value:
            reconstructed[value[ID_MARKER]] = result
        for k, v in value[DICT_MARKER].items():
            result[k] = reconstruct_value(v, namespace, model, reconstructed)
        return result

    # List/tuple
    if LIST_MARKER in value:
        result = []
        # Register BEFORE recursing (handles circular refs)
        if ID_MARKER in value:
            reconstructed[value[ID_MARKER]] = result
        for item in value[LIST_MARKER]:
            result.append(reconstruct_value(item, namespace, model, reconstructed))
        if value.get(TUPLE_MARKER):
            result = tuple(result)
            # Update registration for tuple
            if ID_MARKER in value:
                reconstructed[value[ID_MARKER]] = result
        return result

    # Set
    if SET_MARKER in value:
        return set(value[SET_MARKER])

    # Enum
    if ENUM_MARKER in value:
        import importlib
        from enum import Enum
        module_name = value['module']
        class_name = value['class']
        member_name = value['member']
        try:
            module = importlib.import_module(module_name)
            enum_class = getattr(module, class_name)
            return enum_class[member_name]
        except (ImportError, AttributeError, KeyError):
            # If we can't reconstruct the enum, return a dict with the info
            return {ENUM_MARKER_FALLBACK: True, "class": class_name, "member": member_name}

    # Weakref - return None (will be rebuilt by torch)
    if WEAKREF_MARKER in value:
        return None

    # Unknown dict - return as-is
    return value


def reconstruct_state(
    state: dict,
    namespace: dict,
    model: Any,
    reconstructed: dict
) -> dict:
    """
    Reconstruct instance state, resolving references.

    Args:
        state: Serialized state dict
        namespace: Current namespace
        model: The model object
        reconstructed: Dict of already reconstructed instances by id

    Returns:
        Reconstructed state dict
    """
    result = {}

    for key, value in state.items():
        result[key] = reconstruct_value(value, namespace, model, reconstructed)

    return result
