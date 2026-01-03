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
import zlib
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
# Changing these values would break backwards compatibility.

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
AUTO_INSTANCE_MARKER = "__auto_instance__"
STATE_MARKER = "__state__"
CALLABLE_REF_MARKER = "__callable_ref__"
TYPE_REF_MARKER = "__type_ref__"
WEAKREF_MARKER = "__weakref__"
SERVER_PROVIDED_MARKER = "__server_provided__"
ENUM_FALLBACK_MARKER = "__enum_fallback__"


# =============================================================================
# Remote Metadata Extraction
# =============================================================================

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

    # Check cache first
    if cls in _auto_discovered_cache:
        cached = _auto_discovered_cache[cls]
        if cls_name not in discovered:
            discovered[cls_name] = cached
        return cached

    # Already @remote decorated - use existing metadata
    if getattr(cls, '_remote_validated', False):
        result = _get_remote_metadata(cls)
        _auto_discovered_cache[cls] = result
        discovered[cls_name] = result
        return result

    # Extract source
    try:
        source = inspect.getsource(cls)
        dedented_source = textwrap.dedent(source)
    except (OSError, TypeError) as e:
        raise SourceSerializationError(
            f"Cannot auto-discover class '{cls_name}': source not available. {e}"
        )

    # Get file/line info
    try:
        source_file = inspect.getfile(cls)
        _, start_line = inspect.getsourcelines(cls)
    except (OSError, TypeError):
        source_file = "<unknown>"
        start_line = 1

    # Parse AST and find external references
    try:
        tree = ast.parse(dedented_source)
    except SyntaxError as e:
        raise SourceSerializationError(
            f"Cannot parse source for '{cls_name}': {e}"
        )

    # Find external references
    external_names = find_external_references(tree, cls)

    # Get globals to resolve references
    module_globals = {}
    if hasattr(cls, '__module__'):
        module = sys.modules.get(cls.__module__)
        module_globals = getattr(module, '__dict__', {}) if module else {}

    # Resolve module references, but also track type references for auto-discovery
    module_refs, resolution_errors = resolve_module_references(external_names, cls)

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
    ast_errors = validate_ast(tree, cls_name)

    # For auto-discovered classes, we're more lenient - just warn
    if filtered_errors or ast_errors:
        import warnings
        all_errors = filtered_errors + ast_errors
        if all_errors:
            warnings.warn(
                f"Auto-discovered class '{cls_name}' has potential issues:\n" +
                "\n".join(f"  - {e}" for e in all_errors[:3])
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
    }

    # Cache and record
    _auto_discovered_cache[cls] = result
    discovered[cls_name] = result

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
        if dep_type.__name__ not in discovered:
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

# Compression threshold: only compress if we save at least 10%
COMPRESSION_THRESHOLD = 0.9


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


def serialize_tensor(value: Any) -> Dict[str, Any]:
    """
    Serialize a tensor to a JSON-compatible dict with optional compression.

    Strategy:
    - Convert to numpy bytes (handling sparse, quantized, device-specific tensors)
    - Try zlib compression (level=1 for speed)
    - Only use compression if it saves at least 10%
    - Base64 encode for JSON transport

    Handles special cases:
    - Sparse tensors: preserved in COO format (memory efficient)
    - Quantized tensors: preserved with quantization parameters (smaller, exact)
    - GPU/MPS tensors: moved to CPU first

    Returns:
        Dict with keys:
        - __tensor__: base64-encoded bytes
        - dtype: string dtype (e.g., "float32")
        - shape: list of dimensions
        - compressed: bool indicating if zlib compressed
        - quantization: (optional) dict with scale, zero_point, qtype for quantized tensors
        - sparse: (optional) dict with indices, dense_shape for sparse COO tensors

    Wire format example (uncompressed):
        {"__tensor__": "base64...", "dtype": "float32", "shape": [768], "compressed": false}

    Wire format example (quantized - preserves exact int8 values, 4x smaller):
        {"__tensor__": "base64...", "dtype": "int8", "shape": [768], "compressed": true,
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

            # Compress indices
            indices_bytes = indices.tobytes()
            indices_compressed = zlib.compress(indices_bytes, level=1)
            if len(indices_compressed) < len(indices_bytes) * COMPRESSION_THRESHOLD:
                indices_data = base64.b64encode(indices_compressed).decode('ascii')
                indices_is_compressed = True
            else:
                indices_data = base64.b64encode(indices_bytes).decode('ascii')
                indices_is_compressed = False

            sparse_info = {
                "indices": indices_data,
                "indices_dtype": str(indices.dtype),
                "indices_shape": list(indices.shape),
                "indices_compressed": indices_is_compressed,
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

    # Get raw bytes
    raw_bytes = np_array.tobytes()

    # Try compression
    compressed_bytes = zlib.compress(raw_bytes, level=1)

    # Only use compression if it actually helps (at least 10% savings)
    if len(compressed_bytes) < len(raw_bytes) * COMPRESSION_THRESHOLD:
        data_bytes = compressed_bytes
        is_compressed = True
    else:
        data_bytes = raw_bytes
        is_compressed = False

    # Base64 encode for JSON
    b64_data = base64.b64encode(data_bytes).decode('ascii')

    result = {
        TENSOR_MARKER: b64_data,
        "dtype": str(np_array.dtype),
        "shape": list(np_array.shape),
        "compressed": is_compressed,
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

    # Decompress if needed
    if data.get("compressed", False):
        raw_bytes = zlib.decompress(raw_bytes)

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
            if s.get("indices_compressed", False):
                indices_bytes = zlib.decompress(indices_bytes)
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


def serialize_source_based(tracer: "Tracer") -> bytes:
    """
    Serialize a tracer for remote execution using source + JSON.

    This is the new serialization format that doesn't require matching
    Python versions between client and server.

    Args:
        tracer: The tracer object containing source and frame information

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
    variables, remote_objects, model_refs = extract_all(frame_locals, traced_model=traced_model)

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

    return json.dumps(payload).encode('utf-8')


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


def extract_all(locals_dict: Dict[str, Any], traced_model: Any = None) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """
    Extract all serializable data from locals.

    Args:
        locals_dict: Dictionary of local variables
        traced_model: The model being traced (for identity-based model ref detection)

    Returns:
        (variables, remote_objects, model_refs) where:
        - variables: JSON-serializable simple values
        - remote_objects: @nnsight.remote functions/classes/instances
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

        # If we get here, we can't serialize this value
        raise SourceSerializationError(
            f"Variable '{name}' of type '{type(value).__name__}' "
            f"cannot be serialized for source-based remote execution.\n"
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
# Functions for handling LanguageModel subclasses (like nnterp's StandardizedTransformer).
# These need their source code sent to the server.

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
# Functions for extracting @remote decorated objects and lambdas from frame locals.

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
    elif callable(value) and hasattr(value, '_remote_source'):
        # It's a function
        cls = value
        is_instance = False
    else:
        # It's an instance
        cls = type(value)
        is_instance = True

    cls_name = cls.__name__

    # Create entry for this class/function if not exists
    if cls_name not in result:
        result[cls_name] = _get_remote_metadata(cls)

    # For instances, serialize their state
    if is_instance:
        # Pass result as discovered_classes so any new dependencies get added
        instance_state = serialize_instance_state(value, discovered_classes=result, traced_model=traced_model)
        result[cls_name]["instances"][str(id(value))] = {
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

    # Use the lambda extraction function
    source, errors = validate_lambda_for_remote(func)

    if errors:
        # Format errors nicely
        error_text = '\n'.join(f"  - {e}" for e in errors)
        raise SourceSerializationError(
            f"Lambda '{var_name}' cannot be serialized for remote execution:\n\n"
            f"{error_text}\n\n"
            f"Consider converting to a named function:\n\n"
            f"  @nnsight.remote\n"
            f"  def {var_name}(...):\n"
            f"      return ..."
        )

    # Get file/line metadata
    try:
        source_file = inspect.getfile(func)
        source_line = func.__code__.co_firstlineno
    except (OSError, TypeError):
        source_file = "<unknown>"
        source_line = 1

    # Extract closure variables if present
    closure_vars = {}
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
                    f"Lambda '{var_name}' captures module '{value.__name__}' "
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
                    f"Lambda '{var_name}' captures another lambda '{name}' in its closure. "
                    f"Nested lambdas are not supported. Please convert to named functions."
                )

            # Functions from allowed modules (skip - available on server)
            if callable(value) and hasattr(value, '__module__'):
                if is_server_available_module(value.__module__ or ''):
                    continue

            # Non-serializable closure variable - error!
            raise SourceSerializationError(
                f"Lambda '{var_name}' captures '{name}' of type '{type(value).__name__}' "
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
# These functions handle serializing different value types (tensors, nn.Modules,
# primitives, containers) into JSON-compatible format.

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
    Serialize a torch.nn.Module instance using object.__new__() + __dict__ approach.

    All nn.Module subclasses (both built-in and custom) store their state in __dict__,
    including _parameters, _buffers, _modules dicts. We recursively serialize __dict__.
    """
    if discovered_classes is None:
        discovered_classes = {}

    cls = type(value)
    module_path = cls.__module__
    class_name = cls.__name__

    # For ALL nn.Module instances (built-in torch, @remote, or other),
    # serialize via class path + __dict__
    # This works because all nn.Module subclasses store state in __dict__
    if module_path.startswith('torch') or getattr(cls, '_remote_validated', False):
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

    # Unknown module type (not torch, not @remote)
    raise SourceSerializationError(
        f"Instance attribute '{key}' is nn.Module '{class_name}' from '{module_path}' "
        f"which cannot be serialized. Use @nnsight.remote to mark custom modules."
    )


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

    # Handle nn.Module instances FIRST (both built-in torch and @remote)
    # This must come before is_remote_object since @remote nn.Module should use __dict__ serialization
    if is_nn_module(value):
        return serialize_nn_module(value, key, memo, discovered_classes, traced_model)

    # Handle other @remote instances (non-nn.Module classes)
    if is_remote_object(value):
        return {REMOTE_REF_MARKER: id(value), REMOTE_TYPE_MARKER: type(value).__name__}

    # Handle auto-discoverable instances (third-party classes like nnterp's LayerAccessor)
    if is_auto_discoverable_instance(value):
        cls = type(value)
        cls_name = cls.__name__

        # Auto-discover the class if not already discovered
        if cls_name not in discovered_classes:
            auto_discover_class(cls, discovered_classes)

        # Serialize like a @remote instance
        ref_id = f"auto_{obj_id}"
        instance_state = {}
        result = {
            AUTO_INSTANCE_MARKER: cls_name,
            STATE_MARKER: instance_state,
            ID_MARKER: ref_id,
        }
        memo[obj_id] = (ref_id, result)

        # Recursively serialize __dict__
        for k, v in value.__dict__.items():
            instance_state[k] = serialize_value(v, f"{key}.{k}", memo, discovered_classes, traced_model)

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
# Main entry points for source-based serialization. serialize_source_based()
# is called client-side, deserialize_source_based() is called server-side.

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

    data = json.loads(payload.decode('utf-8'))

    # Build base namespace with allowed modules
    base_modules = {
        'torch': torch,
        'numpy': numpy,
        'np': numpy,
        'os': os,
        'pathlib': pathlib,
        'random': random,
        'model': model,
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
    for obj_name, obj_data in data.get('remote_objects', {}).items():
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

        # Handle different types
        obj_type = obj_data.get('type', 'function')

        if obj_type == 'lambda':
            # Lambda: wrap in assignment and execute
            var_name = obj_data.get('var_name', obj_name)
            lambda_assignment = f"{var_name} = {source_code}"
            exec_func(lambda_assignment, namespace, source_file, start_line)
        else:
            # Function or class: execute definition directly with line numbers
            exec_func(source_code, namespace, source_file, start_line)

            # For classes, reconstruct instances
            if obj_type == 'class':
                cls = namespace[obj_name]
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
# These functions handle reconstructing serialized values back to Python objects.
# Called server-side during deserialization.

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

    # Auto-discovered instance (from third-party packages like nnterp)
    if AUTO_INSTANCE_MARKER in value:
        cls_name = value[AUTO_INSTANCE_MARKER]

        # Get class from namespace (should have been exec'd from source)
        cls = namespace.get(cls_name)
        if cls is None:
            raise ValueError(f"Auto-discovered class '{cls_name}' not found in namespace")

        # Create instance without __init__
        instance = object.__new__(cls)

        # Register BEFORE recursing (handles circular refs)
        if ID_MARKER in value:
            reconstructed[value[ID_MARKER]] = instance

        # Reconstruct __dict__ from __state__
        if STATE_MARKER in value:
            for k, v in value[STATE_MARKER].items():
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
