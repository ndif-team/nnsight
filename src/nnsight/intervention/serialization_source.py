"""
Source-based serialization for nnsight remote execution.

This module provides serialization that uses source code + JSON instead of
cloudpickle bytecode. This enables:
- Python version independence (3.10 client can work with 3.12 server)
- Third-party libraries decorated with @nnsight.remote work without server installation
- Clear, early errors instead of mysterious runtime failures
"""

from __future__ import annotations

import base64
import json
import types
import zlib
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..intervention.tracing.base import Tracer

from ..remote import (
    is_json_serializable, ALLOWED_MODULES,
    is_lambda, extract_lambda_source, LambdaExtractionError, validate_lambda_for_remote,
)


class SourceSerializationError(Exception):
    """Raised when source-based serialization fails."""
    pass


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
        "__tensor__": b64_data,
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
    raw_bytes = base64.b64decode(data["__tensor__"])

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

    # Extract file/line metadata for error mapping
    source_metadata = extract_source_metadata(tracer)

    # Extract variables and remote objects
    variables, remote_objects, model_refs = extract_all(frame_locals)

    payload = {
        "version": "2.1",  # Bumped for metadata support
        "source": source_metadata,  # Now includes file/line info
        "variables": variables,
        "remote_objects": remote_objects,
        "model_refs": model_refs,
    }

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


def extract_all(locals_dict: Dict[str, Any], seen: set = None) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """
    Extract all serializable data from locals.

    Args:
        locals_dict: Dictionary of local variables
        seen: Set of already-seen object ids (for cycle detection)

    Returns:
        (variables, remote_objects, model_refs) where:
        - variables: JSON-serializable simple values
        - remote_objects: @nnsight.remote functions/classes/instances
        - model_refs: list of variable names that reference the model
    """
    if seen is None:
        seen = set()

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
            extract_remote_object(name, value, remote_objects)
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
            root = value.__name__.split('.')[0]
            if root in ALLOWED_MODULES:
                continue
            raise SourceSerializationError(
                f"Variable '{name}' references module '{value.__name__}' "
                f"which is not available on NDIF server."
            )

        # Skip functions from allowed modules
        if callable(value) and hasattr(value, '__module__'):
            root = value.__module__.split('.')[0] if value.__module__ else ''
            if root in ALLOWED_MODULES:
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
    """Check if value is a reference to the model (Envoy or NNsight)."""
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


def is_remote_object(obj: Any) -> bool:
    """Check if obj is a @nnsight.remote function/class or instance thereof."""
    # Check if it's a decorated function or class
    if callable(obj) and getattr(obj, '_remote_validated', False):
        return True

    # Check if it's an instance of a decorated class
    if getattr(type(obj), '_remote_validated', False):
        return True

    return False


def extract_remote_object(var_name: str, value: Any, result: Dict[str, Any]) -> None:
    """
    Extract a @nnsight.remote object (function, class, or instance) for serialization.

    Args:
        var_name: The variable name in the trace
        value: The @nnsight.remote decorated object or instance
        result: Dict to add the extracted data to
    """
    import inspect

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
        # Get file/line metadata for remote object
        try:
            source_file = inspect.getfile(cls)
            source_lines, start_line = inspect.getsourcelines(cls)
        except (OSError, TypeError):
            source_file = "<unknown>"
            start_line = 1

        result[cls_name] = {
            "source": {
                "code": getattr(cls, '_remote_source', ''),
                "file": source_file,
                "line": start_line,
            },
            "module_refs": getattr(cls, '_remote_module_refs', {}),
            "closure_vars": getattr(cls, '_remote_closure_vars', {}),
            "type": "class" if isinstance(cls, type) else "function",
            "instances": {},
            # Version metadata for server-side caching
            "library": getattr(cls, '_remote_library', None),
            "version": getattr(cls, '_remote_version', None),
        }

    # For instances, serialize their state
    if is_instance:
        instance_state = serialize_instance_state(value)
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
                root = value.__name__.split('.')[0]
                if root in ALLOWED_MODULES:
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
                root = value.__module__.split('.')[0] if value.__module__ else ''
                if root in ALLOWED_MODULES:
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


def serialize_instance_state(obj: Any, seen: set = None) -> Dict[str, Any]:
    """
    Serialize a @nnsight.remote instance's __dict__.

    Args:
        obj: Instance of a @nnsight.remote class
        seen: Set of already-seen object ids (for cycle detection)

    Returns:
        Dict containing the serialized state

    Raises:
        SourceSerializationError: If an attribute cannot be serialized or cycles detected
    """
    if seen is None:
        seen = set()

    # Cycle detection
    obj_id = id(obj)
    if obj_id in seen:
        raise SourceSerializationError(
            f"Circular reference detected in '{type(obj).__name__}' instance. "
            f"Circular references between @nnsight.remote objects are not yet supported."
        )
    seen.add(obj_id)

    state = {}

    for key, value in obj.__dict__.items():
        # Handle model references
        if is_model_reference(value):
            state[key] = {"__model_ref__": True}
            continue

        # Handle other @remote instances
        if is_remote_object(value):
            state[key] = {"__remote_ref__": id(value), "__remote_type__": type(value).__name__}
            continue

        # Handle tensors
        if is_tensor(value):
            state[key] = serialize_tensor(value)
            continue

        # JSON-serializable values
        if is_json_serializable(value):
            state[key] = value
            continue

        # Functions from allowed modules (including nested attributes like torch.nn.functional.relu)
        if callable(value):
            callable_ref = get_callable_reference(value)
            if callable_ref:
                state[key] = {"__callable_ref__": callable_ref}
                continue

        raise SourceSerializationError(
            f"Instance attribute '{key}' of type '{type(value).__name__}' "
            f"cannot be serialized for remote execution.\n"
            f"Options:\n"
            f"  - Use a JSON-serializable type (int, float, str, list, dict)\n"
            f"  - Mark it with @nnsight.remote if it's a custom class\n"
            f"  - Use functions from allowed modules (torch, numpy, etc.)"
        )

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
    if not module:
        return None

    # Check if it's from an allowed module
    root = module.split('.')[0]
    if root not in ALLOWED_MODULES:
        return None

    # Handle special cases for qualified names (like methods)
    if hasattr(value, '__qualname__'):
        qualname = value.__qualname__
        # If qualname differs from __name__, use it for nested classes/functions
        if '.' in qualname and qualname != value.__name__:
            return f"{module}.{qualname}"

    return f"{module}.{value.__name__}"


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
        extract_all(frame_locals)
        return True, None
    except SourceSerializationError as e:
        return False, str(e)


# Server-side deserialization

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
        exec_func = lambda code, ns: exec(
            compile_user_code(
                code,
                allowed_modules=allowed_modules,
                user_id=effective_user_id,
                job_id=effective_job_id,
            ),
            ns
        )
    else:
        namespace = dict(base_modules)
        exec_func = lambda code, ns: exec(code, ns)

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

    # Reconstruct @nnsight.remote functions and classes
    for obj_name, obj_data in data.get('remote_objects', {}).items():
        # Add captured module-level references
        namespace.update(obj_data.get('module_refs', {}))

        # Add closure variables
        namespace.update(obj_data.get('closure_vars', {}))

        # Extract source code (handle both new and legacy format)
        source_data = obj_data.get('source', '')
        if isinstance(source_data, dict):
            source_code = source_data.get('code', '')
        else:
            source_code = source_data

        # Handle different types
        obj_type = obj_data.get('type', 'function')

        if obj_type == 'lambda':
            # Lambda: wrap in assignment and execute
            var_name = obj_data.get('var_name', obj_name)
            lambda_assignment = f"{var_name} = {source_code}"
            exec_func(lambda_assignment, namespace)
        else:
            # Function or class: execute definition directly
            exec_func(source_code, namespace)

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
        if isinstance(value, dict):
            if '__model_ref__' in value:
                result[key] = model
            elif '__remote_ref__' in value:
                # Reference to another remote object
                ref_id = str(value['__remote_ref__'])
                if ref_id in reconstructed:
                    result[key] = reconstructed[ref_id]
                else:
                    # Placeholder, will be resolved later if needed
                    result[key] = value
            elif '__callable_ref__' in value:
                # Reference to a function from allowed modules
                ref = value['__callable_ref__']
                parts = ref.rsplit('.', 1)
                if len(parts) == 2:
                    mod_name, func_name = parts
                    import importlib
                    try:
                        mod = importlib.import_module(mod_name)
                        result[key] = getattr(mod, func_name)
                    except (ImportError, AttributeError):
                        result[key] = value
                else:
                    result[key] = value
            elif '__tensor__' in value:
                # Tensor data
                result[key] = deserialize_tensor(value)
            else:
                result[key] = value
        else:
            result[key] = value

    return result
