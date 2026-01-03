"""
Source-based serialization for nnsight remote execution.

This module provides serialization that uses source code + JSON instead of
cloudpickle bytecode. This enables:
- Python version independence (3.10 client can work with 3.12 server)
- Third-party libraries decorated with @nnsight.remote work without server installation
- Clear, early errors instead of mysterious runtime failures
"""

from __future__ import annotations

import json
import types
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..intervention.tracing.base import Tracer

from ..remote import is_json_serializable, ALLOWED_MODULES


class SourceSerializationError(Exception):
    """Raised when source-based serialization fails."""
    pass


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
    frame_locals = tracer.info.frame.f_locals if tracer.info.frame else {}

    # Extract variables and remote objects
    variables, remote_objects, model_refs = extract_all(frame_locals)

    payload = {
        "version": "2.0",
        "source": source,
        "variables": variables,
        "remote_objects": remote_objects,
        "model_refs": model_refs,
    }

    return json.dumps(payload).encode('utf-8')


def extract_all(locals_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """
    Extract all serializable data from locals.

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
            extract_remote_object(name, value, remote_objects)
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
        result[cls_name] = {
            "source": getattr(cls, '_remote_source', ''),
            "module_refs": getattr(cls, '_remote_module_refs', {}),
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


def serialize_instance_state(obj: Any) -> Dict[str, Any]:
    """
    Serialize a @nnsight.remote instance's __dict__.

    Args:
        obj: Instance of a @nnsight.remote class

    Returns:
        Dict containing the serialized state

    Raises:
        SourceSerializationError: If an attribute cannot be serialized
    """
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

        # JSON-serializable values
        if is_json_serializable(value):
            state[key] = value
            continue

        # Functions from allowed modules
        if callable(value) and hasattr(value, '__module__'):
            root = value.__module__.split('.')[0] if value.__module__ else ''
            if root in ALLOWED_MODULES:
                state[key] = {"__callable_ref__": f"{value.__module__}.{value.__name__}"}
                continue

        raise SourceSerializationError(
            f"Instance attribute '{key}' of type '{type(value).__name__}' "
            f"cannot be serialized for remote execution."
        )

    return state


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

def deserialize_source_based(payload: bytes, model: Any) -> Dict[str, Any]:
    """
    Deserialize a source-based payload and prepare for execution.

    This reconstructs @nnsight.remote classes and functions, their instances,
    and simple variables into a namespace suitable for exec().

    Args:
        payload: JSON-encoded bytes from serialize_source_based()
        model: The model object to inject into the namespace

    Returns:
        Namespace dict ready for code execution
    """
    import torch
    import numpy

    data = json.loads(payload.decode('utf-8'))

    # Build base namespace with allowed modules
    namespace = {
        'torch': torch,
        'numpy': numpy,
        'np': numpy,
        'model': model,
    }

    # Track reconstructed remote objects for cross-referencing
    reconstructed_instances = {}

    # Reconstruct @nnsight.remote functions and classes
    for obj_name, obj_data in data.get('remote_objects', {}).items():
        # Add captured module-level references
        namespace.update(obj_data.get('module_refs', {}))

        # Execute function/class definition
        exec(obj_data['source'], namespace)

        # For classes, reconstruct instances
        if obj_data['type'] == 'class':
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

    # Add simple variables
    namespace.update(data.get('variables', {}))

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
            else:
                result[key] = value
        else:
            result[key] = value

    return result
