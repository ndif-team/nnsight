# nnsight Source-Based Serialization Design Doc

## TL;DR

Replace cloudpickle with source-based serialization to eliminate Python version lock-in and enable third-party libraries to work with NDIF without server-side installation.

**Key changes:**
- New `@nnsight.remote` decorator for functions and classes used in traces
- Serialization sends source code + JSON instead of pickled bytecode
- Module aliases (`np`, `F`, `torch`) and constants auto-captured
- Backward compatible: falls back to cloudpickle with deprecation warning

**User-facing change:**
```python
@nnsight.remote  # Add this decorator
class MyAnalyzer:
    def __init__(self, model, top_k=10):
        self.model = model
        self.top_k = top_k

    def analyze(self, layer):
        h = self.model.transformer.h[layer].output[0]
        return h.topk(self.top_k)
```

**Benefits:**
- Works across Python versions (3.10 client ↔ 3.12 server)
- Libraries like nnterp work without NDIF server installation
- Validation at import time, not mysterious runtime failures

---

## Problem Statement

nnsight currently uses cloudpickle to serialize trace blocks for remote execution on NDIF. This creates several problems:

1. **Python version lock-in**: cloudpickle serializes bytecode, which is Python-version-specific. Clients must run the exact same Python version as NDIF servers.

2. **Illusory flexibility**: While cloudpickle appears to allow "arbitrary Python," in practice code can only use libraries installed on the server. The flexibility is more limited than it appears.

3. **No organic library growth**: Third-party libraries (like nnterp) cannot be used in traces without installing them on NDIF servers, requiring coordination with the NDIF team.

4. **Fragile failures**: Version mismatches and missing dependencies cause mysterious runtime errors rather than clear, early failures.

## Goals

1. **Version-agnostic serialization**: Clients on Python 3.10 should work with servers on Python 3.12+
2. **Enable organic library ecosystem**: Third-party libraries can work with NDIF without server-side installation
3. **Fail fast and loud**: Incompatible code should error at import time with clear messages
4. **Preserve expressiveness**: Loops, conditionals, and server-side computation should still work
5. **Minimize breaking changes**: Existing simple traces should continue to work

## Non-Goals

1. Support for arbitrary Python objects (we accept explicit constraints)
2. Support for libraries with C extensions in traces
3. Indefinite cloudpickle support (deprecated, removed in 2.0)

---

## Current Architecture

### What nnsight already does well

nnsight already captures source code as text:

```python
# tracing/base.py - capture()
source_lines, offset = inspect.getsourcelines(frame)

# tracing/base.py - parse()
tree = ast.parse("".join(source_lines))
# ... finds the with block via AST visitor

# tracing/base.py - compile()
self.info.source = [
    f"def {function_name}(__nnsight_tracer__, __nnsight_tracing_info__):\n",
    "    __nnsight_tracer__.pull()\n",
    *self.info.source,  # <-- Source as text!
    "    __nnsight_tracer__.push()\n",
]
```

### Where cloudpickle enters

Cloudpickle is used in `serialization.py` to serialize:

1. `tracer.info.frame` - the FrameType containing closure variables
2. Any objects referenced in the trace (via `pull()`)

```python
# serialization.py
import cloudpickle

def save(obj):
    CustomCloudPickler(file, protocol=4).dump(obj)  # <-- Here
```

### The pull/push mechanism

```python
# pull() - imports variables from original scope into trace execution
original_state = self.info.frame.f_locals
push_variables(current_frame, filtered_state)

# push() - exports variables back after execution
push_variables(target_frame, filtered_state)
```

This is where closure variables get transferred. Currently they're cloudpickled; we need to JSON-serialize them instead.

---

## Proposed Design

### Core Principle

**Libraries are query generators, not server-side code.**

```
┌─────────────────────────────────────────────────────┐
│  Client-side (pip installable, version-agnostic)   │
│                                                     │
│  User code                                          │
│    ↓                                                │
│  Third-party libs (nnterp, logitlenskit, etc.)     │
│    ↓                                                │
│  nnsight (generates traces)                         │
│    ↓                                                │
│  Serialized: source text + JSON variables           │
├─────────────────────────────────────────────────────┤
│  NDIF Server                                        │
│                                                     │
│  Receives: source + JSON                            │
│  Reconstructs: @nnsight.remote classes/functions    │
│  Executes: with torch, numpy, model access          │
│  Returns: requested tensors                         │
└─────────────────────────────────────────────────────┘
```

### New Serialization Format

```python
{
    "version": "2.0",
    "source": [
        "def __nnsight_tracer_12345__(...):\n",
        "    __nnsight_tracer__.pull()\n",
        "    h = model.layers[10].output[0]\n",
        "    result = (h @ model.lm_head.weight.T).topk(10).save()\n",
        "    __nnsight_tracer__.push()\n"
    ],
    "variables": {
        "threshold": 0.5,
        "layers": [10, 15, 20],
        "top_k": 10
    },
    "remote_objects": {
        "LogitLensKit": {
            "source": "class LogitLensKit:\n    ...",
            "module_refs": {"DEFAULT_TOP_K": 10},
            "instances": {
                "kit_12345": {
                    "var_name": "kit",
                    "state": {"top_k": 10, "layers": [10, 15, 20]}
                }
            }
        },
        "normalize": {
            "source": "def normalize(x):\n    return x / x.norm()",
            "module_refs": {}
        }
    },
    "model_refs": ["model", "kit.model"]
}
```

---

## The @nnsight.remote Decorator

### Purpose

Mark functions and classes as safe for remote execution. Validates at import time. One decorator for both.

### Usage

```python
import nnsight

# Works on functions
@nnsight.remote
def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)

@nnsight.remote
def project_to_vocab(h, unembed):
    return h @ unembed.T

# Works on classes
@nnsight.remote
class LogitLensKit:
    def __init__(self, model, top_k=10):
        self.model = model
        self.top_k = top_k

    def get_hidden(self, layer):
        return self.model.transformer.h[layer].output[0]

    def project(self, h):
        return h @ self.model.lm_head.weight.T
```

### Import-Time Validation

The decorator performs these checks when the function/class is defined:

```python
@nnsight.remote
def my_func():
    ...

@nnsight.remote
class MyClass:
    ...

# At decoration time:
# ✓ Source is available (inspect.getsource works)
# ✓ No disallowed imports (only torch, numpy, math, builtins)
# ✓ Module-level references are JSON-serializable (auto-captured)
# ✓ Base classes are @nnsight.remote or object (for classes)
# ✓ No metaclass (for classes)
# ✓ No __slots__ (for classes)
# ✓ All code uses only allowed operations
# ✓ No external side effects (file I/O, network, etc.)
```

### Module-Level Reference Handling

The decorator intelligently handles module-level references by checking what they resolve to:

```python
# my_library/analyzer.py

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_TOP_K = 10
VOCAB_SIZE = 50257
COMPLEX_CONFIG = SomeClass()  # Not JSON-serializable

@nnsight.remote
class Analyzer:
    def __init__(self, top_k=DEFAULT_TOP_K):
        self.top_k = top_k

    def analyze(self, h):
        # Module aliases - automatically recognized as allowed
        h_norm = h / np.linalg.norm(h, axis=-1, keepdims=True)
        logits = F.softmax(h_norm @ self.weights, dim=-1)

        # Constants - auto-captured
        return logits.topk(VOCAB_SIZE)

    def broken(self):
        return COMPLEX_CONFIG.value  # ERROR: not serializable
```

**How references are resolved:**

| Reference | Resolves to | Action |
|-----------|-------------|--------|
| `np` | `numpy` module | ✓ Skip (available on server) |
| `F` | `torch.nn.functional` module | ✓ Skip (available on server) |
| `torch` | `torch` module | ✓ Skip (available on server) |
| `DEFAULT_TOP_K` | `10` (int) | ✓ Capture: `{"DEFAULT_TOP_K": 10}` |
| `VOCAB_SIZE` | `50257` (int) | ✓ Capture: `{"VOCAB_SIZE": 50257}` |
| `COMPLEX_CONFIG` | `SomeClass` instance | ✗ Error: not serializable |

At decoration time:
```
@nnsight.remote validation for 'Analyzer':
  ✓ Module alias 'np' -> numpy (available on server)
  ✓ Module alias 'F' -> torch.nn.functional (available on server)
  ✓ Will capture constant: DEFAULT_TOP_K = 10
  ✓ Will capture constant: VOCAB_SIZE = 50257
  ✗ ERROR: Reference 'COMPLEX_CONFIG' (type 'SomeClass') is not JSON-serializable
    Options:
      - Make it a class/instance attribute instead
      - Pass it as a function/method argument
      - Use a JSON-serializable type (int, float, str, list, dict)
```

### Error Messages

```
ImportError: @nnsight.remote validation failed for 'MyClass':

  Line 3: imports 'pandas' (not available on NDIF server)
    Allowed imports: torch, numpy, math, builtins

  Line 12: references 'COMPLEX_CONFIG' (type 'SomeClass', not serializable)
    Options: make it an argument, or use JSON-serializable type

  Line 25: calls 'open()'
    @nnsight.remote code cannot perform file I/O
```

### Implementation Sketch

```python
# nnsight/remote.py

import ast
import inspect
import sys
from typing import Union, Type, Callable

ALLOWED_MODULES = {'torch', 'numpy', 'math', 'builtins'}
DISALLOWED_CALLS = {'open', 'exec', 'eval', 'compile', 'input'}

def remote(obj: Union[Type, Callable]) -> Union[Type, Callable]:
    """Decorator that marks a function or class as safe for NDIF remote execution."""

    # Step 1: Verify source is available
    try:
        source = inspect.getsource(obj)
    except OSError:
        raise ImportError(
            f"@nnsight.remote requires source code for '{obj.__name__}'. "
            f"Ensure .py files are included in your package distribution."
        )

    # Step 2: Parse AST
    tree = ast.parse(source)

    # Step 3: Find module-level references and validate
    module = sys.modules.get(obj.__module__)
    external_names = find_external_references(tree, obj)
    module_refs, errors = resolve_module_references(external_names, module)

    # Step 4: Validate AST for disallowed patterns
    errors.extend(validate_ast(tree, obj.__name__))

    # Step 5: For classes, additional validation
    if isinstance(obj, type):
        errors.extend(validate_class(obj))

    if errors:
        raise ImportError(format_validation_errors(obj.__name__, errors))

    # Step 6: Store metadata for serialization
    obj._remote_source = source
    obj._remote_module_refs = module_refs
    obj._remote_validated = True

    return obj


def find_external_references(tree: ast.AST, obj) -> set:
    """Find names referenced but not defined locally."""
    # ... AST analysis to find Name nodes not in local scope
    pass


def resolve_module_references(names: set, obj) -> tuple:
    """
    Resolve external names to their values from the function/class globals.
    Returns (captured_refs, errors).

    Handles three cases:
    1. Module aliases (np, F, torch) -> skip if allowed module
    2. JSON-serializable constants (TOP_K = 10) -> capture
    3. Non-serializable objects -> error
    """
    import types
    captured = {}
    errors = []

    for name in names:
        # Skip builtins
        if name in dir(__builtins__):
            continue

        # Get the actual value from globals
        value = obj.__globals__.get(name)

        if value is None:
            errors.append(f"Unknown reference '{name}'")
            continue

        # Case 1: Module or module alias (np, F, torch, etc.)
        if isinstance(value, types.ModuleType):
            root = value.__name__.split('.')[0]
            if root in ALLOWED_MODULES:
                continue  # Available on server, no need to capture
            else:
                errors.append(
                    f"Module '{name}' ({value.__name__}) not available on NDIF server. "
                    f"Allowed: torch, numpy, math"
                )
            continue

        # Case 2: JSON-serializable constant
        if is_json_serializable(value):
            captured[name] = value
            continue

        # Case 3: Non-serializable object
        errors.append(
            f"Reference '{name}' (type '{type(value).__name__}') "
            f"is not JSON-serializable"
        )

    return captured, errors


def validate_ast(tree: ast.AST, name: str) -> list:
    """Validate AST for disallowed patterns."""
    errors = []

    class Validator(ast.NodeVisitor):
        def visit_Import(self, node):
            for alias in node.names:
                module = alias.name.split('.')[0]
                if module not in ALLOWED_MODULES:
                    errors.append((node.lineno, f"imports '{alias.name}'"))

        def visit_ImportFrom(self, node):
            module = (node.module or '').split('.')[0]
            if module not in ALLOWED_MODULES:
                errors.append((node.lineno, f"imports from '{node.module}'"))

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                if node.func.id in DISALLOWED_CALLS:
                    errors.append((node.lineno, f"calls '{node.func.id}()'"))
            self.generic_visit(node)

    Validator().visit(tree)
    return errors


def validate_class(cls: type) -> list:
    """Additional validation for classes."""
    errors = []

    # Check base classes
    for base in cls.__bases__:
        if base is not object and not getattr(base, '_remote_validated', False):
            errors.append(f"Base class '{base.__name__}' is not @nnsight.remote")

    # Check for metaclass
    if type(cls) is not type:
        errors.append(f"Uses metaclass '{type(cls).__name__}'")

    # Check for __slots__
    if hasattr(cls, '__slots__'):
        errors.append("Uses __slots__")

    return errors
```

---

## Serialization Changes

### New serialization.py

```python
# nnsight/intervention/serialization_v2.py

import ast
import inspect
import json
from typing import Any, Dict, Union, Callable

class SerializationError(Exception):
    """Raised when an object cannot be serialized for remote execution."""
    pass


def serialize_for_remote(tracer) -> bytes:
    """Serialize a tracer for remote execution using source + JSON."""

    source = tracer.info.source
    variables = extract_variables(tracer.info.frame.f_locals)
    remote_objects = extract_remote_objects(tracer.info.frame.f_locals)

    payload = {
        "version": "2.0",
        "source": source,
        "variables": variables,
        "remote_objects": remote_objects,
    }

    return json.dumps(payload).encode('utf-8')


def extract_variables(locals_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract JSON-serializable variables from locals."""
    result = {}

    for name, value in locals_dict.items():
        if name.startswith('__'):
            continue

        if is_json_serializable(value):
            result[name] = value
        elif is_remote_object(value):
            result[name] = {"__remote_ref__": id(value)}
        elif is_model_reference(value):
            result[name] = {"__model_ref__": True}
        else:
            raise SerializationError(
                f"Variable '{name}' of type '{type(value).__name__}' "
                f"cannot be serialized.\n"
                f"Options:\n"
                f"  - Use JSON-serializable type (int, float, str, list, dict)\n"
                f"  - Mark with @nnsight.remote\n"
                f"  - Compute inside the trace block"
            )

    return result


def is_remote_object(obj: Any) -> bool:
    """Check if obj is a @nnsight.remote function/class or instance thereof."""
    if callable(obj) and getattr(obj, '_remote_validated', False):
        return True
    if getattr(type(obj), '_remote_validated', False):
        return True
    return False


def extract_remote_objects(locals_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract @nnsight.remote functions, classes, and instances."""
    result = {}

    for name, value in locals_dict.items():
        if not is_remote_object(value):
            continue

        # Get the class or function
        if isinstance(value, type):
            # It's a class itself
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

        if cls_name not in result:
            result[cls_name] = {
                "source": cls._remote_source,
                "module_refs": cls._remote_module_refs,
                "type": "class" if isinstance(cls, type) else "function",
                "instances": {}
            }

        if is_instance:
            instance_state = serialize_instance_state(value)
            result[cls_name]["instances"][str(id(value))] = {
                "var_name": name,
                "state": instance_state
            }

    return result


def serialize_instance_state(obj: Any) -> Dict[str, Any]:
    """Serialize a @nnsight.remote instance's __dict__."""
    state = {}

    for key, value in obj.__dict__.items():
        if is_json_serializable(value):
            state[key] = value
        elif is_model_reference(value):
            state[key] = {"__model_ref__": True}
        elif is_remote_object(value):
            state[key] = {"__remote_ref__": id(value)}
        else:
            raise SerializationError(
                f"Instance attribute '{key}' of type '{type(value).__name__}' "
                f"cannot be serialized."
            )

    return state
```

---

## Server-Side Changes

### Deserialization

```python
# Server-side: reconstruct and execute

import json
import torch
import numpy

def deserialize_and_execute(payload: bytes, model) -> Any:
    data = json.loads(payload.decode('utf-8'))

    # Build namespace with allowed modules
    namespace = {
        'torch': torch,
        'numpy': numpy,
        'np': numpy,
        'model': model,
    }

    # Reconstruct @nnsight.remote functions and classes
    for obj_name, obj_data in data.get('remote_objects', {}).items():
        # Add captured module-level references to namespace
        namespace.update(obj_data.get('module_refs', {}))

        # Execute function/class definition
        exec(obj_data['source'], namespace)

        # For classes, reconstruct instances
        if obj_data['type'] == 'class':
            cls = namespace[obj_name]
            for instance_id, instance_data in obj_data.get('instances', {}).items():
                obj = object.__new__(cls)
                obj.__dict__ = reconstruct_state(instance_data['state'], namespace)
                namespace[instance_data['var_name']] = obj

    # Add simple variables
    namespace.update(resolve_refs(data.get('variables', {}), namespace))

    # Compile and execute trace source
    source = ''.join(data['source'])
    code = compile(source, '<nnsight-remote>', 'exec')
    exec(code, namespace)

    return namespace.get('__nnsight_result__')


def reconstruct_state(state: dict, namespace: dict) -> dict:
    """Reconstruct instance state, resolving references."""
    result = {}
    for key, value in state.items():
        if isinstance(value, dict):
            if '__model_ref__' in value:
                result[key] = namespace['model']
            elif '__remote_ref__' in value:
                # Reference to another remote object, resolve by id
                result[key] = value  # TODO: proper resolution
            else:
                result[key] = value
        else:
            result[key] = value
    return result
```

---

## Migration Path

### Phase 1: Add new serialization (non-breaking)

1. Implement `@nnsight.remote` decorator
2. Implement new serialization format
3. Add `serialization_version` header to requests
4. Server supports both old (cloudpickle) and new (source+JSON) formats

### Phase 2: Encourage adoption

1. Add warnings when using non-remote objects in traces
2. Document the new approach
3. Update nnterp and other libraries to use `@nnsight.remote`

### Phase 3: Deprecate cloudpickle

1. Emit deprecation warnings for cloudpickle serialization
2. Set timeline for removal
3. Eventually require source-based serialization

---

## Backward Compatibility

During the transition period, nnsight will maintain full backward compatibility by falling back to cloudpickle when source-based serialization isn't possible.

### Client-Side: Try New Path, Fall Back with Warning

```python
# nnsight/intervention/serialization.py

import warnings
from typing import Tuple

def save(tracer) -> Tuple[bytes, str]:
    """
    Serialize tracer for remote execution.

    Attempts source-based serialization first. Falls back to cloudpickle
    with a deprecation warning if any variables can't be serialized.

    Returns:
        Tuple of (serialized_bytes, format_string)
        format_string is either "source" or "cloudpickle"
    """
    try:
        return serialize_source_based(tracer), "source"
    except SerializationError as e:
        warnings.warn(
            f"Falling back to cloudpickle serialization:\n"
            f"  {e}\n\n"
            f"This requires matching Python versions between client and server.\n"
            f"To use version-agnostic serialization:\n"
            f"  - Use JSON-serializable variables (int, float, str, list, dict)\n"
            f"  - Mark functions and classes with @nnsight.remote\n\n"
            f"Cloudpickle fallback will be removed in nnsight 2.0.",
            DeprecationWarning,
            stacklevel=4
        )
        return serialize_cloudpickle(tracer), "cloudpickle"
```

### Request Header Indicates Format

```python
# nnsight/intervention/backends/remote.py

def request(self, tracer) -> Tuple[bytes, Dict[str, str]]:
    interventions = super().__call__(tracer)

    # Serialize with automatic fallback
    data, serialization_format = save(tracer)

    headers = {
        "nnsight-model-key": self.model_key,
        "nnsight-version": __version__,
        "nnsight-serialization": serialization_format,  # "source" or "cloudpickle"
        "python-version": python_version,  # Still sent for cloudpickle compat
        ...
    }

    return data, headers
```

### Server-Side: Route to Appropriate Deserializer

```python
# NDIF server

def handle_request(request_data: bytes, headers: Dict[str, str], model) -> Any:
    serialization_format = headers.get("nnsight-serialization", "cloudpickle")

    if serialization_format == "source":
        # New path: source + JSON
        return deserialize_source_based(request_data, model)
    else:
        # Legacy path: cloudpickle
        # Verify Python version match
        client_version = headers.get("python-version", "")
        if not versions_compatible(client_version, sys.version):
            raise RemoteException(
                f"Python version mismatch: client={client_version}, "
                f"server={sys.version}. Use @nnsight.remote for "
                f"version-agnostic serialization."
            )
        return deserialize_cloudpickle(request_data, model)
```

### What Users See

**Case 1: Simple trace with primitives only**
```python
with model.trace("Hello") as t:
    h = model.layers[10].output[0]
    result = h.mean().save()
# Uses source path silently, no warning
```

**Case 2: @nnsight.remote function**
```python
@nnsight.remote
def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)

with model.trace("Hello") as t:
    h = normalize(model.layers[10].output[0])
    result = h.mean().save()
# Uses source path silently, no warning
```

**Case 3: @nnsight.remote class**
```python
@nnsight.remote
class Analyzer:
    def __init__(self, top_k):
        self.top_k = top_k

analyzer = Analyzer(10)
with model.trace("Hello") as t:
    result = analyzer.analyze(h)
# Uses source path silently, no warning
```

**Case 4: Non-remote class (legacy code)**
```python
class LegacyAnalyzer:  # No decorator
    def __init__(self, top_k):
        self.top_k = top_k

analyzer = LegacyAnalyzer(10)
with model.trace("Hello") as t:
    result = analyzer.analyze(h)

# Warning:
# DeprecationWarning: Falling back to cloudpickle serialization:
#   Variable 'analyzer' of type 'LegacyAnalyzer' cannot be serialized.
#
# This requires matching Python versions between client and server.
# To use version-agnostic serialization:
#   - Use JSON-serializable variables (int, float, str, list, dict)
#   - Mark functions and classes with @nnsight.remote
#
# Cloudpickle fallback will be removed in nnsight 2.0.
```

**Case 5: @nnsight.remote on non-conforming code**
```python
@nnsight.remote
def bad_function():
    import pandas  # Not allowed
    return pandas.read_csv("data.csv")

# ImportError at import time (loud, fast):
# ImportError: @nnsight.remote validation failed for 'bad_function':
#   Line 2: imports 'pandas' (not available on NDIF server)
```

### Compatibility Matrix

| Scenario | Serialization | Warning | Works? |
|----------|---------------|---------|--------|
| Primitives only | source | No | ✅ |
| @nnsight.remote functions | source | No | ✅ |
| @nnsight.remote classes | source | No | ✅ |
| Non-decorated custom code | cloudpickle | Yes | ✅ (if Python matches) |
| Non-decorated + version mismatch | cloudpickle | Yes | ❌ (clear error) |
| @nnsight.remote with violations | N/A | N/A | ❌ (import-time error) |

### Gradual Migration Path

1. **Phase 1** (nnsight 1.x): Both paths work, warnings guide users
2. **Phase 2** (nnsight 1.x+): Increase warning visibility, add docs
3. **Phase 3** (nnsight 2.0): Remove cloudpickle, source-only

This allows existing code to keep working while nudging users toward the better path.

---

## Testing Strategy

### Unit Tests

```python
def test_remote_valid_function():
    @nnsight.remote
    def normalize(x):
        return x / x.norm()

    assert hasattr(normalize, '_remote_source')
    assert hasattr(normalize, '_remote_module_refs')

def test_remote_valid_class():
    @nnsight.remote
    class ValidClass:
        def __init__(self, x):
            self.x = x
        def compute(self, y):
            return self.x + y

    assert hasattr(ValidClass, '_remote_source')

def test_remote_captures_module_constants():
    # In a module with TOP_K = 10
    @nnsight.remote
    class Analyzer:
        def __init__(self, k=TOP_K):
            self.k = k

    assert Analyzer._remote_module_refs == {'TOP_K': 10}

def test_remote_rejects_external_import():
    with pytest.raises(ImportError, match="imports 'pandas'"):
        @nnsight.remote
        def bad():
            import pandas
            return pandas.DataFrame()

def test_remote_rejects_non_serializable_module_ref():
    # In a module with COMPLEX = SomeClass()
    with pytest.raises(ImportError, match="not JSON-serializable"):
        @nnsight.remote
        def bad():
            return COMPLEX.value

def test_serialization_round_trip():
    @nnsight.remote
    class Kit:
        def __init__(self, top_k):
            self.top_k = top_k

    kit = Kit(10)
    serialized = serialize_for_remote(mock_tracer_with(kit))
    reconstructed = deserialize(serialized)

    assert reconstructed.top_k == 10
```

### Integration Tests

```python
def test_remote_class_in_trace():
    @nnsight.remote
    class Analyzer:
        def __init__(self, model, layers):
            self.model = model
            self.layers = layers

        def get_hidden(self, layer):
            return self.model.transformer.h[layer].output[0]

    model = nnsight.LanguageModel("gpt2")
    analyzer = Analyzer(model, [5, 10, 15])

    with model.trace("Hello world", backend="remote"):
        results = []
        for layer in analyzer.layers:
            h = analyzer.get_hidden(layer)
            results.append(h.mean().save())

    assert len(results) == 3

def test_remote_function_in_trace():
    @nnsight.remote
    def normalize(x):
        return x / x.norm(dim=-1, keepdim=True)

    model = nnsight.LanguageModel("gpt2")

    with model.trace("Hello world", backend="remote"):
        h = model.transformer.h[10].output[0]
        h_norm = normalize(h)
        result = h_norm.mean().save()

    assert result is not None
```

---

## Open Questions

1. **Tensor serialization**: How do we handle tensors passed as variables? Use numpy array serialization? torch.save to bytes?

2. **Nested @nnsight.remote**: If class A contains an instance of class B, both need to be @nnsight.remote. How do we validate this at import time vs runtime?

3. **Lambda handling**: Should we support lambdas via source extraction? Or require explicit functions?

4. **Error recovery**: If server-side execution fails, how much context can we provide given we only have source?

5. **Caching**: Can we cache compiled class/function definitions on the server to avoid re-parsing?

6. **Versioning**: How do we handle @nnsight.remote code that changes between library versions?

---

## Appendix: Allowed Operations in @nnsight.remote

### Always allowed
- Python builtins (`len`, `range`, `zip`, `enumerate`, `hasattr`, `getattr`, `isinstance`, etc.)
- Control flow (`if`, `for`, `while`, `try/except`)
- Arithmetic and comparison operators
- Attribute access and introspection
- Method calls
- Object construction

**Note:** Introspection of the model (e.g., `hasattr(model, 'transformer')`, `model.config.hidden_size`) is fully allowed. This enables libraries like nnterp to detect model architecture dynamically, whether that detection runs client-side or server-side.

### Allowed modules
- `torch.*`
- `numpy.*` (including alias `np`)
- `math.*`

### Disallowed
- File I/O (`open`, `Path.read_text`, etc.)
- Network (`requests`, `urllib`, `socket`)
- Subprocess (`os.system`, `subprocess`)
- Dynamic code (`exec`, `eval`, `compile`)
- Threading (`threading`, `multiprocessing`)
- Non-allowed imports (e.g., `pandas`, `sklearn`)

---

## Summary

This design replaces cloudpickle with source-based serialization:

| Aspect | Current (cloudpickle) | Proposed (source+JSON) |
|--------|----------------------|------------------------|
| Python version | Must match exactly | Any version |
| Third-party libs | Must be on server | @nnsight.remote works anywhere |
| Failure mode | Runtime, mysterious | Import-time, clear |
| Flexibility | Appears unlimited | Explicitly constrained |
| Library ecosystem | Requires NDIF coordination | Organic growth |

The key insight is that nnsight already captures source as text. We're simply changing the serialization of closure variables from cloudpickle to JSON, with the `@nnsight.remote` decorator providing validation and enabling classes/functions to be shipped with their source.

**One decorator for everything**: `@nnsight.remote` works on both functions and classes. Module-level constants are auto-captured if JSON-serializable. The mental model is simple: "Mark things you use in traces with `@nnsight.remote`."
