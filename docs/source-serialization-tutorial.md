# Source Serialization: A Tutorial for Engineers

This document is a tutorial introduction to nnsight's source-based serialization system. It's designed to be read before or while exploring the code, and provides the context, mental models, and architectural understanding you'll need to work with this subsystem effectively.

---

## Table of Contents

1. [The Problem We're Solving](#the-problem-were-solving)
2. [The Key Insight](#the-key-insight)
3. [Architecture Overview](#architecture-overview)
4. [The `@remote` Decorator](#the-remote-decorator)
5. [Serialization Format](#serialization-format)
6. [The Serialization Pipeline](#the-serialization-pipeline)
7. [The Deserialization Pipeline](#the-deserialization-pipeline)
8. [Handling Special Cases](#handling-special-cases)
9. [Design Decisions and Trade-offs](#design-decisions-and-trade-offs)
10. [Code Layout](#code-layout)
11. [Testing Strategy](#testing-strategy)
12. [Common Patterns and Idioms](#common-patterns-and-idioms)
13. [Debugging Tips](#debugging-tips)

---

## The Problem We're Solving

### What is nnsight?

nnsight lets researchers inspect and modify neural network internals during inference. A typical usage looks like:

```python
with model.trace("Hello world"):
    hidden = model.layers[10].output[0]
    result = hidden.topk(10).save()
```

The code inside the `trace` block doesn't execute immediately—it builds a symbolic computation graph that's executed later.

### The Remote Execution Problem

When `remote=True`, this computation runs on NDIF (a shared GPU cluster):

```python
with model.trace("Hello world", remote=True):
    # This code needs to be sent to a remote server
    hidden = model.layers[10].output[0]
    result = hidden.topk(10).save()
```

The challenge: **How do you send Python code to a remote server?**

### The Cloudpickle Era (Legacy)

Previously, nnsight used cloudpickle, which serializes Python bytecode. This had severe limitations:

| Problem | Impact |
|---------|--------|
| **Version lock-in** | Client and server must run identical Python versions (3.10 bytecode won't run on 3.12) |
| **Library installation** | Every library used in traces must be installed on NDIF servers |
| **Mysterious failures** | Code that serializes fine may fail at runtime with cryptic errors |
| **No validation** | Problems only surface when code actually runs |

### The Third-Party Library Problem

Consider a helper library like `nnterp`:

```python
from nnterp import StandardizedTransformer

st = StandardizedTransformer(model)
with model.trace("Hello", remote=True):
    h = st.get_hidden(10)  # This uses nnterp's code
```

With cloudpickle, this only works if nnterp is installed on NDIF servers. Every third-party library requires server-side deployment coordination—the ecosystem can't grow organically.

---

## The Key Insight

The breakthrough comes from understanding what helper libraries actually do inside traces: **they're query generators, not runtime computation**.

When `st.get_hidden(10)` runs inside a trace, it's not actually fetching hidden states—it's building a symbolic graph. All the Python code used by the trace, along with its dependencies, acts as a DSL for constructing these graphs.

This means we can:
1. **Identify** all the needed helper classes and functions
2. **Extract** the source code of those dependencies
3. **Send** that source to the server
4. **Reconstruct** the classes there
5. **Execute** the trace using the reconstructed classes

Since Python source code is version-independent (unlike bytecode), this eliminates version lock-in. And since we're sending source, third-party libraries work without server installation.

---

## Architecture Overview

### Two Worlds

The serialization system spans two execution environments:

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT SIDE                              │
│                                                                  │
│   User Code              Serialization              Network     │
│  ┌─────────────┐        ┌─────────────┐         ┌──────────┐   │
│  │ @remote     │───────▶│ extract_all │────────▶│ JSON     │───┼──▶
│  │ classes     │        │ serialize   │         │ payload  │   │
│  │ & functions │        └─────────────┘         └──────────┘   │
│  └─────────────┘                                                │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                         SERVER SIDE                              │
│                                                                  │
│   Network          Deserialization            Execution          │
│  ┌──────────┐     ┌─────────────────┐     ┌─────────────────┐   │
│  │ JSON     │────▶│ deserialize_    │────▶│ Reconstructed   │   │
│  │ payload  │     │ source_based    │     │ classes/funcs   │   │
│  └──────────┘     └─────────────────┘     │ + namespace     │   │
│                                            └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Location | Role |
|-----------|----------|------|
| `@remote` decorator | `remote.py` | Marks classes/functions for source serialization; validates at import time |
| `serialize_source_based()` | `serialization_source.py` | Client-side: extracts and packages everything |
| `deserialize_source_based()` | `serialization_source.py` | Server-side: reconstructs namespace from payload |
| `extract_all()` | `serialization_source.py` | Walks frame locals, extracts remote objects, tensors, variables |

---

## The `@remote` Decorator

### What It Does

The `@remote` decorator is the user-facing API for source serialization:

```python
from nnsight import remote

@remote
class MyAnalyzer:
    def __init__(self, top_k=10):
        self.top_k = top_k

    def analyze(self, hidden):
        return hidden.topk(self.top_k)
```

When Python imports this module, the decorator:

1. **Extracts source code** using `inspect.getsource()`
2. **Validates the AST** for prohibited patterns (relative imports, `exec`, `eval`)
3. **Captures external references** (module aliases, constants)
4. **Stores metadata** on the class/function

### The Metadata Contract

After decoration, the class has these attributes:

```python
MyAnalyzer._remote_source      # str: the source code
MyAnalyzer._remote_module_refs # dict: {"np": "numpy", "F": "torch.nn.functional", ...}
MyAnalyzer._remote_closure     # dict: captured closure variables
MyAnalyzer._is_remote          # bool: True (marker for detection)
```

### Why Validate at Import Time?

Early validation is a core design principle. Instead of:

```
User writes code → Serializes → Sends to server → Server fails → User confused
```

We want:

```
User writes code → Import fails with clear error → User fixes immediately
```

This is why the decorator raises exceptions for things like:
- Relative imports (`from .utils import helper`)
- Dynamic code (`exec()`, `eval()`)
- Closures over non-serializable values

---

## Serialization Format

### The Payload Structure

The JSON payload has this shape:

```python
{
    "version": "2.2",

    # Main entry point source (the trace block code)
    "source": {
        "code": "...",
        "file": "/path/to/user/script.py",
        "line": 42
    },

    # All @remote decorated objects found in the trace
    "remote_objects": {
        "MyAnalyzer": {
            "type": "class",  # or "function", "lambda"
            "source": {"code": "...", "file": "...", "line": ...},
            "module_refs": {"np": "numpy", "torch": "torch"},
            "closure_vars": {"CONSTANT": 42},
            "instances": {
                "140234567890": {  # id(instance)
                    "var_name": "analyzer",
                    "state": {...}  # serialized __dict__
                }
            }
        }
    },

    # Simple variables from frame locals
    "variables": {
        "threshold": 0.5,
        "layer_idx": 10
    },

    # References to the model
    "model_refs": ["model", "m"],

    # For LanguageModel subclasses (like nnterp's StandardizedTransformer)
    "model_subclass": {
        "class_name": "StandardizedTransformer",
        "discovered_classes": {...},
        "state": {...}
    }
}
```

### Type Markers

Complex types use marker keys for unambiguous reconstruction:

```python
# Tensor
{"__tensor__": {"data": "base64...", "dtype": "float32", "shape": [768]}}

# Reference to already-serialized object (deduplication)
{"__ref__": "140234567890"}

# Reference to the model
{"__model_ref__": True}

# nn.Module (just the state_dict)
{"__nn_module__": {"class": "Linear", "state_dict": {...}}}
```

The marker constants are defined at the top of `serialization_source.py`:

```python
TENSOR_MARKER = "__tensor__"
REF_MARKER = "__ref__"
MODEL_REF_MARKER = "__model_ref__"
# etc.
```

---

## The Serialization Pipeline

### Entry Point: `serialize_source_based()`

```python
def serialize_source_based(tracer: Tracer) -> bytes:
    """Client-side serialization entry point."""
```

The pipeline:

```
1. Get frame locals from tracer
         │
         ▼
2. extract_all() - Walk locals, categorize values
         │
         ├── Remote objects → extract_remote_object()
         ├── Lambdas → extract_lambda_object()
         ├── Tensors → serialize_tensor()
         ├── Model refs → record in model_refs
         └── JSON-safe → store in variables
         │
         ▼
3. Handle model subclasses (if any)
         │
         ▼
4. Build payload dict
         │
         ▼
5. json.dumps().encode('utf-8')
```

### `extract_all()`: The Classification Engine

This function walks the frame locals and decides what to do with each value:

```python
def extract_all(
    frame_locals: Dict[str, Any],
    traced_model: Any = None,
) -> Dict[str, Any]:
```

The classification logic (simplified):

```python
for var_name, value in frame_locals.items():
    if is_the_traced_model(value, traced_model):
        # It's the model itself
        result['model_refs'].append(var_name)

    elif is_remote_object(value):
        # @remote decorated class or function
        extract_remote_object(var_name, value, result)

    elif is_lambda(value):
        # Lambda expression
        extract_lambda_object(var_name, value, result)

    elif is_json_serializable(value):
        # Primitives, lists, dicts of primitives
        result['variables'][var_name] = value

    elif is_tensor(value):
        # torch.Tensor or numpy.ndarray
        result['variables'][var_name] = serialize_tensor(value)

    # ... more cases
```

### Tensor Serialization

Tensors are base64-encoded with optional compression:

```python
def serialize_tensor(tensor) -> dict:
    # Convert to numpy, then to bytes
    arr = tensor.detach().cpu().numpy()
    data = arr.tobytes()

    # Compress if it saves space
    compressed = zlib.compress(data)
    if len(compressed) < len(data) * COMPRESSION_THRESHOLD:
        data = compressed
        is_compressed = True

    return {
        "__tensor__": {
            "data": base64.b64encode(data).decode('ascii'),
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "compressed": is_compressed,
        }
    }
```

### Instance State Serialization

For instances of `@remote` classes, we serialize their `__dict__`:

```python
def serialize_instance_state(instance, discovered_classes, traced_model):
    state = {}
    for key, value in instance.__dict__.items():
        state[key] = serialize_value(value, ...)
    return state
```

The `serialize_value()` function recursively handles nested structures, maintaining a `seen` dict for cycle detection.

---

## The Deserialization Pipeline

### Entry Point: `deserialize_source_based()`

```python
def deserialize_source_based(
    payload: bytes,
    model: Any,
    user_id: Optional[str] = None,
    job_id: Optional[str] = None,
    use_restricted: bool = False,
) -> Dict[str, Any]:
    """Server-side: reconstruct namespace from payload."""
```

The pipeline:

```
1. json.loads(payload)
         │
         ▼
2. Build base namespace (torch, numpy, model, etc.)
         │
         ▼
3. Handle model subclass (if present)
         │
         ▼
4. For each remote_object:
   │  ├── Add module_refs to namespace
   │  ├── Add closure_vars to namespace
   │  └── exec() the source code
   │      └── For classes: reconstruct instances
         │
         ▼
5. Deserialize variables (including tensors)
         │
         ▼
6. Return namespace dict
```

### Code Execution with Source Info

A key feature is preserving original file/line numbers in error tracebacks:

```python
def _exec_with_source_info(source_code, namespace, source_file, start_line):
    """Execute code with proper traceback info."""
    tree = ast.parse(source_code)
    ast.increment_lineno(tree, start_line - 1)  # Adjust line numbers
    code_obj = compile(tree, source_file, 'exec')  # Use original filename
    exec(code_obj, namespace)
```

This means when an error occurs, the traceback shows:

```
File "/home/user/research/analysis.py", line 47, in analyze
    return hidden.topk(self.invalid_attr)
AttributeError: 'MyAnalyzer' object has no attribute 'invalid_attr'
```

Rather than the unhelpful:

```
File "<string>", line 3
```

### Instance Reconstruction

Instances are reconstructed without calling `__init__`:

```python
# Create empty instance
obj = object.__new__(cls)

# Restore state
obj.__dict__ = reconstruct_state(serialized_state, ...)
```

This is important because `__init__` might have side effects or require arguments we don't have.

---

## Handling Special Cases

### Auto-Discovery

When a `LanguageModel` subclass (like nnterp's `StandardizedTransformer`) is used, we need to serialize not just the class but all its dependencies—base classes, helper functions, etc.

```python
def auto_discover_class(cls, discovered):
    """Recursively discover all dependencies of a class."""
    # Already discovered?
    if cls.__name__ in discovered:
        return

    # Get source and parse AST
    source = inspect.getsource(cls)
    tree = ast.parse(source)

    # Find external references
    refs = find_external_references(tree)

    # Resolve each reference
    for ref in refs:
        resolved = resolve_in_module(ref, cls.__module__)
        if needs_serialization(resolved):
            auto_discover_class(resolved, discovered)

    # Add this class
    discovered[cls.__name__] = _get_remote_metadata(cls)
```

### Cycle Detection

Objects can reference each other circularly:

```python
a.other = b
b.other = a  # Circular!
```

We handle this with ID-based deduplication:

```python
def serialize_value(value, seen):
    obj_id = id(value)

    if obj_id in seen:
        # Already serialized, emit a reference
        return {"__ref__": str(obj_id)}

    seen[obj_id] = True

    # Serialize with __id__ so we can reference it later
    result = {"__id__": str(obj_id), ...actual serialization...}
    return result
```

On deserialization, we maintain `reconstructed_instances[id] = obj` and resolve `__ref__` markers.

### Lambda Expressions

Lambdas are tricky because `inspect.getsource()` returns the whole line they're on:

```python
items = sorted(data, key=lambda x: x.score)  # getsource returns this entire line
```

We use bytecode analysis to disambiguate:

```python
def extract_lambda_source(func):
    # Get the full line
    source_line = inspect.getsource(func).strip()

    # Parse to find all lambdas
    candidates = find_all_lambdas(source_line)

    if len(candidates) == 1:
        return candidates[0]

    # Multiple lambdas on same line - use bytecode comparison
    for candidate in candidates:
        test_func = eval(candidate)
        if bytecode_matches(test_func, func):
            return candidate

    raise LambdaExtractionError("Cannot disambiguate")
```

### Server-Available Modules

Some modules are guaranteed to be on the server (torch, numpy, etc.). We don't need to serialize code that only uses these:

```python
SERVER_AVAILABLE_MODULES = {
    'torch', 'numpy', 'math', 'functools', 'itertools',
    'collections', 'typing', 'dataclasses', ...
}
```

When we find a reference to these, we record an import instruction rather than extracting source:

```python
"server_imports": {
    "F": {"type": "module", "module": "torch.nn.functional"},
    "Linear": {"type": "class", "module": "torch.nn", "name": "Linear"}
}
```

---

## Design Decisions and Trade-offs

### Why JSON Instead of a Binary Format?

**Pros of JSON:**
- Human-readable (crucial for debugging)
- Standard, well-supported everywhere
- No schema versioning issues

**Cons:**
- ~33% overhead for base64-encoded tensors
- Slower parsing than binary formats

**Verdict:** Debuggability wins for now. Future optimization: "sidecar" binary format for tensors only.

### Why Not Use `ast.unparse()`?

We could theoretically modify the AST and regenerate source. But:

1. `ast.unparse()` loses comments and formatting
2. Debugging becomes harder (source doesn't match what user wrote)
3. The original source is known to work; transformed source might have bugs

We store original source and only use AST for validation and analysis.

### Why Validate at Import Time?

An alternative is lazy validation at serialization time. But:

1. Import-time errors are closer to where the problem is
2. Developers fix issues before committing code
3. CI catches problems before deployment

The small startup cost is worth the debugging time saved.

### Why Reconstruct Without `__init__`?

Calling `__init__` would require:
- Serializing all constructor arguments
- Handling side effects (logging, file creation, etc.)
- Complex dependency ordering

Instead, we use `object.__new__()` + direct `__dict__` assignment. This is how `pickle` works.

**Limitation:** Classes that rely on `__init__` for invariants may not work correctly. But this is rare for the "query generator" pattern these classes follow.

---

## Code Layout

### File: `src/nnsight/remote.py`

The user-facing `@remote` decorator and related utilities:

```
remote.py
├── remote() decorator
├── is_remote_object() - detection
├── Validation functions
│   ├── validate_ast()
│   ├── find_external_references()
│   └── resolve_module_references()
├── Lambda handling
│   ├── is_lambda()
│   ├── extract_lambda_source()
│   └── validate_lambda_for_remote()
└── Constants
    ├── ALLOWED_MODULES
    ├── SERVER_AVAILABLE_MODULES
    └── ALLOWED_BASE_CLASSES
```

### File: `src/nnsight/intervention/serialization_source.py`

The serialization/deserialization engine:

```
serialization_source.py
├── Constants (TENSOR_MARKER, REF_MARKER, etc.)
│
├── Metadata Extraction
│   └── _get_remote_metadata()
│
├── Auto-Discovery
│   └── auto_discover_class()
│
├── Tensor Handling
│   ├── serialize_tensor()
│   └── deserialize_tensor()
│
├── Model Subclass Handling
│   ├── extract_model_subclass()
│   └── reconstruct_model_subclass()
│
├── Remote Object Extraction
│   ├── extract_remote_object()
│   └── extract_lambda_object()
│
├── Value Serialization
│   ├── serialize_value()
│   └── serialize_instance_state()
│
├── Top-Level API
│   ├── serialize_source_based()
│   └── deserialize_source_based()
│
└── Value Reconstruction
    ├── reconstruct_value()
    └── reconstruct_state()
```

### File: `src/nnsight/intervention/restricted_execution.py`

Security sandbox for server-side code execution:

```
restricted_execution.py
├── compile_user_code() - RestrictedPython wrapper
├── create_restricted_globals() - Safe namespace
├── Guarded operations
│   ├── _guarded_getattr()
│   ├── _guarded_getitem()
│   └── _guarded_import()
└── Constants
    ├── BLOCKED_MODULES
    ├── SUSPICIOUS_ATTRS
    └── DEFAULT_ALLOWED_MODULES
```

---

## Testing Strategy

### Test Files

| File | Purpose |
|------|---------|
| `test_remote.py` | Core remote decorator, serialization, restricted execution |
| `test_serialization_edge_cases.py` | Edge cases: cycles, nested structures, special types |

### Test Categories

**Unit tests** for individual functions:
```python
def test_serialize_tensor_basic():
    t = torch.tensor([1.0, 2.0, 3.0])
    serialized = serialize_tensor(t)
    assert "__tensor__" in serialized
```

**Round-trip tests** for end-to-end correctness:
```python
def test_roundtrip_with_closure():
    threshold = 0.5  # Captured in closure

    @remote
    def analyze(x):
        return x > threshold

    # Serialize → Deserialize → Execute
    payload = serialize_source_based(...)
    namespace = deserialize_source_based(payload, model)
    result = namespace['analyze'](0.7)
    assert result == True
```

**Error case tests** for validation:
```python
def test_relative_import_rejected():
    with pytest.raises(SourceSerializationError, match="relative import"):
        @remote
        class Bad:
            from .utils import helper  # Should fail
```

**Security tests** for restricted execution:
```python
def test_restricted_blocks_dunder():
    with pytest.raises(SyntaxError):
        compile_user_code("obj.__class__.__bases__")
```

---

## Common Patterns and Idioms

### Pattern: Type Dispatch

The code uses marker-based dispatch rather than `isinstance()`:

```python
# In serialization:
if is_tensor(value):
    return {TENSOR_MARKER: serialize_tensor_data(value)}
elif is_remote_instance(value):
    return {REMOTE_REF_MARKER: class_name, STATE_MARKER: state}

# In deserialization:
if TENSOR_MARKER in value:
    return deserialize_tensor(value)
elif REMOTE_REF_MARKER in value:
    return reconstruct_remote_instance(value)
```

### Pattern: ID-Based Deduplication

For object identity preservation:

```python
# Serialization
seen = {}
def serialize(obj):
    if id(obj) in seen:
        return {REF_MARKER: str(id(obj))}
    seen[id(obj)] = True
    return {ID_MARKER: str(id(obj)), ...data...}

# Deserialization
reconstructed = {}
def deserialize(data):
    if REF_MARKER in data:
        return reconstructed[data[REF_MARKER]]
    obj = ...create object...
    if ID_MARKER in data:
        reconstructed[data[ID_MARKER]] = obj
    return obj
```

### Pattern: Graceful Degradation

Handle legacy formats and missing data:

```python
source_data = obj_data.get('source', '')
if isinstance(source_data, dict):
    # New format with file/line info
    code = source_data.get('code', '')
    file = source_data.get('file', '<unknown>')
    line = source_data.get('line', 1)
else:
    # Legacy format: source is just a string
    code = source_data
    file = '<unknown>'
    line = 1
```

---

## Debugging Tips

### Inspecting Payloads

Add this temporarily to see what's being serialized:

```python
# In serialize_source_based():
payload_dict = {...}
import json
print(json.dumps(payload_dict, indent=2, default=str))  # default=str for non-JSON types
```

### Checking Decorator Metadata

```python
@remote
class MyClass:
    pass

# Inspect what was captured
print(MyClass._remote_source)
print(MyClass._remote_module_refs)
```

### Tracing Extraction

```python
# In extract_all():
for var_name, value in frame_locals.items():
    print(f"{var_name}: {type(value).__name__}")
    if is_remote_object(value):
        print(f"  -> remote object")
    elif is_tensor(value):
        print(f"  -> tensor {value.shape}")
```

### Watching Deserialization

```python
# In deserialize_source_based():
for obj_name, obj_data in data.get('remote_objects', {}).items():
    print(f"Reconstructing {obj_name} from {obj_data.get('source', {}).get('file', '?')}")
```

### Testing in Isolation

You can test serialization without an actual trace:

```python
from nnsight.intervention.serialization_source import extract_all, serialize_source_based

# Create fake frame locals
fake_locals = {
    'threshold': 0.5,
    'model': mock_model,
    'analyzer': my_analyzer_instance,
}

# Test extraction
result = extract_all(fake_locals, traced_model=mock_model)
print(result)
```

---

## Summary

The source serialization system enables Python-version-independent remote execution by:

1. **Extracting source code** instead of bytecode
2. **Validating early** at import time
3. **Capturing context** (module refs, closures, constants)
4. **Preserving identity** through ID-based deduplication
5. **Maintaining debuggability** with original file/line info

The key mental model: `@remote` decorated code is a **query generator**, not runtime code. We're shipping the query language (source) to the server, not compiled instructions (bytecode).

For detailed specifications and future roadmap, see `nnsight-source-serialization-design.md`.
