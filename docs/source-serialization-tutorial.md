# Source Serialization: A Tutorial for Engineers

This document is a tutorial introduction to nnsight's source-based serialization system. It's designed to be read before or while exploring the code, and provides the context, mental models, and architectural understanding you'll need to work with this subsystem effectively.

---

## Table of Contents

1. [The Problem We're Solving](#the-problem-were-solving)
2. [The Key Insight](#the-key-insight)
3. [Architecture Overview](#architecture-overview)
4. [How Code Discovery Works](#how-code-discovery-works)
5. [Serialization Format](#serialization-format)
6. [The Serialization Pipeline](#the-serialization-pipeline)
7. [The Deserialization Pipeline](#the-deserialization-pipeline)
8. [Handling Special Cases](#handling-special-cases)
9. [The `@remote` Decorator (Optional)](#the-remote-decorator-optional)
10. [Design Decisions and Trade-offs](#design-decisions-and-trade-offs)
11. [Code Layout](#code-layout)
12. [Testing Strategy](#testing-strategy)
13. [Common Patterns and Idioms](#common-patterns-and-idioms)
14. [Debugging Tips](#debugging-tips)
15. [Summary](#summary)

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

### Why Not Just Use Pickle?

Python's built-in `pickle` module can serialize objects, but it has fundamental limitations for remote code execution:

**Traditional pickle** serializes object *state*, not *behavior*. When you pickle a class instance, pickle stores the instance's data and a reference to the class by name. To unpickle, the *same class must already exist* on the receiving end. Pickle doesn't transmit the class definition—it assumes the class is importable.

```python
# If MyClass isn't installed on the server, this fails:
pickle.dumps(MyClass())  # Works locally
pickle.loads(data)       # Fails: "ModuleNotFoundError: No module named 'my_module'"
```

This makes pickle unsuitable for our use case: we want users to define custom helper classes that execute on servers where those classes aren't installed.

### The Cloudpickle Alternative

**Cloudpickle** extends pickle to serialize functions and classes by capturing their bytecode. This *does* transmit behavior, not just state. However, it introduces severe problems:

| Problem | Impact |
|---------|--------|
| **Version lock-in** | Python bytecode is version-specific—code pickled on Python 3.10 may not run on Python 3.12 |
| **Library installation** | Dependencies referenced by the bytecode must still be installed on the server |
| **Mysterious failures** | Bytecode may serialize successfully but fail at runtime with cryptic errors |
| **No validation** | Problems only surface when code actually runs on the server |

nnsight's 0.5 implementation introduced cloudpickle for remote execution, but these limitations created constant friction. Users couldn't use helper libraries unless those libraries were pre-installed on NDIF servers—and even then, version mismatches caused subtle bugs.

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

Since Python source code is version-independent (unlike bytecode), this eliminates version lock-in.

**Third-party libraries do not need to be installed on the server**: all the necessary code is transmitted as part of the request. For example, transmitting nnterp's `StandardizedTransformer` and its dependencies adds about 36KB to the request—a small price for complete library independence.

---

## Architecture Overview

### Two Worlds

The serialization system spans two execution environments:

```
┌───────────────────────────────────────────────────────────────┐
│                         CLIENT SIDE                           │
│                                                               │
│  User Code            Serialization              Network      │
│  ┌─────────────┐     ┌─────────────┐        ┌──────────┐     │
│  │ User-defined│────▶│ extract_all │───────▶│ JSON     │─────┼──▶
│  │ classes     │     │ serialize   │        │ payload  │     │
│  │ & functions │     └─────────────┘        └──────────┘     │
│  └─────────────┘                                              │
└───────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                         SERVER SIDE                           │
│                                                               │
│  Network            Deserialization            Execution      │
│  ┌──────────┐     ┌─────────────────┐     ┌───────────────┐  │
│  │ JSON     │────▶│ deserialize_    │────▶│ Reconstructed │  │
│  │ payload  │     │ source_based    │     │ classes/funcs │  │
│  └──────────┘     └─────────────────┘     │ + namespace   │  │
│                                           └───────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Location | Role |
|-----------|----------|------|
| `serialize_source_based()` | `serialization_source.py` | Client-side: extracts and packages everything |
| `deserialize_source_based()` | `serialization_source.py` | Server-side: reconstructs namespace from payload |
| `extract_all()` | `serialization_source.py` | Walks frame locals, auto-discovers and extracts objects, tensors, variables |
| `@remote` decorator (optional) | `remote.py` | Explicitly marks classes/functions for transport; validates at import time |

---

## How Code Discovery Works

A key question for source serialization: **how does the system figure out which code to serialize?** This section explains the discovery algorithm step by step.

### The Starting Point: Frame Locals

When a user calls `model.trace(..., remote=True)`, the system captures the **frame locals**—all the variables in scope at the trace site:

```python
threshold = 0.5
analyzer = MyAnalyzer(top_k=10)

with model.trace("Hello", remote=True):
    h = model.layers[10].output[0]
    result = analyzer.analyze(h)  # Uses threshold internally
```

At serialization time, the frame locals include `threshold`, `analyzer`, and `model`. The entry point is `serialize_source_based()` in `serialization_source.py`, which calls `extract_all()` to walk through these locals.

### Classification: What Kind of Value Is This?

For each variable in frame locals, `extract_all()` must decide: **what is this, and how should I handle it?**

```
extract_all(frame_locals, traced_model)
    │
    ├── Is it the traced model itself?
    │   └── Record in model_refs (injected server-side)
    │
    ├── Is it JSON-serializable (int, float, str, list, dict)?
    │   └── Store directly in variables
    │
    ├── Is it a tensor (torch.Tensor, numpy.ndarray)?
    │   └── Call serialize_tensor() → base64 encoded data
    │
    ├── Is it from an allowed module (torch, numpy, etc.)?
    │   └── Skip (available on server)
    │
    ├── Is it a lambda function?
    │   └── Call extract_lambda_object() to extract lambda source
    │
    ├── Is it a user-defined class/function (auto-discovered or @remote)?
    │   └── Extract source code + dependencies
    │
    └── Otherwise (forbidden or no source available)?
        └── Raise SourceSerializationError
```

### Extracting User-Defined Objects: Source + Dependencies

When `extract_all()` finds a user-defined class or function (whether auto-discovered or explicitly decorated with `@remote`), it extracts the source code and dependencies. For `@remote`-decorated objects, metadata was pre-computed at decoration time. For auto-discovered objects, the same extraction happens at serialization time:

```python
def extract_remote_object(var_name, value, result, traced_model):
    # Get pre-validated metadata from the decorator
    metadata = _get_remote_metadata(value)

    # metadata contains:
    #   _remote_source: The source code
    #   _remote_module_refs: {"np": "numpy", "F": "torch.nn.functional"}
    #   _remote_closure_vars: {"threshold": 0.5}
    #   _remote_validated: True (passed import-time validation)

    # For instances, also serialize the instance state
    if is_instance_of_remote_class(value):
        state = serialize_instance_state(value, ...)
```

For `@remote`-decorated objects, most of the work happened at decoration time. For auto-discovered objects, extraction happens at serialization time using the same logic.

### Recursive Instance State Serialization

When a user-defined class instance is found, we need to serialize not just the class but the instance's state (`__dict__`). This can contain arbitrarily nested objects:

```python
def serialize_instance_state(obj, memo, discovered_classes, traced_model):
    state = {}
    for key, value in obj.__dict__.items():
        state[key] = serialize_value(value, memo, discovered_classes, traced_model)
    return state
```

The `serialize_value()` function handles recursive cases:
- **Tensors** → base64 encoded
- **Nested user-defined instances** → recurse
- **Collections (list, dict, tuple)** → recurse into elements
- **nn.Modules** → serialize class path + `__dict__`
- **Circular references** → use `memo` dict to detect and emit `__ref__` markers

### Understanding External References

Before diving into auto-discovery, it's important to understand a key concept: **external references** (also called **unbound variables** in programming language theory).

When we serialize a function or class, we need more than just its source code. The code may reference names that aren't defined locally—they come from the surrounding environment:

```python
import numpy as np
MAX_VALUE = 5.3
helper_fn = lambda x: x * 2

@remote
def analyze(hidden):
    # 'np' is an external reference (module alias)
    # 'MAX_VALUE' is an external reference (constant)
    # 'helper_fn' is an external reference (function)
    return np.clip(hidden, 0, MAX_VALUE)
```

In this example, `hidden` is a **locally bound** parameter, but `np`, `MAX_VALUE`, and `helper_fn` are **external references**—values from the environment that must be available for the code to run. Without them, execution would fail with `NameError`.

The serialization system identifies these external references and captures them in the payload:
- **Module references** like `np → "numpy"` go into the `module_refs` field, telling the server which modules to import
- **Closure variables** like `MAX_VALUE → 5.3` go into the `closure_vars` field, embedding the actual values

This distinction between code (source) and environment (external references) is what makes source serialization different from regular object serialization. We're not just sending bytes—we're reconstructing the complete execution context.

### Auto-Discovery: Third-Party Classes

Classes decorated with `@remote` have their external references captured at decoration time. But what about third-party classes that aren't decorated?

**Auto-discovery is enabled by default.** When you use any class or function with available source code, the system automatically discovers and serializes it—no `@remote` annotation required. This works for third-party libraries like nnterp, custom helper classes, and any code where `inspect.getsource()` can retrieve the source.

For users who prefer explicit control, pass `strict=True` to require `@remote` annotations. See [Auto-Discovery vs @remote Annotation](#auto-discovery-vs-remote-annotation) in Design Decisions for discussion of when to use each approach.

```python
def auto_discover_class(cls, discovered):
    # Skip if already discovered or server-available
    if cls.__name__ in discovered or is_server_available(cls):
        return

    # 1. Get source code
    source = inspect.getsource(cls)
    tree = ast.parse(source)

    # 2. Find all external names used in the source
    external_names = find_external_references(tree, cls)

    # 3. Resolve each name to a value and classify it
    for name in external_names:
        value = resolve_in_globals(name, cls)
        classification = classify_reference_value(name, value)

        if classification == ValueClassification.CAPTURE:
            # JSON-serializable constant → include in closure_vars
            discovered['closure_vars'][name] = value
        elif classification == ValueClassification.SKIP:
            # Module available on server → record in module_refs
            discovered['module_refs'][name] = get_module_path(value)
        elif classification == ValueClassification.ERROR:
            # Can't serialize → raise error
            raise SourceSerializationError(f"Cannot serialize {name}")

    # 4. Recursively discover base classes
    for base in cls.__bases__:
        if needs_discovery(base):
            auto_discover_class(base, discovered)

    # 5. Add this class to discovered
    discovered[cls.__name__] = {
        'source': source,
        'module_refs': ...,
        'closure_vars': ...
    }
```

### Finding External References via AST

The `find_external_references()` function uses AST analysis to find all names that aren't locally defined:

```python
class ReferenceCollector(ast.NodeVisitor):
    def __init__(self):
        self.scope_stack = [set()]  # Track local definitions
        self.external_refs = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            # This name is being read
            if not self.is_locally_defined(node.id):
                self.external_refs.add(node.id)

    def visit_FunctionDef(self, node):
        # Parameters are local to the function
        self.scope_stack.append(set(arg.arg for arg in node.args.args))
        self.generic_visit(node)
        self.scope_stack.pop()
```

For example, given:
```python
class MyAnalyzer:
    def analyze(self, x, threshold=DEFAULT_THRESHOLD):
        return F.softmax(x) if x.max() > threshold else x
```

The collector identifies external references: `DEFAULT_THRESHOLD`, `F` (for `torch.nn.functional`).

### Value Classification: Skip, Capture, or Error?

Once external names are found, `classify_reference_value()` decides what to do with each:

```python
def classify_reference_value(name, value):
    # Modules from server-available list → SKIP
    if is_module(value) and module_name(value) in SERVER_AVAILABLE_MODULES:
        return ValueClassification.SKIP

    # @remote decorated → SKIP (serialized separately)
    if is_remote_object(value):
        return ValueClassification.SKIP

    # JSON-serializable constants → CAPTURE
    if is_json_serializable(value):
        return ValueClassification.CAPTURE

    # Callable from allowed modules → SKIP
    if callable(value) and is_from_allowed_module(value):
        return ValueClassification.SKIP

    # Everything else → ERROR
    return ValueClassification.ERROR
```

### When Data vs Code Is Serialized

The system distinguishes between **code** (source to be exec'd) and **data** (values to be reconstructed):

| Type | Serialized As | Function |
|------|---------------|----------|
| `@remote` class/function | Source code + metadata | `extract_remote_object()` |
| Lambda expression | Extracted source | `extract_lambda_object()` |
| Auto-discovered class | Source code + dependencies | `auto_discover_class()` |
| `torch.Tensor` | Base64 bytes + dtype/shape | `serialize_tensor()` |
| `nn.Module` | Class path + `__dict__` | `serialize_nn_module()` |
| Primitives (int, str, etc.) | JSON value | Direct |
| Collections | Recursive with markers | `serialize_value()` |
| Instance state | Recursive `__dict__` | `serialize_instance_state()` |

### When Serialization Is Complete

Serialization completes when `serialize_source_based()` has processed all frame locals and built the final payload:

```python
def serialize_source_based(tracer):
    # 1. Extract source metadata (file, line info)
    source_metadata = extract_source_metadata(tracer)

    # 2. Categorize all frame locals
    variables, remote_objects, model_refs = extract_all(
        tracer.frame_locals,
        tracer.model
    )

    # 3. Handle LanguageModel subclasses (if any)
    model_subclass = None
    if is_language_model_subclass(tracer.model):
        model_subclass = serialize_model_subclass(tracer.model)

    # 4. Build final payload
    payload = {
        "version": SERIALIZATION_VERSION,
        "source": source_metadata,
        "variables": variables,
        "remote_objects": remote_objects,
        "model_refs": model_refs,
    }
    if model_subclass:
        payload["model_subclass"] = model_subclass

    # 5. Return JSON bytes
    return json.dumps(payload).encode('utf-8')
```

The system knows it's done when:
1. All frame locals have been classified and processed
2. All external references have been resolved (no unresolved names)
3. All auto-discovered dependencies have been accumulated
4. The JSON payload is complete and valid

### Summary: The Discovery Flow

```
User code with model.trace(remote=True)
           │
           ▼
    ┌──────────────────┐
    │ serialize_source │  Entry point
    │     _based()     │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │   extract_all()  │  Walk frame locals
    └────────┬─────────┘
             │
    ┌────────┼────────┬────────────┬─────────────┐
    ▼        ▼        ▼            ▼             ▼
 Model   @remote   Lambda      Tensor      Primitive
  ref    object   function      data         JSON
    │        │        │            │             │
    │        ▼        │            │             │
    │  ┌───────────┐  │            │             │
    │  │ extract_  │  │            │             │
    │  │  remote_  │  │            │             │
    │  │  object() │  │            │             │
    │  └─────┬─────┘  │            │             │
    │        │        │            │             │
    │        ▼        │            │             │
    │  Instance?──────┼────────────┼─────────────┤
    │  Yes │ No       │            │             │
    │      ▼          │            │             │
    │  ┌───────────┐  │            │             │
    │  │serialize_ │  │            │             │
    │  │ instance_ │  │            │             │
    │  │  state()  │  │            │             │
    │  └─────┬─────┘  │            │             │
    │        │        │            │             │
    │        ▼        ▼            ▼             ▼
    │   ┌─────────────────────────────────────────┐
    │   │           serialize_value()             │
    │   │  (recursive, handles nested objects)    │
    │   └─────────────────────────────────────────┘
    │                      │
    └──────────────────────┼──────────────────────┘
                           ▼
                  ┌──────────────────┐
                  │  Build payload   │
                  │  JSON → bytes    │
                  └──────────────────┘
```

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

JSON doesn't distinguish between different Python types—a list is just `[...]`, a dict is just `{...}`. To reconstruct Python objects correctly, we use **marker keys** that indicate the type and carry the necessary data for reconstruction.

The marker constants are defined at the top of `serialization_source.py`. Here's a complete reference:

#### Tensor and Parameter Markers

```python
# TENSOR_MARKER = "__tensor__"
# For torch.Tensor and numpy.ndarray
{"__tensor__": {
    "data": "base64-encoded-bytes...",
    "dtype": "float32",
    "shape": [768, 512]
}}

# NN_PARAMETER_MARKER = "__nn_parameter__"
# For nn.Parameter (wraps tensor with requires_grad info)
{"__nn_parameter__": {
    "data": "base64...",
    "dtype": "float32",
    "shape": [768],
    "requires_grad": true
}}
```

Tensors are serialized as base64-encoded bytes with metadata about dtype and shape. On deserialization, the bytes are decoded and reshaped into the original tensor. Note: the full JSON payload is compressed at the transport layer, so per-tensor compression is not needed.

#### Collection Markers

```python
# DICT_MARKER = "__dict__"
# For dicts with non-string keys (JSON requires string keys)
{"__dict__": {
    "keys": [{"__tensor__": ...}, 42, "normal_key"],
    "values": ["value1", "value2", "value3"]
}}

# LIST_MARKER = "__list__"
# For lists containing complex types
{"__list__": [item1, item2, ...]}

# TUPLE_MARKER = "__tuple__"
# Distinguishes tuples from lists (JSON has only arrays)
{"__tuple__": [item1, item2, ...]}

# SET_MARKER = "__set__"
# For sets (no JSON equivalent)
{"__set__": [item1, item2, ...]}
```

Python has rich collection types; JSON has only arrays and string-keyed objects. These markers preserve the distinction so `(1, 2)` doesn't become `[1, 2]` after round-trip.

#### Reference Markers (Deduplication and Identity)

```python
# ID_MARKER = "__id__"
# Marks an object's identity for later reference
{"__id__": "140234567890", ...rest of object...}

# REF_MARKER = "__ref__"
# References a previously-serialized object by ID
{"__ref__": "140234567890"}
```

These work together to handle object identity and circular references. When an object is first serialized, it gets an `__id__`. If the same object appears again (same `id()` in Python), we emit a `__ref__` instead of re-serializing. On deserialization, we maintain a lookup table to reconnect references.

```python
# Example: circular reference
a = {"name": "a"}
b = {"name": "b", "other": a}
a["other"] = b  # circular!

# Serializes as:
{"__id__": "111", "name": "a", "other": {"__id__": "222", "name": "b", "other": {"__ref__": "111"}}}
```

#### Model Reference Marker

```python
# MODEL_REF_MARKER = "__model_ref__"
# Placeholder for the traced model (injected server-side)
{"__model_ref__": true}
```

The traced model exists on the server already—we don't serialize it. Instead, we emit this marker wherever the model is referenced. During deserialization, the server replaces these markers with the actual model instance.

#### Remote Object Markers

```python
# REMOTE_REF_MARKER = "__remote_ref__"
# CLASS_MARKER = "__class__"
# Instance of a user-defined class (auto-discovered or @remote decorated)
# Uses short class name (looked up in namespace populated by exec'ing source)
{"__class__": "MyAnalyzer", "__dict__": {"top_k": 10, "threshold": 0.5}, "__id__": "auto_12345"}

# REMOTE_TYPE_MARKER = "__remote_type__"
# The class itself (not an instance) as a value
{"__remote_type__": "MyAnalyzer"}
```

User-defined classes are serialized as source code (in the `remote_objects` section of the payload). The `__class__` marker identifies which class to instantiate, and `__dict__` contains the instance state. The `__id__` marker enables deduplication for shared/circular references.

Note: If two classes with the same name from different modules are discovered, a collision error is raised with a suggestion to rename one of the classes.

#### Callable and Type References

```python
# CALLABLE_REF_MARKER = "__callable_ref__"
# Reference to a function/method from a server-available module
{"__callable_ref__": {"module": "torch.nn.functional", "name": "softmax"}}

# TYPE_REF_MARKER = "__type_ref__"
# Reference to a type/class from a server-available module
{"__type_ref__": {"module": "torch", "name": "float32"}}
```

When code references functions or types from allowed modules (like `torch`, `numpy`), we don't serialize their source—we just record where to import them from. These markers encode that import path.

#### Special Case Markers

```python
# NN_MODULE_MARKER = "__nn_module__", DICT_MARKER = "__dict__"
# For nn.Module instances (serializes class path + __dict__)
{
    "__nn_module__": "torch.nn.Linear",
    "__dict__": {"weight": {"__tensor__": ...}, "bias": {"__tensor__": ...}, ...}
}

# ENUM_MARKER = "__enum__"
# For enum values
{"__enum__": {"class": "MyEnum", "name": "VALUE_A"}}

# ENUM_FALLBACK_MARKER = "__enum_fallback__"
# For enums that can't be reconstructed by name
{"__enum_fallback__": {"class": "MyEnum", "value": 42}}

# WEAKREF_MARKER = "__weakref__"
# Placeholder for weakref (can't serialize, will be None)
{"__weakref__": null}

# SERVER_PROVIDED_MARKER = "__server_provided__"
# Value that will be provided by the server environment
{"__server_provided__": "some_server_resource"}
```

These handle edge cases: `nn.Module` is reconstructed via `object.__new__()` with `__dict__` restoration (same as regular classes); enums are reconstructed by name; weakrefs can't be serialized (they're ephemeral by design); and some values are expected to exist server-side.

#### Marker Design Principles

1. **Unambiguous**: Each marker is a unique key that can't appear in normal data
2. **Self-describing**: The marker name indicates what type to reconstruct
3. **Minimal**: Only include data necessary for reconstruction
4. **Composable**: Markers can nest (e.g., a `__tuple__` containing `__tensor__` values)

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

Tensors are base64-encoded:

```python
def serialize_tensor(tensor) -> dict:
    # Convert to numpy, then to bytes
    arr = tensor.detach().cpu().numpy()
    data = arr.tobytes()

    return {
        "__tensor__": {
            "data": base64.b64encode(data).decode('ascii'),
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        }
    }
```

Note: The full JSON payload is compressed at the transport layer, so per-tensor compression is not needed.

### Instance State Serialization

For instances of user-defined classes, we serialize their `__dict__`:

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

### Forbidden Serialization

Some objects *could* technically be serialized, but *shouldn't* be—either because they represent OS resources that can't be transferred, or because they trigger massive warning cascades before eventually failing.

**Problem categories:**

1. **OS/System Resources**: `socket.socket`, `threading.Lock`, file handles—these have Python wrappers but represent system state that can't transfer between machines.

2. **Data Science Objects**: `pandas.DataFrame` generates 50+ warnings before failing on an internal `range` type. `matplotlib.Figure` fails on callback references.

3. **Test Framework Leakage**: `pytest` fixtures like `capsys` can accidentally leak into trace scope.

**Solution**: Early detection with actionable error messages:

```python
FORBIDDEN_MODULE_PREFIXES = frozenset({
    'socket', 'multiprocessing', '_pytest', 'pytest',
    'sqlalchemy', 'logging', ...
})

FORBIDDEN_CLASSES = {
    'pandas.core.frame.DataFrame': (
        "pandas.DataFrame cannot be serialized.\n"
        "Convert to tensor: torch.tensor(df.values)"
    ),
    'matplotlib.figure.Figure': (
        "matplotlib.Figure cannot be serialized.\n"
        "Save to bytes: fig.savefig(buf, format='png')"
    ),
}
```

The check happens BEFORE any auto-discovery attempt, so users get a clear error immediately instead of a cascade of warnings.

### Collections with Tensors

A common pattern accumulates results across traces:

```python
results = []
for prompt in prompts:
    with model.trace(prompt, remote='local'):
        result = model.layer.output.save()
    results.append(result)  # results now contains tensors
```

On subsequent traces, `results` contains tensors from previous iterations. Simple `is_json_serializable()` fails because tensors aren't JSON-serializable, but these lists ARE serializable via our tensor handling.

**Solution**: `_can_deep_serialize()` recursively checks if a value can be serialized:

```python
def _can_deep_serialize(value, seen=None):
    """Check if value (including nested collections) can be serialized."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if is_tensor(value):
        return True
    if is_remote_object(value):
        return True
    if isinstance(value, (list, tuple)):
        return all(_can_deep_serialize(item, seen) for item in value)
    # ... similar for dict, set
    return False
```

When `extract_all()` encounters a collection that isn't pure JSON but CAN be deep-serialized, it uses `serialize_value()` for recursive handling.

### The `remote_noop` Decorator

**Problem**: When code with `@remote` decorators is deserialized and exec'd on the server:
1. The decorator tries to extract source (fails—code was created via exec)
2. It tries to re-validate (unnecessary—already validated client-side)

**Solution**: Provide `remote_noop` in the deserialization namespace:

```python
def remote_noop(obj=None, *, version=None, library=None):
    """No-op @remote for deserialization context."""
    def apply_noop(obj):
        obj._remote_validated = True
        obj._remote_source = None  # Already transmitted
        return obj
    return apply_noop(obj) if obj else apply_noop
```

The deserialization namespace maps `'remote': remote_noop`, so transmitted code like `@remote class MyClass: ...` just marks the class as validated without source extraction.

### Class Deduplication and Instance Identity

When multiple instances of the same class are serialized:

```python
a = Counter(1)
b = Counter(2)
c = Counter(3)
```

We must:
1. Transmit the class source only once (efficiency)
2. Ensure all instances share the same class object after deserialization (`isinstance` must work)

**Serialization format** groups instances under their class:

```json
{
  "remote_objects": {
    "Counter": {
      "source": {"code": "class Counter:..."},
      "instances": {
        "12345678": {"var_name": "a", "state": {"value": 1}},
        "23456789": {"var_name": "b", "state": {"value": 2}},
        "34567890": {"var_name": "c", "state": {"value": 3}}
      }
    }
  }
}
```

**Deserialization** reconstructs all instances from the SAME class:

```python
# Class is exec'd once
exec_func(source_code, namespace)
cls = namespace['Counter']

# All instances created from same class
for instance_id, instance_data in obj_data['instances'].items():
    obj = object.__new__(cls)  # Same cls for all!
    obj.__dict__ = reconstruct_state(instance_data['state'], ...)
    namespace[instance_data['var_name']] = obj
```

This guarantees `type(a) is type(b)` and `isinstance(b, type(a))`.

---

## The `@remote` Decorator (Optional)

By default, nnsight **auto-discovers** user-defined classes and functions and serializes their source code automatically. The `@remote` decorator provides an **optional explicit annotation** when you want additional control or early validation.

### When Auto-Discovery is Sufficient

For most use cases, you don't need `@remote` at all:

```python
class MyAnalyzer:
    """This class will be auto-discovered and transported."""
    def __init__(self, top_k=10):
        self.top_k = top_k

    def analyze(self, hidden):
        return hidden.topk(self.top_k)

analyzer = MyAnalyzer(top_k=10)
with model.trace("Hello", remote=True):
    result = analyzer.analyze(model.layers[10].output[0])
```

The serialization system detects `MyAnalyzer`, extracts its source, and transmits it automatically.

### When to Use `@remote`

The `@remote` decorator is useful when you want:

1. **Import-time validation**: Problems are caught when the module loads, not during trace serialization
2. **Explicit documentation**: Makes it clear which code is designed for remote execution
3. **Dependency specification**: Explicitly declare library versions for reproducibility

```python
from nnsight import remote

@remote
class MyAnalyzer:
    """Explicitly marked for remote transport."""
    def __init__(self, top_k=10):
        self.top_k = top_k

    def analyze(self, hidden):
        return hidden.topk(self.top_k)
```

### How It Works Under the Hood

When Python imports a module containing `@remote`, the decorator:

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

### Import-Time Validation

Early validation is a key benefit of explicit `@remote`. Instead of:

```
User writes code → Serializes → Sends to server → Server fails → User confused
```

With `@remote`:

```
User writes code → Import fails with clear error → User fixes immediately
```

The decorator raises exceptions for things like:
- Relative imports (`from .utils import helper`)
- Dynamic code (`exec()`, `eval()`)
- Closures over non-serializable values

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

### Auto-Discovery vs @remote Annotation

The `@remote` decorator provides explicit annotation for remote serialization. But what about existing third-party libraries that don't use `@remote`?

**Current default behavior:** Auto-discovery is **enabled by default** for all classes and functions with available source code. This means third-party libraries (like nnterp's `StandardizedTransformer`) work out-of-the-box without requiring `@remote` annotations.

```python
# No @remote needed - auto-discovered at serialization time
class MyHelper:
    def __init__(self, scale):
        self.scale = scale
    def apply(self, x):
        return x * self.scale

helper = MyHelper(2.0)

with model.trace("test", remote=True):
    result = helper.apply(hidden)  # Auto-discovered and serialized
```

**Strict mode:** For users who prefer explicit control, pass `strict_remote=True` to require `@remote` annotations:

```python
with model.trace("test", remote=True, strict_remote=True):
    result = helper.apply(hidden)  # Error: MyHelper not @remote decorated
```

**Design trade-offs:**

| Mode | Behavior | Use Case |
|------|----------|----------|
| **Default (strict_remote=False)** | Auto-discover any class/function with source | Easy integration with third-party libraries; less boilerplate |
| **Strict (strict_remote=True)** | Require explicit `@remote` annotations | Early validation at import time; clear serialization contract |

**When to use @remote:**
- When you want import-time validation (catch errors early)
- When you want to clearly document which code runs remotely
- When you need guaranteed serialization behavior

**When to rely on auto-discovery:**
- When using third-party libraries you don't control
- For quick prototyping without annotation overhead
- When working with existing codebases

**Upload size warnings:** Large payloads can cause slow transmission. The default threshold is 10 MB. Adjust with `max_upload_mb`:

```python
with model.trace("test", remote=True, max_upload_mb=5.0):
    # Warning if upload payload exceeds 5 MB
    ...

with model.trace("test", remote=True, max_upload_mb=0):
    # Disable upload size warnings
    ...
```

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

### Local Round-Trip Testing

You can exercise the entire serialization and deserialization pipeline locally without sending anything to a remote server. This is how the test suite works:

```python
import json
from nnsight import remote
from nnsight.intervention.serialization_source import (
    serialize_instance_state,
    reconstruct_state,
    deserialize_source_based,
)

# 1. Define a @remote class
@remote
class MyAnalyzer:
    def __init__(self, top_k=10):
        self.top_k = top_k

    def analyze(self, hidden):
        return hidden.topk(self.top_k)

# 2. Create an instance
analyzer = MyAnalyzer(top_k=5)

# 3. Serialize the instance state
state = serialize_instance_state(analyzer)
print("Serialized state:", json.dumps(state, indent=2))

# 4. Reconstruct the instance
restored = object.__new__(MyAnalyzer)
restored.__dict__ = reconstruct_state(state, {}, None, {})

# 5. Verify it works
print(f"Restored top_k: {restored.top_k}")  # Should print 5

# For full payload round-trip, construct the JSON manually:
payload = json.dumps({
    "version": "2.2",
    "source": {"code": "", "file": "test.py", "line": 1},
    "variables": {"threshold": 0.5},
    "remote_objects": {
        "MyAnalyzer": {
            "type": "class",
            "source": {"code": MyAnalyzer._remote_source, "file": "test.py", "line": 1},
            "module_refs": MyAnalyzer._remote_module_refs,
            "closure_vars": {},
            "instances": {},
        }
    },
    "model_refs": [],
}).encode('utf-8')

# Mock model for deserialization
class MockModel:
    pass

# Deserialize - this reconstructs the full namespace
namespace = deserialize_source_based(payload, MockModel())
print("Reconstructed namespace keys:", list(namespace.keys()))
```

This approach lets you:
- Test serialization logic without network latency
- Debug payload structure by inspecting the JSON
- Verify round-trip correctness for new types
- Run fast unit tests in CI

### Proposed: `remote='local'` Mode

Currently, testing the full serialization pipeline requires manual payload construction. A more convenient approach would be a `remote='local'` mode that exercises the entire serialize/deserialize round-trip locally:

```python
# Proposed API (not yet implemented):
with model.trace("Hello world", remote='local'):
    hidden = model.layers[10].output[0]
    result = hidden.topk(10).save()
```

This would:
1. Serialize using `serialize_source_based()` (identical to `remote=True`)
2. Create a fresh namespace (simulating the server environment)
3. Deserialize using `deserialize_source_based()` into that namespace
4. Execute the trace in the isolated namespace
5. Return results normally

The key benefit: it catches bugs where code accidentally depends on closure variables or imports that aren't properly captured during serialization. With `remote=True`, these bugs only surface on the actual server; with `remote='local'`, you'd catch them immediately during development.

Implementation would involve a `LocalSimulationBackend` that wraps the serialization/deserialization functions instead of making network calls.

---

## Summary

The source serialization system enables Python-version-independent remote execution by:

1. **Auto-discovering user code** and extracting source instead of bytecode
2. **Validating early** (at import time for `@remote`, at serialization for auto-discovered)
3. **Capturing context** (module refs, closures, constants)
4. **Preserving identity** through ID-based deduplication
5. **Maintaining debuggability** with original file/line info

The key mental model: user-defined code in traces is a **query generator**, not runtime code. We're shipping the query language (source) to the server, not compiled instructions (bytecode).

For detailed specifications and future roadmap, see `nnsight-source-serialization-design.md`.
