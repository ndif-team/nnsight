# Disambiguation Design: Handling Identity and Name Collisions

This document describes how nnsight's source serialization system handles object identity, shared state, and name collisions when serializing Python code for remote execution.

---

## The Problem

When serializing Python code for remote execution, we must preserve not just the code but also the *relationships* between objects. Python's runtime identity model creates several challenges.

### Challenge 1: Name Collisions

**Example:**
```python
# analytics/helpers.py
class Helper:
    def process(self, x):
        return x * 2

# visualization/helpers.py
class Helper:
    def process(self, x):
        return self.format(x)
```

Both classes are named `Helper`, but they are completely different. A user might import both:

```python
from analytics.helpers import Helper as AnalyticsHelper
from visualization.helpers import Helper as VizHelper

with model.trace(input, remote=True):
    a = AnalyticsHelper()
    v = VizHelper()
    # Both must work correctly after serialization
```

**Why it matters:** If we only use the short name "Helper" as an identifier, the two classes would collide. One would overwrite the other, causing incorrect behavior or crashes.

### Challenge 2: Shared Module-Level State

**Example:**
```python
# cache.py
CACHE = {}

def get(key):
    return CACHE.get(key)

def put(key, value):
    CACHE[key] = value
```

In Python, `get` and `put` share the exact same `CACHE` dictionary. This is fundamental to how the module works - `put` stores values that `get` retrieves.

**The invariant we must preserve:**
```python
# Before serialization:
put("x", 42)
assert get("x") == 42  # Works because they share CACHE

# After round-trip serialization:
put("x", 42)
assert get("x") == 42  # Must still work!
```

**Why it matters:** If deserialization gives `get` and `put` separate copies of `CACHE`, the module is broken. `put` would store to one dict while `get` reads from another.

### Challenge 3: Cross-Reference Identity

**Example:**
```python
# config.py
DEFAULTS = {"timeout": 30, "retries": 3}

class Client:
    def __init__(self):
        self.config = DEFAULTS  # Points to same dict

    def update(self, key, value):
        self.config[key] = value  # Modifies DEFAULTS too!
```

Here, `client.config` and `DEFAULTS` are the same object. Modifying one modifies the other.

**The invariant we must preserve:**
```python
client = Client()
assert client.config is DEFAULTS  # Same object

# After round-trip:
assert client.config is DEFAULTS  # Must still be same object!
```

**Why it matters:** Code may rely on this identity for correctness. For example, modifications through `client.config` might intentionally update the global defaults.

### Challenge 4: Closure State

**Example:**
```python
def make_counter(start=0):
    count = start
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

counter_a = make_counter(0)   # Has its own 'count'
counter_b = make_counter(100) # Has different 'count'

counter_a()  # Returns 1
counter_a()  # Returns 2
counter_b()  # Returns 101 (independent!)
```

Each call to `make_counter` creates a new closure with its own `count` variable. The two counters are independent.

**The invariant we must preserve:**
```python
# counter_a and counter_b must remain independent
# counter_a's count must not affect counter_b's count
```

**Why it matters:** Closures are a common pattern for encapsulating state. Serialization must not accidentally merge independent closures or share their captured state.

---

## Design Principles

### Principle 1: Use Qualified Names for Source Identity

**What:** Classes and functions are identified by their fully qualified name: `module.submodule.ClassName`.

**Why:**
- Qualified names are unique across the codebase
- They are human-readable and debuggable
- They match Python's own module system semantics

**Example:**
```python
# Instead of:
{"Helper": {"source": "class Helper: ..."}}

# We use:
{"analytics.helpers.Helper": {"source": "class Helper: ..."}}
{"visualization.helpers.Helper": {"source": "class Helper: ..."}}
```

**Benefit:** No collisions. Two `Helper` classes coexist peacefully.

### Principle 2: Use `id()` for Runtime Identity

**What:** During serialization, we track object identity using Python's `id()` function to detect sharing.

**Why:**
- `id()` returns a unique identifier for each object in memory
- If `id(a) == id(b)`, then `a is b` (same object)
- This lets us detect when multiple references point to the same object

**Example:**
```python
CACHE = {}
def get(key): return CACHE.get(key)
def put(key, value): CACHE[key] = value

# During serialization:
# id(get.__globals__["CACHE"]) == id(put.__globals__["CACHE"])
# Therefore: they share CACHE, serialize it once, reference it twice
```

**Benefit:** Shared objects are serialized once and reconstructed once, preserving identity.

### Principle 3: Group by Shared Globals

**What:** Functions that share `__globals__` (i.e., from the same module) are grouped together and given the same namespace during deserialization.

**Why:**
- In Python, functions defined in the same module share a `__globals__` dict
- `func1.__globals__ is func2.__globals__` is true for module-mates
- To preserve sharing, we must give them the same namespace

**How we detect it:**
```python
# If id(get.__globals__) == id(put.__globals__):
#   They are in the same namespace group
#   They will share a namespace after deserialization
```

**Benefit:** Module-level state sharing is automatically preserved.

### Principle 4: Fail Fast on Unsupported Patterns

**What:** Detect problematic patterns during serialization and raise clear errors immediately.

**Why:**
- Late failures (during deserialization or execution) are confusing
- Clear error messages help users fix issues quickly
- Some patterns are inherently difficult to serialize correctly

**Benefit:** Users get actionable errors, not mysterious runtime failures.

---

## The Hybrid Solution

We call our approach "hybrid" because it combines:
- **Qualified names** as the primary keys in the serialized format (human-readable)
- **`id()` tracking** as the mechanism to detect sharing (correct semantics)

### How It Works

**Step 1: Collect all objects to serialize**

Starting from frame locals, we recursively discover:
- Classes and functions (identified by qualified name)
- Instances (with references to their classes)
- Values (module-level variables, captured closure variables)

**Step 2: Track identity with `id()`**

For every object we encounter:
- Record its `id()` in a registry
- If we see the same `id()` again, it's a shared reference
- Assign each unique object a serial identifier (e.g., `sv_0`, `sv_1`)

**Step 3: Detect namespace groups**

For every function:
- Record `id(func.__globals__)`
- Functions with the same `__globals__` id are grouped together
- Each group will share a namespace during deserialization

**Step 4: Build the serialization format**

```
sources:          qualified_name -> source code + metadata
shared_values:    serial_id -> serialized value
namespace_groups: group_id -> {members, bindings}
instances:        instance_id -> {class, __dict__}
variables:        name -> value or reference
```

**Step 5: Deserialize in correct order**

1. Deserialize shared values first (so references can resolve)
2. Create namespace dicts for each group (with shared values bound)
3. Execute source code into appropriate namespaces
4. Reconstruct instances (with references resolved)
5. Build final variables namespace

---

## Invariants Guaranteed

### Invariant 1: Qualified Name Uniqueness

> Two different classes/functions always have different qualified names.

This follows from Python's module system: `module.Class` uniquely identifies one class.

### Invariant 2: Identity Preservation

> If `a is b` before serialization, then `a' is b'` after deserialization (where `a'`, `b'` are the deserialized versions).

This is guaranteed by `id()` tracking: shared objects get one serial id, and that id resolves to one deserialized object.

### Invariant 3: Namespace Sharing

> If `func1.__globals__ is func2.__globals__` before serialization, they share `__globals__` after deserialization.

This is guaranteed by namespace groups: functions with the same `__globals__` id are executed into the same namespace dict.

### Invariant 4: Class Instance Relationship

> If `isinstance(obj, MyClass)` before serialization, then `isinstance(obj', MyClass')` after deserialization.

This is guaranteed by qualified name lookup: instances reference their class by qualified name, which resolves to the deserialized class.

---

## Capabilities

### Supported: Same-Named Classes from Different Modules

```python
from package_a import Helper
from package_b import Helper as OtherHelper

h1 = Helper()
h2 = OtherHelper()
```

Both work correctly. They have different qualified names:
- `package_a.Helper`
- `package_b.Helper`

### Supported: Shared Module-Level State

```python
# mymodule.py
STATE = {"count": 0}

def increment():
    STATE["count"] += 1

def get_count():
    return STATE["count"]
```

After round-trip:
```python
increment()
increment()
assert get_count() == 2  # Works! They share STATE.
```

### Supported: Cross-Reference Identity

```python
SHARED_LIST = [1, 2, 3]

class Container:
    def __init__(self):
        self.data = SHARED_LIST

c = Container()
```

After round-trip:
```python
assert c.data is SHARED_LIST  # Still the same object!
SHARED_LIST.append(4)
assert c.data[-1] == 4  # Mutation visible through both references
```

### Supported: Circular References

```python
a = {"name": "a"}
b = {"name": "b", "ref": a}
a["ref"] = b  # Circular!
```

After round-trip:
```python
assert a["ref"] is b
assert b["ref"] is a
assert a["ref"]["ref"] is a  # Cycle intact
```

### Supported: Multiple Instances of Same Class

```python
class Counter:
    def __init__(self, start):
        self.value = start

c1 = Counter(10)
c2 = Counter(20)
```

After round-trip:
```python
assert type(c1) is type(c2)  # Same class
assert isinstance(c2, type(c1))  # isinstance works
assert c1.value == 10  # Independent state
assert c2.value == 20
```

---

## Limitations

### Limitation 1: Closures with Mutable `nonlocal` State

**What's not supported:**
```python
def make_counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

counter = make_counter()
```

**Why:** The `nonlocal` keyword requires `count` to exist in an enclosing function's scope, not in globals. Reconstructing this requires either:
- Generating wrapper code to recreate the closure structure
- Using `types.FunctionType` with custom closure cells

Both approaches are complex and error-prone.

**Error message:**
```
SourceSerializationError: Function 'increment' captures mutable closure
variable 'count' via nonlocal. This cannot be serialized for remote execution.

Workaround: Refactor to use a class with instance state:
    class Counter:
        def __init__(self):
            self.count = 0
        def increment(self):
            self.count += 1
            return self.count
```

**Justification:** This pattern is rare in trace helper code. Most helpers use classes or module-level functions. The workaround is straightforward.

### Limitation 2: Mutable Class Attributes

**What happens:**
```python
class Registry:
    instances = []  # Class-level mutable attribute

    def __init__(self, name):
        self.name = name
        Registry.instances.append(self)
```

Class attributes are reset to their source-code values. If `Registry.instances` had accumulated items, they are lost.

**Warning message:**
```
Warning: Class 'Registry' has mutable class attribute 'instances' (type: list).
Class attributes are reset to source values during remote execution.
Current value will not be preserved.
```

**Why not preserve them:** Preserving modified class attributes requires:
- Detecting which attributes changed from source defaults
- Serializing current values separately
- Restoring after class creation

This adds significant complexity for a rare use case.

**Workaround:** Use module-level state:
```python
_registry_instances = []

class Registry:
    def __init__(self, name):
        self.name = name
        _registry_instances.append(self)
```

Module-level state is properly shared through namespace groups.

### Limitation 3: Immutable Closure Captures (Partial Support)

**What happens:**
```python
def make_multiplier(factor):
    def multiply(x):
        return x * factor
    return multiply

double = make_multiplier(2)
```

Immutable captured values (like `factor = 2`) are serialized and made available as globals. The function works, but it's not a true closure.

**Caveat:** If code somehow relied on closure cell identity (extremely rare), behavior may differ.

**Why partial:** True closure reconstruction is complex. For immutable values that are only read, treating them as globals is functionally equivalent and much simpler.

---

## Benefits of This Design

### Benefit 1: Correct Semantics

The primary goal - shared references remain shared, independent references remain independent - is achieved through `id()` tracking.

### Benefit 2: Human-Readable Format

Using qualified names as keys means the serialized format is inspectable:
```json
{
  "sources": {
    "myproject.helpers.DataProcessor": { ... },
    "myproject.analyzers.ResultCollector": { ... }
  }
}
```

Debugging is straightforward - you can read the qualified names and understand what's being serialized.

### Benefit 3: No Collisions

Qualified names eliminate the possibility of name collisions. Two `Helper` classes from different modules simply have different keys.

### Benefit 4: Minimal Serialization

We only serialize what's actually referenced. If a module has 100 functions but only 2 are used in the trace, only those 2 are serialized.

### Benefit 5: Clear Error Messages

Unsupported patterns are detected early with actionable error messages that explain the limitation and suggest workarounds.

---

## Why Not Other Approaches?

### Why not use short names only?

**Problem:** Collisions.

```python
from pkg_a import Helper  # pkg_a.Helper
from pkg_b import Helper as H  # pkg_b.Helper

# With short names only, both would be "Helper"
```

### Why not serialize entire modules?

**Problem:** Bloat and unnecessary dependencies.

A module might have 50 functions, complex imports, and initialization code. Serializing all of it when you only need 2 functions wastes bandwidth and may fail on non-serializable parts.

### Why not use pickle for sharing detection?

**Problem:** Pickle captures bytecode, which is version-specific.

Our goal is version-independent serialization. Source code works across Python versions; bytecode doesn't.

### Why not reconstruct exact module structure?

**Problem:** Complexity and edge cases.

Python's module system has many features: `__init__.py`, relative imports, namespace packages, etc. Reconstructing this exactly is complex and often unnecessary. Namespace groups give us the sharing semantics we need without full module reconstruction.

---

## Examples in Practice

### Example 1: Analytics Pipeline

```python
# analytics/core.py
METRICS = {"calls": 0, "errors": 0}

def track_call():
    METRICS["calls"] += 1

def track_error():
    METRICS["errors"] += 1

def get_metrics():
    return METRICS.copy()
```

**Serialization:**
- `analytics.core.track_call`, `analytics.core.track_error`, `analytics.core.get_metrics` all reference `METRICS`
- They share `__globals__`, so they're in the same namespace group
- `METRICS` is serialized once as a shared value
- All three functions bind `METRICS` to that shared value

**After round-trip:**
```python
track_call()
track_call()
track_error()
print(get_metrics())  # {"calls": 2, "errors": 1}
```

### Example 2: Configuration with Defaults

```python
# config.py
DEFAULTS = {"timeout": 30}

class Client:
    def __init__(self, **overrides):
        self.config = {**DEFAULTS, **overrides}

    def get_timeout(self):
        return self.config["timeout"]
```

**Serialization:**
- `config.Client` is serialized with its source
- `DEFAULTS` is serialized as a shared value
- The `Client` class's namespace binds `DEFAULTS`

**After round-trip:**
```python
c1 = Client()
c2 = Client(timeout=60)
assert c1.get_timeout() == 30  # Got default
assert c2.get_timeout() == 60  # Got override
```

### Example 3: Cross-Module References

```python
# types.py
class DataPoint:
    def __init__(self, x, y):
        self.x, self.y = x, y

# analysis.py
from .types import DataPoint

def create_origin():
    return DataPoint(0, 0)
```

**Serialization:**
- `mypackage.types.DataPoint` serialized
- `mypackage.analysis.create_origin` serialized
- `create_origin`'s namespace group binds `DataPoint` to the class

**After round-trip:**
```python
origin = create_origin()
assert isinstance(origin, DataPoint)  # Works!
```

---

## Implementation Notes

### Qualified Names in Extract Functions

The following functions use qualified names `f"{cls.__module__}.{cls.__name__}"` as dictionary keys to prevent collisions:

- `extract_remote_object()` - Extracts `@remote` decorated objects
- `extract_auto_discovered_object()` - Extracts auto-discovered class instances
- `extract_auto_discovered_type()` - Extracts auto-discovered class types
- `extract_auto_discovered_function()` - Extracts auto-discovered functions

The deserialization code in `deserialize_source_based()` handles both qualified and short names for backwards compatibility, extracting the short name for namespace lookup:

```python
if '.' in obj_qualified_name:
    obj_short_name = obj_qualified_name.split('.')[-1]
else:
    obj_short_name = obj_qualified_name
```

### Detecting Nonlocal Closures

The `_has_nonlocal_closure()` function uses AST analysis to detect functions with `nonlocal` statements. This is used by `_auto_discover_function()` to reject such functions with a clear error message suggesting the class-based workaround.

### Mutable Class Attribute Warnings

The `_check_mutable_class_attributes()` function inspects class `__dict__` for common mutable types (list, dict, set) and issues warnings when found. This helps users understand that class-level state won't be preserved.

---

## Related Documents

- [Source Serialization Tutorial](source-serialization-tutorial.md) - User-facing guide
- [Source Serialization Design](nnsight-source-serialization-design.md) - Overall architecture
- [Forbidden Serialization Design](forbidden-serialization-design.md) - Early rejection of problematic objects
