# Remotable Python: What Code Can Run on NDIF

This guide explains which Python code patterns can be sent to NDIF for remote execution, which patterns are forbidden, and how to work around limitations.

---

## Quick Reference

| Pattern | Status | Notes |
|---------|--------|-------|
| Classes with `__dict__` | ✅ Supported | Instance state is serialized |
| Functions with source code | ✅ Supported | Source is transmitted |
| Lambdas | ✅ Supported | Extracted and transmitted |
| Closures (immutable captures) | ✅ Supported | Captured values become globals |
| Tensors (torch/numpy) | ✅ Supported | Base64 encoded |
| JSON-serializable values | ✅ Supported | Primitives, lists, dicts |
| Module-level shared state | ✅ Supported | Identity preserved via `id()` |
| Circular references | ✅ Supported | Handled via reference markers |
| `__main__` definitions | ✅ Supported | Qualified as `__main__.ClassName` |
| Server-available module types | ✅ Supported | No source needed (see list below) |
| Subclasses of allowed bases | ✅ Supported | e.g., `nn.Module` subclasses |
| Closures with `nonlocal` | ❌ Forbidden | Use class instead |
| Mutable class attributes | ⚠️ Warning | Reset to source values |
| pandas DataFrames | ❌ Forbidden | Convert to tensor |
| matplotlib Figures | ❌ Forbidden | Save as bytes |
| File handles, sockets | ❌ Forbidden | Cannot transfer OS resources |
| Database connections | ❌ Forbidden | Cannot transfer sessions |

---

## What's Pre-Installed on NDIF Servers

The following modules and classes are assumed to exist on NDIF servers. Objects from these don't need source transmission—they're referenced by name and imported server-side.

## Server-Available Modules

### Standard Library (Safe Subset)

```python
# Core builtins and typing
'builtins', 'abc', 'typing', 'types'

# Collections and functional
'collections', 'functools', 'itertools', 'operator'

# Data structures and serialization
'enum', 'dataclasses', 'copy', 'pickle', 'json'

# Math and numbers
'math', 'numbers', 'random'

# String processing
're', 'string', 'textwrap'

# Utilities
'warnings', 'contextlib', 'weakref', 'inspect'

# I/O (path manipulation only - no actual file operations)
'io', 'os', 'sys', 'pathlib'
```

### Machine Learning Libraries

```python
# Core ML
'torch', 'numpy', 'scipy', 'sklearn'

# Hugging Face ecosystem
'transformers', 'huggingface_hub', 'tokenizers', 'safetensors', 'accelerate'

# Other common libraries
'einops'
```

**Example:** Using `torch.nn.Linear` doesn't require source transmission:

```python
layer = torch.nn.Linear(768, 768)

with model.trace("Hello", remote=True):
    h = model.layers[10].output[0]
    result = layer(h)  # ✅ Works - torch.nn.Linear is server-available
```

---

## Allowed Base Classes

These base classes are trusted for subclassing. Your subclass source is transmitted, but the base class is available on the server:

```python
# torch.nn module
'torch.nn.Module'

# torch.utils.data module
'torch.utils.data.Dataset'
'torch.utils.data.IterableDataset'
```

**Example:** Custom `nn.Module` subclasses work:

```python
import torch.nn as nn

class MyLayer(nn.Module):  # ✅ Allowed base class
    def __init__(self, size):
        super().__init__()
        self.linear = nn.Linear(size, size)
        self.scale = nn.Parameter(torch.ones(size))

    def forward(self, x):
        return self.linear(x) * self.scale

layer = MyLayer(768)

with model.trace("Hello", remote=True):
    result = layer(model.layers[10].output[0])  # ✅ Works
```

**Example:** Custom Dataset subclasses work:

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):  # ✅ Allowed base class
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

---

## The Core Idea

When you write code inside a `model.trace(remote=True)` block, nnsight needs to send that code to a remote server. Unlike traditional pickle, we send **source code**, not bytecode. This means:

1. **Python version independence**: Code written on Python 3.10 runs on Python 3.12
2. **Third-party libraries work**: Your helper classes are transmitted, not just referenced
3. **Clear errors**: Problems are caught early with actionable messages

The tradeoff is that some Python patterns cannot be accurately reconstructed from source code alone.

---

## What IS Remotable

### Classes and Their Instances

Any class with available source code can be remoted:

```python
class MyAnalyzer:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def analyze(self, hidden):
        return hidden > self.threshold

analyzer = MyAnalyzer(threshold=0.7)

with model.trace("Hello", remote=True):
    result = analyzer.analyze(model.layers[10].output[0])
```

**How it works:**
- The class source code is extracted via `inspect.getsource()`
- Instance state (`analyzer.__dict__`) is serialized as JSON
- On the server, the class is recreated via `exec()` and instances are reconstructed

**Classes from `__main__`** (defined in scripts or REPLs) work fine. They're identified as `__main__.MyAnalyzer` internally.

### Functions

Standalone functions work the same way:

```python
def compute_similarity(a, b):
    return torch.cosine_similarity(a, b, dim=-1)

with model.trace("Hello", remote=True):
    h1 = model.layers[5].output[0]
    h2 = model.layers[10].output[0]
    sim = compute_similarity(h1, h2)
```

### Lambdas

Lambdas are extracted and transmitted:

```python
normalize = lambda x: x / x.norm(dim=-1, keepdim=True)

with model.trace("Hello", remote=True):
    h = model.layers[10].output[0]
    h_norm = normalize(h)
```

### Closures with Immutable Captures

Closures that capture **immutable** values work correctly:

```python
def make_threshold_checker(threshold):
    def check(x):
        return x > threshold  # 'threshold' is captured
    return check

is_high = make_threshold_checker(0.9)

with model.trace("Hello", remote=True):
    result = is_high(model.layers[10].output[0].mean())
```

**How it works:** The captured value (`threshold = 0.9`) is serialized and made available as a global variable. The function works correctly, though technically it's not a true closure anymore.

### Tensors

PyTorch tensors and NumPy arrays are fully supported:

```python
weights = torch.randn(768, 768)
bias = torch.zeros(768)

with model.trace("Hello", remote=True):
    h = model.layers[10].output[0]
    result = h @ weights + bias
```

**Supported tensor types:**
- Regular dense tensors
- Sparse COO tensors (sparsity preserved)
- Quantized tensors (quantization parameters preserved)
- bfloat16 tensors (bit-cast to int16 for transport)

### Shared References

When multiple objects reference the same data, that sharing is preserved:

```python
shared_config = {"scale": 2.0, "offset": 0.5}

class Processor:
    def __init__(self):
        self.config = shared_config  # Points to same dict

p1 = Processor()
p2 = Processor()

with model.trace("Hello", remote=True):
    # p1.config and p2.config still point to the same dict
    p1.config["scale"] = 3.0  # Would affect p2.config too
```

**How it works:** During serialization, we track object identity using `id()`. Objects seen multiple times are serialized once and referenced thereafter.

### Module-Level Shared State

Functions from the same module correctly share module-level state:

```python
# In mymodule.py:
CACHE = {}

def get(key):
    return CACHE.get(key)

def put(key, value):
    CACHE[key] = value
```

```python
from mymodule import get, put

with model.trace("Hello", remote=True):
    put("hidden", model.layers[10].output[0])
    h = get("hidden")  # Works! Same CACHE object
```

**How it works:** Functions with the same `__globals__` are grouped together and executed into a shared namespace.

### Circular References

Objects that reference each other work correctly:

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

a = Node(1)
b = Node(2)
a.next = b
b.next = a  # Circular!

with model.trace("Hello", remote=True):
    # Both nodes are correctly reconstructed with the cycle intact
    assert a.next.next is a
```

### Multiple Instances of the Same Class

When you have multiple instances, they share the same class:

```python
class Counter:
    def __init__(self, start):
        self.value = start

c1 = Counter(10)
c2 = Counter(20)

with model.trace("Hello", remote=True):
    # Both are instances of the SAME Counter class
    assert type(c1) is type(c2)
    assert isinstance(c2, type(c1))
```

---

## What is NOT Remotable

### Closures with `nonlocal` (Mutable Closure State)

**Forbidden.** Functions that use `nonlocal` to mutate captured variables cannot be serialized:

```python
def make_counter():
    count = 0
    def increment():
        nonlocal count  # ← This is the problem
        count += 1
        return count
    return increment

counter = make_counter()

with model.trace("Hello", remote=True):
    x = counter()  # ❌ SourceSerializationError
```

**Why it fails:** The `nonlocal` keyword requires `count` to exist in an enclosing function's scope. We can't reconstruct this structure from source code alone—we'd need to generate wrapper functions or manipulate closure cells, which is fragile.

**Workaround:** Use a class instead:

```python
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
        return self.count

counter = Counter()

with model.trace("Hello", remote=True):
    x = counter.increment()  # ✅ Works
```

### Mutable Class Attributes

**Warning issued.** Class-level mutable state is reset to source values:

```python
class Registry:
    instances = []  # Class-level mutable attribute

    def __init__(self, name):
        self.name = name
        Registry.instances.append(self)

r1 = Registry("first")
r2 = Registry("second")
# Registry.instances is now ["first", "second"]

with model.trace("Hello", remote=True):
    # WARNING: Registry.instances is reset to []
    # The accumulated instances are lost!
```

**Why it happens:** We transmit the class source code, which defines `instances = []`. We don't track or transmit modifications to class attributes.

**Workaround:** Use module-level state:

```python
_registry_instances = []  # Module-level, not class-level

class Registry:
    def __init__(self, name):
        self.name = name
        _registry_instances.append(self)
```

Module-level state is properly serialized and shared.

### pandas DataFrames and Series

**Forbidden.** DataFrames have complex internal state that cannot be cleanly serialized:

```python
import pandas as pd

df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

with model.trace("Hello", remote=True):
    # ❌ SourceSerializationError
```

**Workaround:** Convert to tensor before the trace:

```python
import pandas as pd
import torch

df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
tensor_data = torch.tensor(df.values)  # Convert before trace

with model.trace("Hello", remote=True):
    # ✅ Use tensor_data instead
```

### matplotlib Figures and Axes

**Forbidden.** Figures contain rendering state and callbacks:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3])

with model.trace("Hello", remote=True):
    # ❌ SourceSerializationError
```

**Workaround:** Save as bytes if you need image data:

```python
import io
buf = io.BytesIO()
fig.savefig(buf, format='png')
image_bytes = buf.getvalue()

# Now image_bytes can be used in the trace
```

### PIL Images

**Forbidden.** Convert to tensor:

```python
from PIL import Image
from torchvision import transforms

img = Image.open("photo.jpg")
tensor = transforms.ToTensor()(img)  # Convert before trace

with model.trace("Hello", remote=True):
    # ✅ Use tensor instead of img
```

### File Handles, Sockets, Database Connections

**Forbidden.** These represent OS or network resources that exist on your machine, not the server:

```python
# All of these are forbidden:
f = open("data.txt")           # File handle
sock = socket.socket()         # Network socket
conn = sqlite3.connect("db")   # Database connection
lock = threading.Lock()        # Thread synchronization
```

**Why:** These objects wrap system resources that cannot be transferred. A file handle on your machine is meaningless on a remote server.

**Workaround:** Read data before the trace:

```python
# Instead of passing the file handle:
with open("data.txt") as f:
    data = f.read()  # Read the content

with model.trace("Hello", remote=True):
    # ✅ Use 'data' (a string) instead of 'f' (a file handle)
```

### Objects Without Source Code

**Forbidden.** Classes defined in C extensions or with unavailable source:

```python
# Built-in types are fine (handled specially)
# But C extension classes without Python source fail:

from some_c_extension import CythonClass
obj = CythonClass()

with model.trace("Hello", remote=True):
    # ❌ Cannot get source for CythonClass
```

**Note:** Standard library and torch/numpy types are handled specially—they're available on the server and don't need source transmission.

### Test Framework Objects

**Forbidden.** pytest fixtures and similar objects are blocked:

```python
def test_something(capsys):  # pytest fixture
    with model.trace("Hello", remote=True):
        # ❌ capsys cannot be serialized
```

**Why:** These are internal test framework objects that should never leak into production code.

---

## Edge Cases and Nuances

### `__main__` Definitions

Classes and functions defined in `__main__` (your script or REPL) work fine:

```python
# In your script:
class MyHelper:
    pass

# Internally identified as "__main__.MyHelper"
# Source is extracted and transmitted normally
```

### Same Name, Different Modules

Two classes with the same short name but different modules work correctly:

```python
from package_a import Helper  # package_a.Helper
from package_b import Helper as OtherHelper  # package_b.Helper

h1 = Helper()
h2 = OtherHelper()

with model.trace("Hello", remote=True):
    # Both work correctly—no collision
    # They're identified by full qualified names
```

### Nested Classes

Classes defined inside other classes work:

```python
class Outer:
    class Inner:
        def __init__(self, x):
            self.x = x

    def make_inner(self, x):
        return self.Inner(x)
```

### Decorated Classes/Functions

Decorators are preserved in the source:

```python
from dataclasses import dataclass

@dataclass
class Config:
    threshold: float = 0.5
    max_items: int = 100

config = Config(threshold=0.7)

with model.trace("Hello", remote=True):
    # Works—the @dataclass decorator is part of the source
```

### Enum Classes

Enum values are serialized by class and member name:

```python
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

color = Color.RED

with model.trace("Hello", remote=True):
    # color is serialized as {"__enum__": True, "class": "Color", "member": "RED"}
```

---

## How to Debug Serialization Issues

### Check What's Being Serialized

Use `remote='local'` to test serialization without actually sending to a server:

```python
with model.trace("Hello", remote='local'):
    # Exercises full serialize/deserialize cycle locally
    # Errors appear immediately, not on the server
```

### Inspect Payload Size

Large payloads slow down execution. Default warning threshold is 10MB:

```python
with model.trace("Hello", remote=True, max_upload_mb=5.0):
    # Warns if payload exceeds 5MB
```

### Read Error Messages

Errors include:
- The variable name that failed
- The type that couldn't be serialized
- Suggested workarounds

```
SourceSerializationError: Cannot serialize 'df' for remote execution:

pandas.DataFrame cannot be serialized for remote execution.
DataFrames have complex internal state that cannot be transferred.

Convert to a tensor before the trace:
  tensor_data = torch.tensor(df.values)
```

---

## Summary of Workarounds

| Problem | Workaround |
|---------|------------|
| `nonlocal` closure | Use a class with instance state |
| Mutable class attribute | Use module-level variable instead |
| pandas DataFrame | `torch.tensor(df.values)` |
| PIL Image | `transforms.ToTensor()(image)` |
| matplotlib Figure | `fig.savefig(buf, format='png')` |
| File handle | Read content before trace |
| Database query | Execute query, pass results as list/dict |

---

## Related Documents

- [Source Serialization Tutorial](source-serialization-tutorial.md) — How serialization works internally
- [Disambiguation Design](disambiguation-design.md) — Handling identity and name collisions
- [Forbidden Serialization Design](forbidden-serialization-design.md) — Early rejection of problematic objects
