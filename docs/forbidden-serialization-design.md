# Forbidden Serialization Design

This document describes the design for early detection and blocking of objects that should not be serialized for remote execution, even if they technically could be.

## Problem Statement

The auto-discovery serialization system attempts to serialize any Python object by extracting its source code and instance state. However, some objects:

1. **Represent OS/system resources** that cannot be transferred between machines (sockets, file handles, locks)
2. **Have huge dependency graphs** that generate many warnings before eventually failing (pandas DataFrames, matplotlib figures)
3. **Leak into scope unintentionally** from test frameworks or other infrastructure (pytest fixtures)

Currently, these objects either:
- "Succeed" at serialization but fail at deserialization (sockets)
- Generate 50-100+ warnings before eventually failing on a specific attribute (pandas)
- Cause confusing errors unrelated to the user's actual code (pytest fixtures)

## Goals

1. **Fail fast** - Detect forbidden objects immediately, not after extensive processing
2. **Fail clearly** - Provide specific, actionable error messages
3. **Suggest alternatives** - Tell users exactly how to work around the issue
4. **Be extensible** - Allow easy addition of new forbidden patterns

## Discovery Findings

### Category 1: OS/System Resources

These have Python wrappers but represent system state that cannot be transferred:

| Type | Current Behavior | Issue |
|------|-----------------|-------|
| `socket.socket` | Serializes with warnings, fails at deserialization | Has Python wrapper (socket.py) but depends on `_socket` C extension |
| `threading.Lock/RLock` | Correctly rejected | Good - no source available |
| `multiprocessing.Process/Queue` | Not tested | Would have similar issues |
| File handles | Correctly rejected | Good - contains `TextIOWrapper` |

### Category 2: Data Science Objects with Dependency Explosion

These trigger massive warning cascades before failing:

| Type | Warnings Generated | Final Error |
|------|-------------------|-------------|
| `pandas.DataFrame` | 50+ class warnings | `_mgr.axes[0]._range` is `range` type |
| `pandas.Series` | 30+ class warnings | Same `range` issue |
| `matplotlib.Figure` | 20+ class warnings | `stale_callback` is function |
| `matplotlib.Axes` | 20+ class warnings | Same callback issue |

### Category 3: Test Framework Leakage

When running tests, pytest fixtures leak into the local scope:

| Type | Issue |
|------|-------|
| `_pytest.fixtures.FixtureRequest` | Auto-discovered, generates warnings |
| `_pytest.fixtures.SubRequest` | Same |
| `_pytest.capture.CaptureFixture` | Same |

### Category 4: Database/Network Connections

These represent network state that cannot be transferred:

| Type | Issue |
|------|-------|
| `sqlalchemy.orm.Session` | Connection pool state |
| `requests.Session` | Contains `RLock` in cookies |
| Database cursors | Connection state |

### Category 5: ML Framework Objects (Correctly Handled)

| Type | Behavior | Notes |
|------|----------|-------|
| `sklearn.LogisticRegression` | Clean rejection | "source code not available" - Cython class |
| `torch.nn.Module` | Should work | This is intentionally supported |

## Design

### Forbidden Lists

We define three types of forbidden patterns:

```python
# 1. Forbidden module prefixes - skip auto-discovery entirely
FORBIDDEN_MODULE_PREFIXES = frozenset({
    '_pytest',          # Test framework internals
    'pytest',           # Test framework
    'unittest',         # Test framework
    'socket',           # Network resources (has Python wrapper over C)
    'multiprocessing',  # Process resources
    'asyncio',          # Async resources
    'concurrent',       # Thread pools
    'queue',            # Thread communication
    'subprocess',       # OS processes
    'sqlite3',          # DB connections
    'sqlalchemy',       # ORM sessions
    'pymongo',          # MongoDB client
    'redis',            # Redis client
    'psycopg',          # PostgreSQL
    'mysql',            # MySQL
    'logging',          # Handlers have file references
})

# 2. Forbidden class patterns with specific messages
FORBIDDEN_CLASSES = {
    'pandas.core.frame.DataFrame': (
        "pandas.DataFrame cannot be serialized for remote execution.\n"
        "Convert to a tensor before the trace:\n"
        "  tensor_data = torch.tensor(df.values)\n"
        "Or extract specific columns:\n"
        "  values = df['column'].tolist()"
    ),
    'pandas.core.series.Series': (
        "pandas.Series cannot be serialized for remote execution.\n"
        "Convert to a tensor or list:\n"
        "  tensor_data = torch.tensor(series.values)\n"
        "  values = series.tolist()"
    ),
    'matplotlib.figure.Figure': (
        "matplotlib.Figure cannot be serialized for remote execution.\n"
        "Figures contain rendering state that cannot be transferred.\n"
        "If you need to pass image data, save to bytes first:\n"
        "  import io\n"
        "  buf = io.BytesIO()\n"
        "  fig.savefig(buf, format='png')\n"
        "  image_bytes = buf.getvalue()"
    ),
    'matplotlib.axes._axes.Axes': (
        "matplotlib.Axes cannot be serialized for remote execution.\n"
        "Axes contain rendering state bound to a Figure."
    ),
    'PIL.Image.Image': (
        "PIL.Image cannot be serialized for remote execution.\n"
        "Convert to a tensor first:\n"
        "  from torchvision import transforms\n"
        "  tensor = transforms.ToTensor()(image)"
    ),
}

# 3. Forbidden instance types (checked via isinstance)
FORBIDDEN_INSTANCE_TYPES = (
    # These are tuples of (type_check_string, message)
    # We use strings to avoid importing the modules
)
```

### Check Order

The forbidden check happens BEFORE any auto-discovery attempt:

```
User object
    │
    ▼
┌─────────────────────────┐
│ 1. Is it JSON-primitive?│──Yes──▶ Serialize as variable
└─────────────────────────┘
    │ No
    ▼
┌─────────────────────────┐
│ 2. Is it a tensor/array?│──Yes──▶ Serialize tensor data
└─────────────────────────┘
    │ No
    ▼
┌─────────────────────────────────┐
│ 3. Is module prefix forbidden?  │──Yes──▶ EARLY ERROR with message
└─────────────────────────────────┘
    │ No
    ▼
┌─────────────────────────────────┐
│ 4. Is class name forbidden?     │──Yes──▶ EARLY ERROR with message
└─────────────────────────────────┘
    │ No
    ▼
┌─────────────────────────────────┐
│ 5. Attempt auto-discovery       │
└─────────────────────────────────┘
```

### Error Message Format

```
Cannot serialize 'df' for remote execution: pandas.DataFrame is not supported.

pandas.DataFrame cannot be serialized for remote execution.
Convert to a tensor before the trace:
  tensor_data = torch.tensor(df.values)
Or extract specific columns:
  values = df['column'].tolist()
```

### API

```python
def is_forbidden_for_serialization(obj: Any) -> Tuple[bool, Optional[str]]:
    """
    Check if an object is forbidden from serialization.

    Returns:
        (is_forbidden, error_message) - If forbidden, error_message explains why
        and suggests alternatives.
    """

def check_forbidden_or_raise(name: str, obj: Any) -> None:
    """
    Check if object is forbidden and raise SourceSerializationError if so.
    """
```

## Implementation Location

The forbidden checks are implemented in `serialization_source.py` and called from `extract_all()` before any auto-discovery is attempted.

## Testing

Tests verify:
1. Each forbidden category is caught early (no warnings generated)
2. Error messages are clear and actionable
3. Allowed objects (torch tensors, user classes) still work
4. The forbidden lists can be extended

## Future Extensions

1. **User-configurable forbidden list** - Allow users to add their own forbidden patterns
2. **Allowed overrides** - Allow `@remote` to override forbidden status for special cases
3. **Warning mode** - Option to warn instead of error for borderline cases
