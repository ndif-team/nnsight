# Local Simulation Backend

The Local Simulation Backend (`remote='local'`) provides a way to test your trace serialization without connecting to NDIF. It performs the full serialization and deserialization cycle locally, catching any serialization errors before you run on the remote server.

## Quick Start

```python
from nnsight import LanguageModel

model = LanguageModel("gpt2")

# Use remote='local' to test serialization
with model.trace("Hello world", remote='local'):
    hidden = model.transformer.h[0].output[0].save()

print(hidden.shape)
```

If your trace uses any non-serializable values, you'll get a clear error message immediately instead of discovering the problem after submitting to NDIF.

## When to Use

Use `remote='local'` when you want to:

1. **Test serialization before remote execution** - Verify your trace will serialize correctly before submitting to NDIF
2. **Debug serialization errors** - Get clear, local error messages for serialization issues
3. **Verify @remote decorations** - Ensure your custom functions and classes are properly decorated
4. **Develop without network access** - Work offline while still exercising the serialization path

## How It Works

When you specify `remote='local'`, the trace follows this path:

1. **Compile** - The trace block is compiled into a function (same as normal execution)
2. **Serialize** - The trace is serialized using source-based serialization (same as remote execution)
3. **Deserialize** - The serialized payload is deserialized into a fresh namespace (simulating server environment)
4. **Execute** - The trace is executed locally using the same model instance

This exercises the same serialization code path used for NDIF, so any serialization errors that would occur remotely will also occur locally.

## Verbose Mode

Enable verbose mode to see serialization details:

```python
with model.trace("Hello world", remote='local', verbose=True):
    hidden = model.transformer.h[0].output[0].save()
```

This prints:
```
[LocalSimulation] Serialized payload: 1234 bytes
[LocalSimulation] Variables: ['my_var', 'config']
[LocalSimulation] Remote objects: ['MyClass', 'helper_fn']
[LocalSimulation] Model refs: ['model']
```

## Common Errors and Solutions

### Non-serializable Variable

```python
def my_function():  # Not decorated with @remote
    return 42

with model.trace("test", remote='local'):
    x = my_function()  # Error!
```

**Solution**: Decorate with `@nnsight.remote`:

```python
from nnsight import remote

@remote
def my_function():
    return 42

with model.trace("test", remote='local'):
    x = my_function()  # Works!
```

### Non-server-available Module

```python
import subprocess  # Not available on NDIF server

with model.trace("test", remote='local'):
    result = subprocess  # Error!
```

**Solution**: Only use modules available on the server (torch, numpy, etc.), or compute the value before the trace and pass it as a JSON-serializable value.

### Lambda with Non-serializable Capture

```python
import my_custom_lib

double = lambda x: my_custom_lib.process(x)  # Captures my_custom_lib

with model.trace("test", remote='local'):
    result = double(hidden)  # Error!
```

**Solution**: Convert to a `@remote` decorated function:

```python
@remote
def double(x):
    import my_custom_lib  # Import inside function
    return my_custom_lib.process(x)
```

## Differences from Remote Execution

While `remote='local'` simulates remote execution, there are some differences:

| Aspect | remote='local' | remote=True |
|--------|---------------|-------------|
| Model | Same instance | Fresh load on server |
| Network | None required | NDIF connection required |
| GPU | Uses local GPU | Uses NDIF GPU |
| Dependencies | Local environment | Server environment |
| Error location | Immediate | After network round-trip |

## API Reference

### trace() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `remote` | `bool \| str` | `False` | `False` (local), `True` (NDIF), or `'local'` (simulation) |
| `verbose` | `bool` | `False` | If `True` and `remote='local'`, print serialization details |
| `strict_remote` | `bool` | `False` | If `True`, require explicit `@remote` decorations |
| `max_upload_mb` | `float` | `10.0` | Warn if upload payload exceeds this size. Set to 0 to disable. |

### LocalSimulationBackend

```python
from nnsight.intervention.backends.local_simulation import LocalSimulationBackend

# Create backend directly (usually not needed)
backend = LocalSimulationBackend(model, verbose=True)

# Access last payload size
size = backend.last_payload_size  # int: bytes

# Access last payload (for debugging)
payload = backend.get_last_payload()  # bytes or None
```

## Examples

### Testing a Complex Trace

```python
from nnsight import LanguageModel, remote

@remote
class LayerAnalyzer:
    def __init__(self, layer_idx):
        self.layer_idx = layer_idx

    def analyze(self, hidden):
        return hidden.mean(dim=-1)

model = LanguageModel("gpt2")
analyzer = LayerAnalyzer(5)

# Test locally before submitting to NDIF
with model.trace("The capital of France", remote='local', verbose=True):
    hidden = model.transformer.h[analyzer.layer_idx].output[0]
    analysis = analyzer.analyze(hidden).save()

print(f"Analysis shape: {analysis.shape}")
# If this works, you can safely run with remote=True
```

### Verifying Tensor Serialization

```python
import torch

replacement = torch.randn(1, 768)

# Test that the tensor serializes correctly
with model.trace("test", remote='local'):
    model.transformer.h[0].output[0][:] = replacement
    result = model.transformer.h[0].output[0].save()

print(f"Tensor serialized and restored correctly: {result.shape}")
```

## See Also

- [Source Serialization Tutorial](source-serialization-tutorial.md) - Detailed explanation of the serialization system
- [@remote Decorator](remote-decorator.md) - How to make functions and classes serializable
- [NDIF Documentation](https://ndif.us/docs) - Remote execution on NDIF
