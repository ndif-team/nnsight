# Why Source Serialization Matters for NDIF

*What we are missing to succeed as a platform*

---

After Adam's NeurIPS presentation on NDIF, Neel Nanda asked us a pointed question: **Why isn't NDIF more widely adopted?** He thought it sounded great, but he wasn't seeing the explosive adoption he expected. Was it (a) not really very good technology, or (b) poor marketing?

The honest answer is: a bit of both. But the deeper issue isn't marketing. It's that **NDIF is missing the single most important capability for building a research ecosystem**: support for user-created libraries.

## The Library Problem

Think about how interpretability research actually works. A researcher develops a new technique, say, a novel way to visualize attention patterns, or a method like Patchscopes or Logit Lens. They want to:

1. **Package it as a library** so others can use it
2. **Share it** without coordinating with the NDIF team
3. **Let others extend it** and build their own versions

This is how healthy research ecosystems grow. PyTorch didn't become dominant because the core team implemented every layer type. It succeeded because anyone could write `class MyCustomLayer(nn.Module)` and share it with the world.

But with NDIF's current serialization approach, **none of this works**. If you write a useful function in `my_interpretability_tools.py` and try to use it in an NDIF trace, it fails. The server doesn't have your module. Your code can't run.

This isn't a minor inconvenience. It's an ecosystem killer. Every researcher who wants to build on NDIF must either:
- Coordinate directly with the NDIF team to get their code deployed
- Inline all their logic directly in trace blocks (no reuse, no sharing)
- Give up and run locally

This is why adoption is slow.

## A Clean Platform Specification

[Source-based serialization](source-serialization-tutorial.md) enables a clean architectural contract: the server guarantees a fixed set of modules and resources (torch, numpy, transformers, etc.), and user code must be fully captured in the serialized payload. Nothing else is assumed to exist.

This is a fundamentally different goal than what cloudpickle was designed for. Cloudpickle aims to enable communication between components of the *same* distributed system, where both ends share the same installed packages and codebase. It's a reasonable approach when you control both endpoints. But NDIF is trying to build something more ambitious: **a broad ecosystem of uncoordinated components developed by different researchers**. A library author in one lab should be able to publish code that runs on NDIF without ever talking to the NDIF team or knowing what other libraries exist.

This requires a true [platform specification](remotable-python.md), not just a serialization format. The `remote='local'` test mode simulates the server's restrictions as closely as possible, blocking access to user modules during deserialization. This lets library designers quickly validate that their code is properly captured and will run remotely. If your tests pass with `remote='local'`, your library is remotable.

## Seeing the Problem Concretely

Let's make this concrete with actual test code. Here's a simple utility module that a researcher might write, a `normalize` function and a `RunningStats` class for analyzing model activations:

```python
# mymethods.py - A researcher's utility library
import torch

def normalize(x, dim=-1, eps=1e-8):
    """Normalize a tensor to unit norm."""
    return x / (x.norm(dim=dim, keepdim=True) + eps)

class RunningStats:
    """Accumulate running mean/variance over batches."""
    def __init__(self):
        self.count = 0
        self._mean = None

    def add(self, x):
        # ... accumulate statistics ...
```

Now here's a test that uses this library with `remote='local'` (which simulates what happens when code is sent to NDIF):

```python
# test_import_serialization.py
from nnsight import LanguageModel
from mymethods import normalize

def test_imported_function(gpt2):
    """Test that an imported function works with remote='local'."""
    with gpt2.trace("Hello", remote='local'):
        hidden = gpt2.transformer.h[0].output[0]
        normed = normalize(hidden)
        result = normed.save()

    # Verify normalization worked
    norms = result.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
```

This test is **identical** in both the v0.5.16 branch and the source-serialization branch. What differs is whether it passes or fails.

### On v0.5.16: FAILURE

```
$ pytest tests/test_import_serialization.py::test_imported_function -v

FAILED tests/test_import_serialization.py::test_imported_function

E           ModuleNotFoundError: No module named 'mymethods'

src/nnsight/intervention/serialization.py:492: ModuleNotFoundError
```

The current serialization uses cloudpickle, which stores a *reference* to `mymethods.normalize` rather than the actual code. When the server tries to deserialize, it attempts to import the module, which doesn't exist on the server.

### On source-serialization: SUCCESS

```
$ pytest tests/test_import_serialization.py::test_imported_function -v

tests/test_import_serialization.py::test_imported_function PASSED
```

With source-based serialization, the actual source code of `normalize` is captured and transmitted. The server reconstructs the function from source, without needing the original module.

## Relevant Commits

The implementation and tests demonstrating this approach:

**v0.5.16-bau branch** (cloudpickle serialization, tests FAIL):
- [Branch root](https://github.com/davidbau/nnsight/tree/v0.5.16-bau)
- [981c313](https://github.com/davidbau/nnsight/commit/981c313) - Add `remote='local'` for local serialization testing
- [1f0e3ed](https://github.com/davidbau/nnsight/commit/1f0e3ed) - Add import tests (these fail on v0.5.16)

**source-serialization-design branch** (source-based serialization, tests PASS):
- [Branch root](https://github.com/davidbau/nnsight/tree/source-serialization-design)
- [750d684](https://github.com/davidbau/nnsight/commit/750d684) - Add module isolation for `remote='local'` validation
- [84291a9](https://github.com/davidbau/nnsight/commit/84291a9) - Add import tests (these pass with source serialization)

## On Performance: Capabilities Before Speed

A natural concern is performance. Source-based serialization has overhead: parsing source code, reconstructing functions, validating namespaces. Won't this make NDIF slow?

Here's the thing: **nobody cares how fast a system is if it can't do what they need**.

A researcher who wants to test their Patchscopes implementation on NDIF doesn't care whether it runs in 50ms or 500ms per request. They care that it *runs at all*. The current approach, where their code simply fails with `ModuleNotFoundError`, isn't slow. It's impossible.

Consider Python itself. Python is often 10-100x slower than C for raw computation. By performance metrics alone, Python should have been abandoned years ago. Instead, it became the standard language for large-scale machine learning. Why? Because it gave researchers what they needed: expressiveness, composability, and the ability to build on each other's work. Performance came later, through PyPy, Cython, and ultimately by letting Python orchestrate optimized C/C++ code.

We need to think like platform designers. The right approach is:

1. **First, enable the capabilities researchers need.** Make it possible to use libraries, share code, and build ecosystems.

2. **Then, optimize within a stable specification.** Once the API is right, there are countless ways to make it fast: caching, compilation, lazy evaluation, server-side optimizations.

3. **Let users contribute to performance.** When researchers can write reusable library code, they can develop clever patterns that economize on bandwidth or avoid slow operations. A clean platform API turns our users into collaborators on performance, not just consumers waiting for us to fix things.

The performance problems are solvable engineering challenges. The capability gap is an architectural limitation that blocks adoption entirely. We should fix the architecture first.

## What This Enables

With source-based serialization, a researcher can:

- **Write a library** with custom analysis tools
- **Share it on PyPI or GitHub** like any other Python package
- **Use it in NDIF traces** without any coordination with the NDIF team
- **Let others extend it** by subclassing or modifying
- **Build an ecosystem** where researchers build on each other's work

This is how you get from "interesting research infrastructure" to "ubiquitous platform that everyone uses." Not by having a fast server that can only run blessed code, but by having a flexible platform that runs whatever researchers need.

The NeurIPS crowd was interested in NDIF. They want to use large models for interpretability research. The reason they're not flooding in isn't marketing. It's that the platform doesn't yet support the way researchers actually work.

Source serialization fixes that.
