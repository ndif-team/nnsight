# Task 14: Activation Caching

**ID:** `advanced_03_caching`
**Difficulty:** advanced
**Tags:** cache, trace

## Setup Code (already provided)

```python
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

## Your Task

Write nnsight code that:
1. Creates a trace with input "Hello world"
2. Uses tracer.cache() to cache activations from all modules
3. After the trace, accesses the cached output from layer 5
4. Stores it in a variable called `cached_layer5`

The cache can be accessed like: cache['model.transformer.h.5'].output or cache.model.transformer.h[5].output
The variable `model` is already loaded.

**Expected Output:** cached_layer5 tensor with hidden dim 768

## Your Code

Write your solution below:

```python
# Your code here
```
