# Task 6: Multiple Invokers for Batching

**ID:** `intermediate_01_multiple_invokers`
**Difficulty:** intermediate
**Tags:** invoke, batching, trace

## Setup Code (already provided)

```python
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

## Your Task

Write nnsight code that:
1. Creates a trace context WITHOUT an input argument
2. Uses tracer.invoke() to add two separate inputs: "Hello" and "World"
3. For each invoke, saves the last-position logits from lm_head.output[:, -1]
4. Store them as `logits_1` and `logits_2` respectively

Use `with model.trace() as tracer:` pattern and nested `with tracer.invoke(...):`
The variable `model` is already loaded.

**Expected Output:** Two logit tensors with shape [..., 50257]

## Your Code

Write your solution below:

```python
# Your code here
```
