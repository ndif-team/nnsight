# Task 16: Scan Mode for Shape Discovery

**ID:** `advanced_05_scan_mode`
**Difficulty:** advanced
**Tags:** scan, shapes

## Setup Code (already provided)

```python
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

## Your Task

Write nnsight code that:
1. Uses model.scan() (not trace) with input "Hello world"
2. Gets the shape of the output from the last transformer block
3. Stores the hidden dimension (last element of shape) as `hidden_dim`

Use model.scan() instead of model.trace() - this gives you shapes without running the full model.
The variable `model` is already loaded.

**Expected Output:** hidden_dim = 768

## Your Code

Write your solution below:

```python
# Your code here
```
