# Task 15: Module Skipping

**ID:** `advanced_04_skip_module`
**Difficulty:** advanced
**Tags:** skip, intervention

## Setup Code (already provided)

```python
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

## Your Task

Write nnsight code that:
1. Creates a trace with input "Hello"
2. Gets the output from layer 0
3. Uses .skip() to skip layer 1, passing layer 0's output as the skip value
4. Saves layer 1's output as `skipped_output`

Layer 1's output should equal layer 0's output since we skipped computation.
The variable `model` is already loaded.

**Expected Output:** skipped_output tensor matching layer 0 output

## Your Code

Write your solution below:

```python
# Your code here
```
