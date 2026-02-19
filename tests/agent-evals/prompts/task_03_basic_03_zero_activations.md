# Task 3: Zero Out Activations

**ID:** `basic_03_zero_activations`
**Difficulty:** basic
**Tags:** trace, intervention, in-place

## Setup Code (already provided)

```python
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

## Your Task

Write nnsight code that:
1. Creates a trace on the model with the input "Hello"
2. Zeros out the output of the first transformer block (model.transformer.h[0])
3. Saves the modified output of that first block in a variable called `zeroed_output`
4. Also saves the final logits in a variable called `logits`

Use in-place modification with slice assignment ([:] = 0) for zeroing.
The variable `model` is already loaded.

**Expected Output:** zeroed_output tensor with all zeros

## Your Code

Write your solution below:

```python
# Your code here
```
