# Task 8: Multi-Token Generation

**ID:** `intermediate_03_generation`
**Difficulty:** intermediate
**Tags:** generate, multi-token

## Setup Code (already provided)

```python
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

## Your Task

Write nnsight code that:
1. Uses model.generate() with input "Once upon a time" and max_new_tokens=5
2. Saves the final generated output using model.generator.output
3. Store it in a variable called `generated_tokens`

Use `with model.generate("Once upon a time", max_new_tokens=5):` pattern.
The variable `model` is already loaded.

**Expected Output:** generated_tokens tensor with original + 5 new tokens

## Your Code

Write your solution below:

```python
# Your code here
```
