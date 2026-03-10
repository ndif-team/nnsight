# Task 9: Iterate Over Generation Steps

**ID:** `intermediate_04_iter_generation`
**Difficulty:** intermediate
**Tags:** generate, iter, multi-token

## Setup Code (already provided)

```python
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

## Your Task

Write nnsight code that:
1. Uses model.generate() with input "Hello" and max_new_tokens=3
2. Uses tracer.iter[:] to iterate over all generation steps
3. In each step, appends the argmax of the last logit position to a list
4. Store the list of predicted tokens as `step_tokens`

Create a list with list().save() inside the trace, then use `with tracer.iter[:]:` to iterate.
The variable `model` is already loaded.

**Expected Output:** step_tokens list with 3 token predictions

## Your Code

Write your solution below:

```python
# Your code here
```
