# Task 13: Model Editing (Persistent)

**ID:** `advanced_02_model_editing`
**Difficulty:** advanced
**Tags:** edit, persistent, intervention

## Setup Code (already provided)

```python
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

## Your Task

Write nnsight code that:
1. Creates a persistent edit using model.edit() that zeros out layer 0's output
2. Runs a trace on the edited model with input "Test"
3. Saves layer 0's output as `edited_output`
4. Verifies it's all zeros

Use `with model.edit() as model_edited:` to create the edit, then trace on model_edited.
The variable `model` is already loaded.

**Expected Output:** edited_output tensor with all zeros

## Your Code

Write your solution below:

```python
# Your code here
```
