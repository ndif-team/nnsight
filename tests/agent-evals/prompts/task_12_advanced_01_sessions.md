# Task 12: Sessions for Multi-Trace

**ID:** `advanced_01_sessions`
**Difficulty:** advanced
**Tags:** session, multi-trace, patching

## Setup Code (already provided)

```python
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

## Your Task

Write nnsight code using sessions that:
1. Creates a session context with model.session()
2. In the first trace with input "The weather is", captures layer 3's output
3. In the second trace with input "The climate is", patches layer 3's output with the value from trace 1
4. Saves the final logits from the second trace as `session_logits`

Use `with model.session() as session:` and nested `with model.trace(...):` blocks.
The variable `model` is already loaded.

**Expected Output:** session_logits tensor

## Your Code

Write your solution below:

```python
# Your code here
```
