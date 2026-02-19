# Task 1: Basic Trace and Save

**ID:** `basic_01_trace_and_save`
**Difficulty:** basic
**Tags:** trace, save, output

## Setup Code (already provided)

```python
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

## Your Task

Write nnsight code that:
1. Creates a trace on the model with the input "Hello world"
2. Saves the output of the final transformer block (model.transformer.h[-1])
3. The saved value should be stored in a variable called `hidden_states`

The variable `model` is already loaded as a LanguageModel("openai-community/gpt2").
After the trace context, `hidden_states` should contain the actual tensor.

**Expected Output:** hidden_states tensor with shape [1, seq_len, 768]

## Your Code

Write your solution below:

```python
# Your code here
```
