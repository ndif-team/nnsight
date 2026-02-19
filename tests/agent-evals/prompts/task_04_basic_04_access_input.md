# Task 4: Access Module Input

**ID:** `basic_04_access_input`
**Difficulty:** basic
**Tags:** trace, input, save

## Setup Code (already provided)

```python
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

## Your Task

Write nnsight code that:
1. Creates a trace on the model with the input "Machine learning"
2. Saves the INPUT to the 5th transformer block (model.transformer.h[5].input)
3. Store it in a variable called `layer_input`

The variable `model` is already loaded.

**Expected Output:** layer_input tensor with shape [..., 768]

## Your Code

Write your solution below:

```python
# Your code here
```
