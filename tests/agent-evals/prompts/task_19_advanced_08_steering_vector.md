# Task 19: Steering Vector Application

**ID:** `advanced_08_steering_vector`
**Difficulty:** advanced
**Tags:** steering, intervention, vector

## Setup Code (already provided)

```python
from nnsight import LanguageModel
import torch
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

## Your Task

Write nnsight code that applies a steering vector:
1. Creates a random steering vector of shape (768,) on the model's device
2. Creates a trace with input "I think that"
3. Adds the steering vector (scaled by 0.1) to layer 10's output at the last token position
4. Saves the modified logits as `steered_logits`

Make sure to match the device of the steering vector with the model's activations.
The variable `model` is already loaded.

**Expected Output:** steered_logits tensor

## Your Code

Write your solution below:

```python
# Your code here
```
