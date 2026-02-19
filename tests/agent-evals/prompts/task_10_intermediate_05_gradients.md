# Task 10: Gradient Access with Backward

**ID:** `intermediate_05_gradients`
**Difficulty:** intermediate
**Tags:** backward, gradient, trace

## Setup Code (already provided)

```python
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

## Your Task

Write nnsight code that:
1. Creates a trace with input "Hello world"
2. Gets the hidden state from layer 5's output and enables gradients on it with requires_grad_(True)
3. Computes a simple loss as the sum of the logits
4. Uses `with loss.backward():` to access the gradient
5. Saves the gradient as `hidden_grad`

Remember: access .grad on the TENSOR (not the module) and only inside the backward() context.
The variable `model` is already loaded.

**Expected Output:** hidden_grad tensor with shape [..., 768]

## Your Code

Write your solution below:

```python
# Your code here
```
