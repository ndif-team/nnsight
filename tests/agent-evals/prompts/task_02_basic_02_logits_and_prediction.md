# Task 2: Access Logits and Predict Token

**ID:** `basic_02_logits_and_prediction`
**Difficulty:** basic
**Tags:** trace, save, logits, lm_head

## Setup Code (already provided)

```python
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

## Your Task

Write nnsight code that:
1. Creates a trace on the model with the input "The capital of France is"
2. Saves the logits from the language model head (model.lm_head.output)
3. Gets the predicted next token by taking argmax of the last position
4. Store the logits in `logits` and the predicted token id in `predicted_token`

The variable `model` is already loaded as a LanguageModel("openai-community/gpt2").

**Expected Output:** logits tensor [1, seq_len, 50257] and predicted_token scalar

## Your Code

Write your solution below:

```python
# Your code here
```
