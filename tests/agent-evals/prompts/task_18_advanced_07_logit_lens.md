# Task 18: Logit Lens Implementation

**ID:** `advanced_07_logit_lens`
**Difficulty:** advanced
**Tags:** logit_lens, analysis, intermediate_layers

## Setup Code (already provided)

```python
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

## Your Task

Write nnsight code that implements a simple logit lens:
1. Creates a trace with input "The Eiffel Tower is in"
2. For each of the first 6 layers (0-5), applies the final layer norm and lm_head to the hidden state
3. Gets the argmax token prediction at the last position for each layer
4. Stores the predictions as a list called `layer_predictions`

Use model.lm_head(model.transformer.ln_f(hidden_state)) to project intermediate hidden states.
The variable `model` is already loaded.

**Expected Output:** layer_predictions list with 6 token predictions

## Your Code

Write your solution below:

```python
# Your code here
```
