# Task 7: Activation Patching Between Invokes

**ID:** `intermediate_02_activation_patching`
**Difficulty:** intermediate
**Tags:** invoke, patching, intervention

## Setup Code (already provided)

```python
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

## Your Task

Write nnsight code to perform activation patching:
1. Create a trace with two invokes
2. First invoke with "The Eiffel Tower is in": capture the hidden state from layer 5's output at the last token position
3. Second invoke with "The Colosseum is in": patch (replace) layer 5's last-token hidden state with the captured one from invoke 1
4. Save the final logits from the second invoke as `patched_logits`

This should "transfer" the location knowledge from the first prompt to the second.
The variable `model` is already loaded.

**Expected Output:** patched_logits tensor with shape [..., 50257]

## Your Code

Write your solution below:

```python
# Your code here
```
