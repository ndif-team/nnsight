# Task 11: Prompt-less Invoker for Batch

**ID:** `intermediate_06_promptless_invoke`
**Difficulty:** intermediate
**Tags:** invoke, batch, promptless

## Setup Code (already provided)

```python
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

## Your Task

Write nnsight code that:
1. Creates a trace context with tracer
2. Adds three inputs via separate invoke calls: "A", "B", "C"
3. Uses a prompt-less invoke (tracer.invoke() with no arguments) to access the combined batch
4. In the prompt-less invoke, saves the lm_head output as `combined_logits`

The combined_logits should have batch dimension 3 (all three inputs batched together).
The variable `model` is already loaded.

**Expected Output:** combined_logits with batch dimension 3

## Your Code

Write your solution below:

```python
# Your code here
```
