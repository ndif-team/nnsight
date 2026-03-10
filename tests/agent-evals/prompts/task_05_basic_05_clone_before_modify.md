# Task 5: Clone Before Modify

**ID:** `basic_05_clone_before_modify`
**Difficulty:** basic
**Tags:** trace, clone, intervention

## Setup Code (already provided)

```python
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

## Your Task

Write nnsight code that:
1. Creates a trace on the model with the input "Test"
2. Clones the output of transformer block 0 and saves it as `before`
3. Then zeros out the output of that block (in-place)
4. Saves the modified output as `after`

After the trace, `before` should contain the original values and `after` should be zeros.
The variable `model` is already loaded.

**Expected Output:** before has original values, after is all zeros

## Your Code

Write your solution below:

```python
# Your code here
```
