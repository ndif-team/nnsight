# Task 17: Barrier Synchronization

**ID:** `advanced_06_barrier_sync`
**Difficulty:** advanced
**Tags:** barrier, synchronization, invoke

## Setup Code (already provided)

```python
from nnsight import LanguageModel
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

## Your Task

Write nnsight code that:
1. Creates a trace context with model.trace() as tracer
2. Creates a barrier for 2 participants using tracer.barrier(2)
3. In first invoke with "Paris is the capital of": captures embeddings from wte.output, then calls barrier()
4. In second invoke with "_ _ _ _ _": calls barrier(), patches wte.output with the captured embeddings
5. Saves the patched logits from invoke 2 as `barrier_logits`

Barriers ensure both invokes wait at the same point before proceeding.
The variable `model` is already loaded.

**Expected Output:** barrier_logits tensor

## Your Code

Write your solution below:

```python
# Your code here
```
