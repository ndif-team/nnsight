# llms.md - NNsight AI Agent Guide

This document provides comprehensive guidance for AI agents working with the `nnsight` library. NNsight enables interpreting and manipulating the internals states of deep learning models through a deferred execution tracing system.

### Related Resources

- **[NNsight.md](./NNsight.md)** - Deep technical documentation covering nnsight's internal architecture (tracing, interleaving, Envoy system, vLLM integration, etc.)
- **[Documentation](https://www.nnsight.net)** - Official docs with tutorials, guides, and API reference
- **[Forum](https://discuss.ndif.us/)** - Community forum for questions, discussions, and troubleshooting

---

## Quick Reference

```python
from nnsight import NNsight, LanguageModel
import torch

# For any PyTorch model
model = NNsight(torch_model)

# For HuggingFace language models
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

# Basic tracing pattern
with model.trace("input text"):
    hidden_states = model.transformer.h[-1].output[0].save()  # Access and save activations
    model.transformer.h[0].output[0][:] = 0  # Modify activations in-place
```

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [NNsight vs LanguageModel](#nnsight-vs-languagemodel)
3. [Tracing Context](#tracing-context)
4. [Accessing Activations](#accessing-activations)
5. [Modifying Activations (Interventions)](#modifying-activations-interventions)
6. [Batching with Invokers](#batching-with-invokers)
7. [Multi-Token Generation](#multi-token-generation)
8. [Gradients and Backpropagation](#gradients-and-backpropagation)
9. [Conditionals and Iteration](#conditionals-and-iteration)
10. [Model Editing](#model-editing)
11. [Scanning and Validation](#scanning-and-validation)
12. [Caching Activations](#caching-activations)
13. [Source Tracing](#source-tracing)
14. [Module Skipping](#module-skipping)
15. [vLLM Integration](#vllm-integration)
16. [Remote Execution (NDIF)](#remote-execution-ndif)
17. [Sessions](#sessions)
18. [Common Patterns](#common-patterns)
19. [Critical Gotchas](#critical-gotchas)
20. [Debugging Tips](#debugging-tips)
21. [Configuration](#configuration)

---

## Core Concepts

### Deferred Execution Model

NNsight uses a **deferred execution** paradigm with **thread-based synchronization**. Here's how it works:

1. **Code extraction**: When you enter a `with model.trace(...)` block, nnsight immediately exits the block (before your code runs) and extracts/compiles your code using AST parsing
2. **Thread execution**: Your intervention code runs in a separate worker thread
3. **Value synchronization**: When your code accesses `.output` or `.input`, the thread **blocks and waits** until that value is available from the model
4. **Hook-based injection**: The model's forward pass uses PyTorch hooks to provide values to waiting threads
5. **Coordinated execution**: After providing a value, the main thread waits for the worker thread to either request another value or finish

```python
with model.trace("Hello"):
    # This code is extracted, compiled, and run in a worker thread
    # When we access .output, the thread WAITS until the model provides it
    hs = model.transformer.h[-1].output[0]
    
    # .save() marks the value to persist after the context exits
    hs = hs.save()

# After exiting, hs contains the actual tensor
print(hs.shape)  # torch.Size([1, 2, 768])
```

**Key insight:** Your code runs directly . When you write `torch.sum(module.output)`, that's real PyTorch code executing in a thread - it just waits for `module.output` to be available first.

### Threading Model and Invokers

Each **invoke** runs its intervention code in a separate worker thread. This is a critical architectural concept:

1. **Each invoke is a thread** - When you call `tracer.invoke(...)`, you're creating a worker thread that runs your intervention code
2. **Threads run serially** - Only one thread executes at a time (no race conditions)
3. **Invokes execute in definition order** - The order you define invokes is the order they run
4. **Threads wait for values** - When your code accesses `.input`, `.output`, or `.source`, the thread blocks until the model provides that value via hooks
5. **Access modules in execution order** - Within an invoke, you MUST access modules in forward-pass order. Requesting layer 5's output then layer 2's output will deadlock (layer 2 already ran)

```python
with model.trace() as tracer:
    # Invoke 1: Worker thread 1 - runs first
    with tracer.invoke("Hello"):
        # Access modules in execution order ONLY
        layer_2 = model.transformer.h[2].output.save()  # Thread waits here (OK)
        layer_5 = model.transformer.h[5].output.save()  # Then waits here (OK)

    # Invoke 2: Worker thread 2 - runs after thread 1 completes
    with tracer.invoke("World"):
        layer_0 = model.transformer.h[0].output.save()
```
/*
!IMPORTANT:
You MUST access modules (and their .output, .input, etc.) in the order they execute in the model's forward pass.
Accessing layer 5's output before layer 2 will result in a deadlock or error!
*/

### Prompt-less Invokers

Call `.invoke()` with **no arguments** to run intervention code on the **entire batch** from all previous invokes:

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        out_1 = model.lm_head.output[:, -1].save()  # Shape: [1, vocab]
    
    with tracer.invoke(["World", "Test"]):
        out_2 = model.lm_head.output[:, -1].save()  # Shape: [2, vocab]
    
    # No-arg invoke: operates on ALL 3 inputs batched together
    with tracer.invoke():
        out_all = model.lm_head.output[:, -1].save()  # Shape: [3, vocab]
    
    # Another no-arg invoke: same batch
    with tracer.invoke():
        out_all_2 = model.lm_head.output[:, -1].save()  # Shape: [3, vocab]

# out_all contains the same data as concatenating out_1 and out_2
```

This is useful for:
- Running different intervention logic on the same batch
- Accessing the combined batch after setting up individual invokes
- Comparing interventions across the full batch

### Key Properties

Every module wrapped by NNsight has these special properties. Accessing them causes the worker thread to **wait** until the value is available:

| Property | Description |
|----------|-------------|
| `.output` | The module's forward pass output (thread waits for hook) |
| `.input` | The first positional argument to the module |
| `.inputs` | All inputs as `(tuple(args), dict(kwargs))` |

**Note on `.grad`:** Gradients are accessed on **tensors** (not modules), and only inside a `with tensor.backward():` context. See [Gradients and Backpropagation](#gradients-and-backpropagation).

---

## NNsight vs LanguageModel

### NNsight (Base Class)

Use `NNsight` for any PyTorch model:

```python
from nnsight import NNsight
import torch

net = torch.nn.Sequential(
    torch.nn.Linear(5, 10),
    torch.nn.Linear(10, 2)
)

model = NNsight(net)

with model.trace(torch.rand(1, 5)):
    output = model.output.save()
```

### LanguageModel (HuggingFace Integration)

Use `LanguageModel` for HuggingFace transformers with automatic tokenization:

```python
from nnsight import LanguageModel

# Loads model + tokenizer automatically
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

# Can pass strings directly - tokenization is handled
with model.trace("The Eiffel Tower is in"):
    hidden_states = model.transformer.h[-1].output[0].save()
```

**How LanguageModel Works:**

`LanguageModel` is a thin wrapper around HuggingFace's `transformers` library. Under the hood, it uses `AutoModelForCausalLM.from_pretrained()` (or similar Auto classes) to load the model. This means:

1. **Keyword arguments are forwarded** - Any kwargs you pass to `LanguageModel()` are forwarded directly to the underlying HuggingFace loading function
2. **Same model, enhanced interface** - The wrapped model is identical to what you'd get from `transformers`, but with NNsight's intervention capabilities added
3. **Tokenizer included** - The appropriate tokenizer is loaded automatically alongside the model

```python
# These kwargs are passed directly to AutoModelForCausalLM.from_pretrained()
model = LanguageModel(
    "meta-llama/Llama-3.1-8B",
    device_map="auto",           # HuggingFace accelerate device mapping
    torch_dtype=torch.float16,   # Precision setting
    trust_remote_code=True,      # For custom model code
    attn_implementation="flash_attention_2",  # Attention backend
)
```

**Important `LanguageModel` parameters:**

| Parameter | Description |
|-----------|-------------|
| `device_map` | Device placement - `"auto"` distributes layers across available GPUs (and CPU if needed). Uses HuggingFace Accelerate. Other options: `"cuda"`, `"cpu"`, or a custom dict |
| `dispatch=True` | Load model weights into memory immediately. Default is lazy loading (meta tensors) which is faster for initialization but requires dispatch before use |
| `torch_dtype` | Model precision (e.g., `torch.float16`, `torch.bfloat16`). Forwarded to HuggingFace |
| `rename={...}` | Create module aliases (see [Module Renaming](#module-renaming)) |

**Note on `device_map="auto"`:** This tells HuggingFace Accelerate to automatically distribute model layers across all available GPUs. If the model doesn't fit on GPUs, it will offload to CPU. This is the recommended setting for large models.

**Wrapping pre-loaded models:**

You can wrap an existing HuggingFace model, but **you must provide the tokenizer**:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model yourself
hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# MUST provide tokenizer
model = LanguageModel(hf_model, tokenizer=tokenizer)
```

Without `tokenizer=`, you'll get:
```
AttributeError: Tokenizer not found. If you passed a pre-loaded model to 
`LanguageModel`, you need to provide a tokenizer when initializing.
```

---

## Tracing Context

### Basic Tracing

```python
# Single input - requires at least one positional argument
with model.trace("Hello World"):
    output = model.output.save()

# With generation (for multi-token output)
with model.generate("Hello", max_new_tokens=5):
    output = model.generator.output.save()
```

### Trace With vs Without Invokes

**Important:** When using `.trace()` without explicit invokes, you **must** provide at least one positional argument:

```python
# CORRECT - positional argument provided
with model.trace("Hello"):
    output = model.output.save()

# CORRECT - using explicit invokes (no arg to trace needed)
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        output = model.output.save()

# WRONG - no positional arg and no invokes
with model.trace():  # Error!
    output = model.output.save()
```

When you provide an argument to `.trace()`, an implicit invoke is created for you. This is equivalent to:

```python
# These are equivalent:
with model.trace("Hello"):
    output = model.output.save()

with model.trace() as tracer:
    with tracer.invoke("Hello"):
        output = model.output.save()
```

### Context Objects

The tracing context returns a `tracer` object with useful methods:

```python
with model.trace("Hello") as tracer:
    print("Debug:", model.layer1.output.shape)  # Print works normally
    tracer.stop()  # Early termination
```

---

## Accessing Activations

### Module Hierarchy

Print the model to see its structure:

```python
print(model)
# GPT2LMHeadModel(
#   (transformer): GPT2Model(
#     (h): ModuleList(
#       (0-11): 12 x GPT2Block(
#         (attn): GPT2Attention(...)
#         (mlp): GPT2MLP(...)
#       )
#     )
#   )
#   (lm_head): Linear(...)
# )
```

### Accessing Outputs

```python
with model.trace("Hello"):
    # Access specific layer output
    layer_5_out = model.transformer.h[5].output[0].save()
    
    # Access attention output
    attn_out = model.transformer.h[0].attn.output[0].save()
    
    # Access MLP output  
    mlp_out = model.transformer.h[0].mlp.output.save()
    
    # Access final logits
    logits = model.lm_head.output.save()
```

### Accessing Inputs

```python
with model.trace("Hello"):
    # First positional argument
    layer_input = model.transformer.h[0].input.save()
    
    # All inputs: (args_tuple, kwargs_dict)
    all_inputs = model.transformer.h[0].inputs.save()
```

### The `.save()` Method

**Critical:** Values are garbage collected unless you call `.save()`:

```python
with model.trace("Hello"):
    # WRONG - value will be lost
    output = model.transformer.h[-1].output[0]

    # CORRECT - value persists after context
    output = model.transformer.h[-1].output[0].save()
```

**Two ways to save:**

```python
# Method 1: .save() on tensor (uses pymount C extension)
output = model.transformer.h[-1].output[0].save()

# Method 2: nnsight.save() - PREFERRED (works for all objects)
import nnsight
output = nnsight.save(model.transformer.h[-1].output[0])

# nnsight.save() is preferred because:
# - Works on any object (including those with their own .save() method)
# - Doesn't require the pymount C extension
# - More explicit about what's happening
```

---

## Modifying Activations (Interventions)

### In-Place Modification

Use slice assignment for in-place modifications:

```python
with model.trace("Hello"):
    # Zero out all activations
    model.transformer.h[0].output[0][:] = 0
    
    # Modify specific positions
    model.transformer.h[0].output[0][:, -1, :] = 0  # Last token only
    model.transformer.h[0].output[0][:, :, 0] = 1   # First hidden dim
```

### Replacement

Use direct assignment to replace the entire output:

```python
with model.trace("Hello"):
    # Clone and modify
    hs = model.transformer.h[0].output[0].clone()
    hs = hs * 2
    model.transformer.h[0].output[0] = hs
    
    # Or with torch operations
    model.transformer.wte.output = model.transformer.wte.output * 0.5
```

### Tuple Outputs

Many modules return tuples. Handle carefully:

```python
with model.trace("Hello"):
    # GPT2 transformer blocks return (hidden_states, attention_weights, ...)
    full_output = model.transformer.h[0].output  # This is a tuple
    
    # Replace entire tuple
    model.transformer.h[0].output = (
        torch.zeros_like(model.transformer.h[0].output[0]),
    ) + model.transformer.h[0].output[1:]
```

### Clone Before Saving Modified Values

If you modify in-place and want to see the "before" state:

```python
with model.trace("Hello"):
    # Clone BEFORE the modification
    before = model.transformer.h[0].output[0].clone().save()
    
    model.transformer.h[0].output[0][:] = 0
    
    after = model.transformer.h[0].output[0].save()
```

---

## Batching with Invokers

Process multiple inputs in a single forward pass using invokers. Remember: **each invoke is a separate logical thread** that runs serially in definition order.

```python
with model.trace() as tracer:
    # First invoke: Thread 1 - runs first
    with tracer.invoke("The Eiffel Tower is in"):
        embeddings = model.transformer.wte.output
        output1 = model.lm_head.output.save()
    
    # Second invoke: Thread 2 - runs after Thread 1 completes
    # Can reference values from first invoke
    with tracer.invoke("_ _ _ _ _ _ _"):
        model.transformer.wte.output = embeddings  # Cross-invoke intervention
        output2 = model.lm_head.output.save()
```

### Prompt-less Invokers for Batch-Wide Operations

Use `.invoke()` with no arguments to run intervention code on the **entire batch**:

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        pass
    
    with tracer.invoke("World"):
        pass
    
    # No-arg invoke: new thread that sees the combined batch
    with tracer.invoke():
        # Can access modules in any order (separate thread)
        all_outputs = model.lm_head.output.save()  # Shape: [2, seq, vocab]
```

### Barriers for Cross-Invoke Synchronization

When you need values from one invoke before another proceeds:

```python
with model.generate(max_new_tokens=3) as tracer:
    barrier = tracer.barrier(2)  # Create barrier for 2 invokes
    
    with tracer.invoke("Madison Square Garden is in the city of"):
        embeddings = model.transformer.wte.output
        barrier()  # Wait here
        output1 = model.generator.output.save()
    
    with tracer.invoke("_ _ _ _ _ _ _ _ _"):
        barrier()  # Wait here
        model.transformer.wte.output = embeddings  # Now safe to use
        output2 = model.generator.output.save()
```

### Batched Inputs

```python
with model.trace() as tracer:
    # Single prompt
    with tracer.invoke("Hello"):
        out1 = model.lm_head.output[:, -1].save()  # Shape: [1, vocab]
    
    # Multiple prompts batched
    with tracer.invoke(["Hello", "World"]):
        out2 = model.lm_head.output[:, -1].save()  # Shape: [2, vocab]
```

---

## Multi-Token Generation

### Using `.generate()`

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    # Interventions here apply to ALL generation steps by default
    hidden_states = model.transformer.h[-1].output[0].save()
    
    # Get the generated output
    output = model.generator.output.save()

decoded = model.tokenizer.decode(output[0])
```

### Iterating Over Generation Steps

Use `tracer.iter[...]` to access specific generation steps. The `iter` property accepts:
- **Slice**: `tracer.iter[:]` (all steps), `tracer.iter[1:3]` (steps 1-2)
- **Int**: `tracer.iter[2]` (step 2 only)
- **List**: `tracer.iter[[0, 2, 4]]` (specific steps)

Each `iter` moves a cursor that tracks which generation step the mediator (worker thread) is requesting.

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    logits = list().save()
    
    # All steps (slice)
    with tracer.iter[:]:
        logits.append(model.lm_head.output[0][-1].argmax(dim=-1))
    
# Or specific steps
with model.generate("Hello", max_new_tokens=5) as tracer:
    logits = list().save()
    
    # Steps 1-3 only (slice)
    with tracer.iter[1:3]:
        logits.append(model.lm_head.output)

# Single step (int)
with model.generate("Hello", max_new_tokens=5) as tracer:
    with tracer.iter[0]:
        first_logits = model.lm_head.output.save()

# Specific steps (list)
with model.generate("Hello", max_new_tokens=5) as tracer:
    logits = list().save()
    with tracer.iter[[0, 2, 4]]:
        logits.append(model.lm_head.output)
```

### Conditional Interventions Per Step

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    outputs = list().save()
    
    with tracer.iter[:] as step_idx:
        if step_idx == 2:
            # Only intervene on step 2
            model.transformer.h[0].output[0][:] = 0
        
        outputs.append(model.transformer.h[-1].output[0])
```

### Using `.all()` for Recursive Application

```python
with model.generate("Hello", max_new_tokens=3) as tracer:
    hidden_states = list().save()
    
    # Apply to all generation steps for all descendants
    with tracer.all():
        model.transformer.h[0].output[0][:] = 0
        hidden_states.append(model.transformer.h[-1].output)
```

### Using `.next()` for Manual Stepping

```python
with model.generate("Hello", max_new_tokens=3) as tracer:
    # First token
    hs1 = model.transformer.h[-1].output[0].save()
    
    # Second token
    hs2 = model.transformer.h[-1].next().output[0].save()
    
    # Third token
    hs3 = model.transformer.h[-1].next().output[0].save()
```

### ⚠️ Critical Footgun: Unbounded Iteration

**When using `tracer.iter[:]` or `tracer.all()`, code AFTER the iter block never executes.**

These unbounded iterators don't know when to stop — they wait forever for the "next" iteration. When generation finishes:

1. The iterator is still waiting for more iterations
2. NNsight issues a warning (not an error)
3. **All code after the iter block is skipped**

```python
# WRONG:
with model.generate("Hello", max_new_tokens=3) as tracer:
    with tracer.iter[:]:
        hidden = model.transformer.h[-1].output.save()
    
    # ⚠️ THIS NEVER EXECUTES!
    final_logits = model.output.save()

print(final_logits)  # NameError: 'final_logits' is not defined
```

**Solutions:**

1. **Use a separate empty invoker** (recommended):
   ```python
   with model.generate("Hello", max_new_tokens=3) as tracer:
       with tracer.invoke():  # First invoker - handles iteration
           with tracer.iter[:]:
               hidden = model.transformer.h[-1].output.save()
       
       with tracer.invoke():  # Second invoker - runs after generation
           final_logits = model.output.save()  # Now runs!
   ```
2. **Use bounded iteration** if you know the count: `tracer.iter[:3]`
3. **Use `tracer.next()`** for explicit step control

---

## Gradients and Backpropagation

**Important:** Gradients are accessed on **tensors** (not modules), and only inside a `with tensor.backward():` context.

### How Backward Works

When you use `with tensor.backward():`, nnsight creates a **completely separate interleaving session**:

1. When you `import nnsight`, it monkey-patches `torch.Tensor.backward` to check if it's being used as a trace
2. Inside the backward context, a new interleaving session runs with tensor gradient hooks
3. You can ONLY access `.grad` on tensors inside this context - not `.output`, `.input`, etc.
4. After the backward context exits, the original forward trace resumes

This means: **get any `.output` values you need BEFORE entering the backward context**.

### Gradient Access Order (Reverse of Forward Pass)

Just like the forward pass requires accessing modules in execution order, **the backward pass requires accessing gradients in reverse order**. This is because gradients flow backwards through the model:

```
Forward pass order:  layer0 → layer1 → ... → layer11 → lm_head → loss
Backward pass order: loss → lm_head → layer11 → ... → layer1 → layer0
```

If you accessed `layer5.output` and `layer10.output` during the forward pass, you must access their gradients in reverse: `layer10.grad` first, then `layer5.grad`.

### Accessing Gradients

```python
with model.trace("Hello"):
    # Get the tensor and enable gradients
    hs = model.transformer.h[-1].output[0]
    hs.requires_grad_(True)
    
    logits = model.lm_head.output
    loss = logits.sum()
    
    # Access gradients ONLY inside a backward context
    # This is a SEPARATE interleaving session
    with loss.backward():
        grad = hs.grad.save()

print(grad.shape)  # Now contains the gradient
```

### Modifying Gradients

```python
with model.trace("Hello"):
    hs = model.transformer.h[-1].output[0]
    hs.requires_grad_(True)
    
    logits = model.lm_head.output
    
    # Use backward as a context to access and modify gradients
    with logits.sum().backward():
        # Access gradient on the tensor
        hs_grad = hs.grad.save()
        
        # Modify gradient on the tensor
        hs.grad[:] = 0
```

### Retain Graph for Multiple Backward Passes

```python
with model.trace("Hello"):
    hs = model.transformer.h[-1].output[0]
    hs.requires_grad_(True)
    logits = model.lm_head.output
    
    with logits.sum().backward(retain_graph=True):
        grad1 = hs.grad.save()
    
    # Second backward pass
    modified_logits = logits * 2
    with modified_logits.sum().backward():
        grad2 = hs.grad.save()
```

### Standalone Backward (Outside a Trace)

You can also use backward tracing on its own, without wrapping it in a `model.trace()`:

```python
# First, run a forward pass to get tensors
with model.trace("Hello"):
    hs = model.transformer.h[-1].output[0]
    hs.requires_grad_(True)
    hs = hs.save()  # Save the tensor for later
    logits = model.lm_head.output.save()

# Then, trace the backward pass separately
loss = logits.sum()
with loss.backward():
    grad = hs.grad.save()

print(grad.shape)
```

This is useful when you want to compute gradients after inspecting forward pass results.

---

## Conditionals and Iteration

### Python Conditionals (v0.5+ Pattern)

Standard Python `if` statements work inside tracing contexts:

```python
with model.trace("Hello") as tracer:
    output = model.transformer.h[0].output[0]
    
    # Python conditionals work with real tensor values
    if torch.all(output < 100000):
        model.transformer.h[-1].output[0][:] = 0
    
    result = model.transformer.h[-1].output[0].save()
```

### Session-Level Conditionals

```python
with model.session() as session:
    with model.trace("Hello"):
        if torch.all(model.transformer.h[5].output[0] < 100000):
            model.transformer.h[-1].output[0][:] = 0
        
        output = model.transformer.h[-1].output[0].save()
```

### Python Loops

Standard Python `for` loops work in session contexts:

```python
with model.session() as session:
    results = list().save()
    
    for prompt in ["Hello", "World", "Test"]:
        with model.trace(prompt):
            results.append(model.lm_head.output.argmax(dim=-1))
```

---

## Model Editing

Create persistently modified versions of a model:

```python
# Non-inplace editing (creates a new model reference)
with model.edit() as model_edited:
    model.transformer.h[1].output[0][:, 1] = 0

# Use original model
with model.trace("Hello"):
    out1 = model.transformer.h[1].output[0].save()

# Use edited model
with model_edited.trace("Hello"):
    out2 = model_edited.transformer.h[1].output[0].save()
```

### In-Place Editing

```python
with model.edit(inplace=True):
    model.transformer.h[1].output[0][:] = 0

# Now ALL traces use the edited model
with model.trace("Hello"):
    output = model.transformer.h[1].output[0].save()  # Will be zeros
```

### Clearing Edits

```python
# Remove all in-place edits
model.clear_edits()
```

---

## Scanning and Validation

### Scan Mode

Get shapes and types without running the full model:

```python
with model.scan("Hello"):
    # Access shape information
    dim = model.transformer.h[0].output[0].shape[-1]
    
print(dim)  # e.g., 768
```

### Validation Mode

Test interventions with fake tensors:

```python
# Validate interventions before running
with model.scan("Hello"):
    model.transformer.h[0].output[0][:, 10] = 0  # Will fail if dim < 11
```

---

## Caching Activations

Automatically cache outputs from multiple modules:

```python
with model.trace("Hello") as tracer:
    cache = tracer.cache()  # Cache all modules

# Access cached values
layer0_out = cache['model.transformer.h.0'].output
```

### Cache Specific Modules

```python
with model.trace("Hello") as tracer:
    cache = tracer.cache(modules=[
        model.transformer.h[0],
        model.transformer.h[1],
        model.lm_head
    ])
```

### Cache Inputs Too

```python
with model.trace("Hello") as tracer:
    cache = tracer.cache(include_inputs=True)

# Access inputs
layer1_input = cache['model.transformer.h.1'].inputs
```

### Cache with Interventions

```python
with model.trace("Hello") as tracer:
    cache = tracer.cache()  # Must call BEFORE interventions
    
    model.transformer.h[0].output[0][:] = 0

# Cache contains the modified values
assert torch.all(cache['model.transformer.h.0'].output[0] == 0)
```

### Attribute-Style Access

```python
with model.trace("Hello") as tracer:
    cache = tracer.cache()

# Both work:
out1 = cache['model.transformer.h.0'].output
out2 = cache.model.transformer.h[0].output
```

---

## Source Tracing

`.source` enables access to **intermediate operations** within a module's forward pass. When you access `.source`, nnsight **rewrites the module's forward method** to hook into all operations (function calls, method calls, etc.) so you can intercept their inputs and outputs.

### How Source Works

1. **Forward Rewriting**: Accessing `.source` injects hooks into every operation in the module's forward method
2. **Operation Discovery**: Print `.source` to see all available operations with their names and line numbers
3. **Operation Access**: Use `.source.<operation_name>` to access a specific operation
4. **Standard Access**: Operations have `.input`, `.inputs`, and `.output` just like modules

### Discovering Available Operations

Print `.source` to see the forward method with operation names highlighted:

```python
# Outside a trace - discover operations
print(model.transformer.h[0].attn.source)

# Output shows operation names and line numbers:
#                                    60
#                                    61     if using_eager and self.reorder_and_upcast_attn:
#   self__upcast_and_reordered_attn_0 -> 62         attn_output, attn_weights = self._upcast_and_reordered_attn(
#                                    63             query_states, key_states, value_states, attention_mask, head_mask
#                                    64         )
#                                    65     else:
#   attention_interface_0             -> 66         attn_output, attn_weights = attention_interface(
#                                    ...
#   attn_output_reshape_0             -> 78     attn_output = attn_output.reshape(...)
#   self_c_proj_0                     -> 79     attn_output = self.c_proj(attn_output)
#   self_resid_dropout_0              -> 80     attn_output = self.resid_dropout(attn_output)
```

### Viewing a Specific Operation

Print a specific operation to see it with surrounding context:

```python
print(model.transformer.h[0].attn.source.attention_interface_0)

# Output shows the operation highlighted:
# .transformer.h.0.attn.attention_interface_0:
#
#      ....
#
#          if using_eager and self.reorder_and_upcast_attn:
#              attn_output, attn_weights = self._upcast_and_reordered_attn(
#                  query_states, key_states, value_states, attention_mask, head_mask
#              )
#          else:
#      -->     attn_output, attn_weights = attention_interface( <--
#                  self,
#                  query_states,
#                  key_states,
#      ....
```

### Accessing Operation Values

Inside a trace, access operations like any module:

```python
with model.trace("Hello"):
    # Access operation output
    attn_out = model.transformer.h[0].attn.source.attention_interface_0.output.save()
    
    # Access operation inputs (args, kwargs)
    attn_args, attn_kwargs = model.transformer.h[0].attn.source.attention_interface_0.inputs
    
    # Modify operation output
    model.transformer.h[0].attn.source.self_c_proj_0.output[:] = 0
```

### Recursive Source Tracing

You can trace into operations that call other functions:

```python
with model.trace("Hello"):
    # Access nested internal operations
    sdpa_out = (
        model.transformer.h[0].attn
        .source.attention_interface_0
        .source.torch_nn_functional_scaled_dot_product_attention_0
        .output.save()
    )
```

**Note:** Don't call `.source` on a module from within another `.source`. Access it directly:

```python
# WRONG
model.transformer.h[0].attn.source.some_submodule.source  # Error!

# CORRECT - access the submodule directly
model.transformer.h[0].attn.some_submodule.source
```

---

## Module Skipping

Skip a module's computation entirely:

```python
with model.trace("Hello"):
    # Get output from layer 0
    layer0_out = model.transformer.h[0].output
    
    # Skip layer 1 entirely, using layer 0's output as layer 1's output
    model.transformer.h[1].skip(layer0_out)
    
    # Layer 1's output now equals layer 0's output
    layer1_out = model.transformer.h[1].output.save()

assert torch.equal(layer0_out[0], layer1_out[0])
```

### Skipping Constraints

- Cannot access inner modules of a skipped module
- Skips must respect execution order (can't skip backwards)

---

## vLLM Integration

NNsight supports vLLM for high-performance inference:

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("gpt2", tensor_parallel_size=1, gpu_memory_utilization=0.1, dispatch=True)
```

### Basic vLLM Tracing

```python
with model.trace("The Eiffel Tower is in", temperature=0.0, top_p=1):
    logits = model.logits.output.save()

next_token = model.tokenizer.decode(logits.argmax(dim=-1))
```

### vLLM Multi-Token Generation

```python
with model.trace("Hello", max_tokens=5) as tracer:
    logits = list().save()
    
    with tracer.iter[:]:
        logits.append(model.logits.output)
```

### vLLM Interventions

```python
with model.trace("The Eiffel Tower is in", temperature=0.0, top_p=1):
    # Modify hidden states
    model.transformer.h[-2].mlp.output = torch.zeros_like(
        model.transformer.h[-2].mlp.output
    )
    
    logits = model.logits.output.save()
```

### vLLM Sampling

```python
with model.trace(max_tokens=3) as tracer:
    with tracer.invoke("Hello", temperature=0.8, top_p=0.95):
        samples = list().save()
        with tracer.iter[:]:
            samples.append(model.samples.output.item())
```

---

## Remote Execution (NDIF)

Run interventions on NDIF's remote infrastructure without local GPU resources.

### Setup

```python
from nnsight import CONFIG
import nnsight

# Set API key (get from login.ndif.us)
CONFIG.set_default_api_key("YOUR_API_KEY")

# Check available models
nnsight.ndif_status()

# Check if specific model is running (may return False even if available)
nnsight.is_model_running("openai-community/gpt2")
```

### Basic Remote Tracing

```python
# Model loads on 'meta' device - no local GPU memory used
model = LanguageModel("meta-llama/Llama-3.1-8B")
print(model.device)  # "meta"

# Add remote=True to execute on NDIF
with model.trace("The Eiffel Tower is in the city of", remote=True):
    logit = model.lm_head.output[0][-1].argmax(dim=-1).save()

print(model.tokenizer.decode(logit))  # "Paris"
```

### Remote Generation

```python
with model.generate("Hello", max_new_tokens=5, remote=True) as tracer:
    # Use tracer.result for generation output
    output = tracer.result.save()
    
    # Or iterate over generation steps
    logits = list().save()
    with tracer.iter[:]:
        logits.append(model.lm_head.output[0][-1].argmax(dim=-1))

print(model.tokenizer.decode(output[0]))
```

### Request Lifecycle

When you run a remote trace, you'll see status updates:

| Status | Description |
|--------|-------------|
| `RECEIVED` | Request validated with API key |
| `QUEUED` | Waiting in model's queue |
| `DISPATCHED` | Forwarded to model deployment |
| `RUNNING` | Executing your intervention |
| `LOG` | Print statements from your code |
| `COMPLETED` | Results ready for download |

### Print Statements in Remote Traces

Print statements inside remote traces are captured and sent back as `LOG` status:

```python
with model.trace("Hello", remote=True):
    hidden = model.transformer.h[0].output[0]
    print(f"Hidden shape: {hidden.shape}")  # Appears as LOG
    print(f"Hidden mean: {hidden.mean()}")  # Appears as LOG
    output = model.lm_head.output.save()
```

### Saving Results Remotely

**Critical:** `.save()` is how values are transmitted back to your local environment.

```python
# WRONG: Local list won't be updated
my_list = list()
with model.trace("Hello", remote=True):
    my_list.append(model.output.save())  # Local list stays empty!

# CORRECT: Create and save inside the trace
with model.trace("Hello", remote=True):
    my_list = list().save()  # Create inside trace
    my_list.append(model.output)

# Best practice: move tensors to CPU for smaller downloads
with model.trace("Hello", remote=True):
    hidden = model.transformer.h[0].output[0].detach().cpu().save()
```

### Non-Blocking Execution

Submit jobs without waiting:

```python
with model.trace("Hello", remote=True, blocking=False) as tracer:
    output = model.lm_head.output.save()

# Get backend to check status
backend = tracer.backend
print(backend.job_id)      # UUID
print(backend.job_status)  # JobStatus.RECEIVED

# Poll for result
import time
while True:
    result = backend()  # Returns None if not complete
    if result is not None:
        break
    print(f"Status: {backend.job_status}")
    time.sleep(1)

# Result is a dict with variable names as keys
print(result.keys())        # dict_keys(['id', 'output'])
print(result['output'].shape)
```

### Disable Remote Logging

```python
CONFIG.APP.REMOTE_LOGGING = False
```

---

## Sessions

Group multiple traces for shared state and efficiency.

### Local Sessions

```python
with model.session() as session:
    # First trace
    with model.trace("Hello"):
        hs1 = model.transformer.h[0].output[0].save()
    
    # Second trace - can reference values from first
    with model.trace("World"):
        model.transformer.h[0].output[0][:] = hs1
        hs2 = model.transformer.h[0].output[0].save()
```

### Remote Sessions

Sessions are especially powerful for remote execution — they bundle multiple traces into a **single request**:

```python
# Single request, single queue wait
with model.session(remote=True):
    # First trace: capture hidden states
    with model.trace("Megan Rapinoe plays the sport of"):
        hs = model.model.layers[5].output[0][:, -1, :]  # No .save() needed!
    
    # Second trace: clean baseline
    with model.trace("Shaquille O'Neal plays the sport of"):
        clean = model.lm_head.output[0][-1].argmax(dim=-1).save()
    
    # Third trace: patch the hidden states
    with model.trace("Shaquille O'Neal plays the sport of"):
        model.model.layers[5].output[0][:, -1, :] = hs  # Direct reference works!
        patched = model.lm_head.output[0][-1].argmax(dim=-1).save()

print(f"Clean: {model.tokenizer.decode(clean)}")
print(f"Patched: {model.tokenizer.decode(patched)}")
```

**Benefits of remote sessions:**
- Single queue wait for all traces
- Values from earlier traces accessible directly (no `.save()` needed)
- Only put `remote=True` on the session, not inner traces

---

## Module Renaming

Create aliases for easier access:

```python
model = LanguageModel(
    "openai-community/gpt2",
    rename={
        "transformer.h": "layers",           # Mount at new path
        "mlp": "feedforward",                # Rename all MLPs
        ".transformer": ["model", "backbone"] # Multiple aliases
    }
)

# Now both work:
with model.trace("Hello"):
    out1 = model.layers[0].feedforward.output.save()
    out2 = model.transformer.h[0].mlp.output.save()  # Original still works
```

---

## Common Patterns

### Activation Patching

```python
with model.trace() as tracer:
    # Clean run
    with tracer.invoke("The Eiffel Tower is in"):
        clean_hs = model.transformer.h[5].output[0][:, -1, :].save()
    
    # Patched run
    with tracer.invoke("The Colosseum is in"):
        model.transformer.h[5].output[0][:, -1, :] = clean_hs
        patched_logits = model.lm_head.output.save()
```

### Logit Lens

```python
with model.trace("The Eiffel Tower is in"):
    # Apply final layer norm and lm_head to intermediate layers
    for i in range(12):
        hs = model.transformer.h[i].output[0]
        logits = model.lm_head(model.transformer.ln_f(hs))
        tokens = logits.argmax(dim=-1).save()
        print(f"Layer {i}:", model.tokenizer.decode(tokens[0][-1]))
```

**Why this works:** When you call `model.lm_head(...)` inside a trace, it uses `.forward()` instead of `__call__()`, bypassing interleaving hooks. This means the module's computation runs without triggering `.input`/`.output` hooks.

### Using Auxiliary Modules (SAEs, LoRA, etc.)

When you add auxiliary modules to a model (like SAEs or LoRA adapters), use `hook=True` to enable `.input`/`.output` access on them:

```python
# Assume model.sae is an auxiliary SAE module you've added
with model.trace() as tracer:
    # First invoke: apply the SAE with hooks enabled
    with tracer.invoke("Hello"):
        hidden = model.transformer.h[5].output[0]
        reconstructed = model.sae(hidden, hook=True)  # Enable hooks!
        model.transformer.h[5].output[0] = reconstructed
    
    # Second invoke: access the SAE's activations
    with tracer.invoke("Hello"):
        sae_activations = model.sae.output.save()  # Works because we used hook=True
```

### Ablation Study

```python
with model.trace() as tracer:
    with tracer.invoke("Hello World"):
        # Baseline
        baseline = model.lm_head.output[:, -1].save()
    
    with tracer.invoke("Hello World"):
        # Ablate specific layer
        model.transformer.h[5].mlp.output[:] = 0
        ablated = model.lm_head.output[:, -1].save()

diff = (baseline - ablated).abs().mean()
```

### Attention Pattern Extraction

```python
with model.trace("Hello World"):
    # Access attention weights (model-specific)
    attn_weights = model.transformer.h[0].attn.source.attention_interface_0.output[0].save()
```

### Steering with Added Vectors

```python
steering_vector = torch.randn(768)  # Pre-computed direction

with model.trace("Hello"):
    model.transformer.h[10].output[0][:, -1, :] += steering_vector * 0.5
    output = model.lm_head.output.save()
```

---

## Critical Gotchas

### 1. Forgetting `.save()`

```python
# WRONG - value is garbage collected
with model.trace("Hello"):
    output = model.transformer.h[-1].output[0]
# output is now useless

# CORRECT
with model.trace("Hello"):
    output = model.transformer.h[-1].output[0].save()
```

### 2. In-Place vs Replacement

```python
# In-place modification (modifies existing tensor)
model.transformer.h[0].output[0][:] = 0

# Replacement (creates new tensor)
model.transformer.h[0].output[0] = torch.zeros_like(model.transformer.h[0].output[0])
```

### 3. Tuple Outputs

```python
# WRONG - trying to assign to tuple element
model.transformer.h[0].output[0] = new_tensor  # This replaces the whole tuple!

# CORRECT for in-place on first element
model.transformer.h[0].output[0][:] = 0

# CORRECT for replacing tuple element
model.transformer.h[0].output = (new_tensor,) + model.transformer.h[0].output[1:]
```

### 4. Clone Before In-Place Modification

```python
# WRONG - modifying and trying to see original
with model.trace("Hello"):
    before = model.transformer.h[0].output[0].save()  # Points to same tensor!
    model.transformer.h[0].output[0][:] = 0
    after = model.transformer.h[0].output[0].save()
# before == after because both point to modified tensor

# CORRECT
with model.trace("Hello"):
    before = model.transformer.h[0].output[0].clone().save()
    model.transformer.h[0].output[0][:] = 0
    after = model.transformer.h[0].output[0].save()
```

### 5. Module Access Must Be in Execution Order

**Within a single invoke**, you MUST access modules in the order they execute in the forward pass. This is because each invoke is a serial thread that waits for values as the model runs forward:

```python
with model.trace("Hello"):
    # CORRECT - access in execution order (layer 1 runs before layer 5)
    out1 = model.transformer.h[1].output.save()  # Thread waits here
    out5 = model.transformer.h[5].output.save()  # Then waits here

with model.trace("Hello"):
    # WRONG - will deadlock! Layer 1 has already passed when you ask for it
    out5 = model.transformer.h[5].output.save()  # Thread waits here
    out1 = model.transformer.h[1].output.save()  # Deadlock! Layer 1 already ran
```

**To access modules "out of order"**, use separate invokes (which are separate forward passes):

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        out5 = model.transformer.h[5].output.save()
    
    with tracer.invoke():  # No-arg invoke on same batch (new forward pass)
        out1 = model.transformer.h[1].output.save()  # Works! Separate thread/pass
```

### 6. Trace Requires Input Without Invokes

```python
# WRONG - no input and no invokes
with model.trace():  # Error!
    output = model.output.save()

# CORRECT - provide input
with model.trace("Hello"):
    output = model.output.save()

# CORRECT - use explicit invokes
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        output = model.output.save()
```

### 7. Values Are Real Tensors (Not Proxies)

In nnsight's thread-based architecture, when you access `.output`, your thread waits and receives the **actual tensor**:

```python
with model.trace("Hello"):
    # Thread waits here and gets the REAL tensor
    hs = model.transformer.h[0].output[0]
    
    # This is a real shape, real operations work directly
    shape = hs.shape  # torch.Size([1, 5, 768])
    zeros = torch.zeros(shape)  # Real tensor operation
    
    # Printing works normally
    print(shape)  # torch.Size([1, 5, 768])
```

Use `.scan()` if you need shapes **without** running the model:

```python
with model.scan("Hello"):
    shape = model.transformer.h[0].output[0].shape  # Shape via fake tensors
```

### 8. Generation vs Trace

```python
# Use .trace() for single forward pass
with model.trace("Hello"):
    output = model.output.save()

# Use .generate() for multi-token generation
with model.generate("Hello", max_new_tokens=5):
    output = model.generator.output.save()
```

### 9. Device Placement

```python
# Tensors must be on the correct device
with model.trace("Hello"):
    device = model.transformer.h[0].output[0].device
    noise = torch.randn(768).to(device)  # Match device!
    model.transformer.h[0].output[0][:, -1, :] += noise
```

---

## Understanding Exceptions in NNsight

Debugging in NNsight can be tricky because of **deferred execution**. Your intervention code is captured, compiled into a function, and run in a worker thread — not where you wrote it. Without special handling, exceptions would point to internal NNsight code instead of your original trace.

### How NNsight Fixes This

NNsight **reconstructs exception tracebacks** to show your original code and line numbers. When an exception occurs:

1. NNsight catches it at the trace boundary
2. Maps the internal line numbers back to your source file
3. Rebuilds the traceback to look like a normal Python exception

**What you see:**
```
Traceback (most recent call last):
  File "my_experiment.py", line 6, in <module>
    hidden = model.transformer.h[100].output.save()
IndexError: list index out of range
```

This points directly to your code, even though it actually ran in a compiled worker thread.

### Exception Type Preservation

NNsight preserves the original exception type, so you can still catch specific exceptions:

```python
try:
    with model.trace("Hello"):
        hidden = model.transformer.h[100].output.save()
except IndexError:
    print("Caught the IndexError!")  # This works!
```

### DEBUG Mode

By default, NNsight hides its internal frames from tracebacks. If the default traceback isn't helpful, enable DEBUG mode to see the full execution path:

```python
from nnsight import CONFIG
CONFIG.APP.DEBUG = True
```

This shows internal NNsight frames, which can help:
- Understand where in the pipeline an error occurred
- Provide more context when the user-facing traceback is unclear

### Common Exceptions

| Exception | Cause | Fix |
|-----------|-------|-----|
| `OutOfOrderError: Value was missed for model.layer.output.i0` | Accessed modules in wrong order within an invoke | Access modules in forward-pass order |
| `ValueError: Execution complete but ... was not provided` | Mediator still waiting - module never called or gradient order wrong | Check module path exists; access gradients in reverse order |
| `ValueError: Cannot return output of Envoy that is not interleaving` | Trace has no input, model never ran | Provide input to `.trace(input)` or use `.invoke()` |
| `ValueError: Cannot invoke during an active model execution` | Tried to create invoke inside another invoke | Use sequential, non-nested invokes |
| `ValueError: Cannot request ... in a backwards tracer` | Accessed `.output`/`.input` inside `backward()` instead of `.grad` | Define tensors before backward, access `.grad` inside |
| `AttributeError: ... has no attribute X` | Nonexistent module accessed | Use `print(model)` to see available modules |
| `AttributeError: Tokenizer not found` | Wrapped pre-loaded model without providing tokenizer | Provide `tokenizer=` when wrapping pre-loaded models |
| `NotImplementedError: Batching not implemented` | Multiple invokes on base `NNsight` | Use `LanguageModel` for multiple invokes, or single invoke |

**The `.i0` suffix** in error messages indicates iteration 0 (first call). In generation, you'd see `.i1`, `.i2`, etc.

**Gradient errors show tensor IDs** (like `139820463417744.grad`) instead of module paths because gradients are tracked per-tensor, not per-module.

---

## Debugging Tips

### 1. Use Print and Breakpoints

Print works normally inside traces:

```python
with model.trace("Hello"):
    out = model.transformer.h[0].output[0]
    print("Shape:", out.shape)
    print("Mean:", out.mean())
```

Use `breakpoint()` for interactive debugging:

```python
with model.trace("Hello"):
    out = model.transformer.h[0].output
    breakpoint()  # Drops into pdb
```

### 2. Enable Scan and Validate

```python
with model.trace("Hello", scan=True, validate=True):
    # Errors will be caught with fake tensors before real execution
    model.transformer.h[0].output[0][:, 1000] = 0  # Will fail early if dim < 1001
```

### 3. Check Module Structure

```python
# Print full model structure
print(model)

# Check specific module
print(model.transformer.h[0])
```

### 4. Check Shapes with Scan

Use `.scan()` to get shapes without running the full model:

```python
with model.scan("Hello"):
    print(model.transformer.h[0].output[0].shape)  # (1, seq_len, hidden_dim)
    print(model.lm_head.output.shape)  # (1, seq_len, vocab_size)
```

### 5. Debug Configuration

```python
import nnsight

# Enable detailed error messages
nnsight.CONFIG.APP.DEBUG = True
nnsight.CONFIG.save()
```

---

## Configuration

NNsight has several configuration options accessible via `nnsight.CONFIG`:

```python
from nnsight import CONFIG

# Debug mode - more verbose error messages
CONFIG.APP.DEBUG = True

# Cross-invoker variable sharing (default: True)
# When True, variables from one invoke can be accessed in another
# When False, each invoke is isolated (useful for debugging)
CONFIG.APP.CROSS_INVOKER = True

# Pymount - enables obj.save() on base Python objects (default: True)
# Uses C extension to add .save() method to base object class
# Set to False if you prefer nnsight.save() exclusively
CONFIG.APP.PYMOUNT = True

# Save config changes
CONFIG.save()
```

### Cross-Invoker Variable Sharing

By default, variables from one invoke can be used in another (because invokes run serially):

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        embeddings = model.transformer.wte.output  # Captured here
    
    with tracer.invoke("World"):
        model.transformer.wte.output = embeddings  # Used here (works because CROSS_INVOKER=True)
```

If you're debugging and want to ensure invokes are isolated:

```python
CONFIG.APP.CROSS_INVOKER = False  # Now cross-invoke references will error
```

### Barrier Synchronization

When **both invokes access the same module**, you need to synchronize them so variables can be shared at that point:

```python
with llm.trace() as tracer:
    
    barrier = tracer.barrier(2)  # Create barrier for 2 participants
    
    with tracer.invoke("The Eiffel Tower is in"):
        paris_embeddings = llm.transformer.wte.output
        barrier()  # Wait here
    
    with tracer.invoke("_ _ _ _ _"):
        barrier()  # Synchronize
        llm.transformer.wte.output = paris_embeddings  # Now available!
        patched_output = llm.lm_head.output[:, -1].save()
```

Without the barrier, the second invoke would fail with `paris_embeddings is not defined` because both invokes access `wte.output` and invokes normally run serially.

### Re-wrapping the Same Model

If you wrap the same PyTorch model with `NNsight` multiple times, hooks are properly cleaned up and re-applied (not stacked):

```python
model1 = NNsight(my_pytorch_model)
model2 = NNsight(my_pytorch_model)  # Safe - hooks replaced, not duplicated
```

---

## API Quick Reference

### Context Managers

| Method | Description |
|--------|-------------|
| `model.trace(input)` | Single forward pass with interventions |
| `model.generate(input, max_new_tokens=N)` | Multi-token generation |
| `model.scan(input)` | Get shapes without full execution |
| `model.edit()` | Create persistent model modifications |
| `model.session()` | Group multiple traces |

### Tracer Methods

| Method | Description |
|--------|-------------|
| `tracer.invoke(input)` | Add input to batch |
| `tracer.barrier(n)` | Synchronization barrier |
| `tracer.cache(...)` | Activation caching |
| `tracer.stop()` | Early termination |
| `tracer.iter[slice]` | Iterate generation steps |
| `tracer.all()` | Apply to all steps |
| `tracer.result` | Get final output of traced function |

### Module Properties

| Property | Description |
|----------|-------------|
| `.output` | Module output |
| `.input` | First positional input |
| `.inputs` | All inputs `(args, kwargs)` |
| `.source` | Internal operation tracing |
| `.next()` | Advance to next generation step |
| `.skip(value)` | Skip module with given output |


**Note on `.grad`:** Access gradients on **tensors** only inside a `with tensor.backward():` context. See [Gradients and Backpropagation](#gradients-and-backpropagation).

---

## Version Notes

This guide is written for **nnsight v0.5+**. Key changes from earlier versions:

- Standard Python `if`/`for` statements now work inside tracing contexts (replaces `nnsight.cond()`, `session.iter()`)
- `nnsight.apply()` is deprecated - use functions directly
- `nnsight.list()`, `nnsight.dict()`, etc. are deprecated - use standard Python types with `.save()`
- `nnsight.local()` and `nnsight.trace` decorator are deprecated

---

## File Structure Reference

```
nnsight/
├── src/nnsight/
│   ├── __init__.py          # Main exports
│   ├── modeling/
│   │   ├── base.py          # NNsight class
│   │   ├── language.py      # LanguageModel class
│   │   └── vllm/            # vLLM integration
│   └── intervention/
│       ├── envoy.py         # Core Envoy wrapper
│       ├── interleaver.py   # Execution interleaving
│       └── tracing/         # Tracer implementations
└── tests/
    ├── test_tiny.py         # Basic NNsight tests
    ├── test_lm.py           # LanguageModel tests
    └── test_vllm.py         # vLLM tests
```

