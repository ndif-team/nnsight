<img src="./nnsight_logo.svg" alt="drawing" style="width:200px;float:left"/>

# nnsight 

[![arXiv](https://img.shields.io/badge/READ%20THE%20PAPER%20HERE!-orange)](https://arxiv.org/abs/2407.14561)
[![Docs](https://img.shields.io/badge/-Read%20the%20Docs%20Here-blue?style=for-the-badge&logo=Read-the-Docs&logoColor=white)](https://www.nnsight.net)
[![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/6uFJmCSwW7)

The `nnsight` package enables interpreting and manipulating the internals of deep learned models. Read our [paper!](https://arxiv.org/abs/2407.14561)

## Installation

```bash
pip install nnsight
```

## Quick Start

```python
from nnsight import LanguageModel

model = LanguageModel('openai-community/gpt2', device_map='auto', dispatch=True)

with model.trace('The Eiffel Tower is in the city of'):
    # Access and save hidden states
    hidden_states = model.transformer.h[-1].output[0].save()
    
    # Intervene on activations
    model.transformer.h[0].output[0][:] = 0
    
    # Get model output
    output = model.output.save()

print(model.tokenizer.decode(output.logits.argmax(dim=-1)[0]))
```

---

## Accessing Activations

```python
with model.trace("The Eiffel Tower is in the city of"):
    # Access attention output
    attn_output = model.transformer.h[0].attn.output[0].save()
    
    # Access MLP output
    mlp_output = model.transformer.h[0].mlp.output.save()

    # Access any layer's output (access in execution order)
    layer_output = model.transformer.h[5].output[0].save()
    
    # Access final logits
    logits = model.lm_head.output.save()
```

**Note:** GPT-2 transformer layers return tuples where index 0 contains the hidden states.

---

## Modifying Activations

### In-Place Modification

```python
with model.trace("Hello"):
    # Zero out all activations
    model.transformer.h[0].output[0][:] = 0
    
    # Modify specific positions
    model.transformer.h[0].output[0][:, -1, :] = 0  # Last token only
```

### Replacement

```python
with model.trace("Hello"):
    # Add noise to activations
    hs = model.transformer.h[-1].mlp.output.clone()
    noise = 0.01 * torch.randn(hs.shape)
    model.transformer.h[-1].mlp.output = hs + noise
    
    result = model.transformer.h[-1].mlp.output.save()
```

---

## Batching with Invokers

Process multiple inputs in one forward pass. Each invoke runs its code in a **separate worker thread**:

- Threads execute serially (no race conditions)
- Each thread waits for values via `.output`, `.input`, etc.
- Invokes run in the order they're defined
- Cross-invoke references work because threads run sequentially
- **Within an invoke, access modules in execution order only**

```python
with model.trace() as tracer:
    # First invoke: worker thread 1
    with tracer.invoke("The Eiffel Tower is in"):
        embeddings = model.transformer.wte.output  # Thread waits here
        output1 = model.lm_head.output.save()
    
    # Second invoke: worker thread 2 (runs after thread 1 completes)
    with tracer.invoke("_ _ _ _ _ _"):
        model.transformer.wte.output = embeddings  # Uses value from thread 1
        output2 = model.lm_head.output.save()
```

### Prompt-less Invokers

Use `.invoke()` with no arguments to operate on the entire batch:

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        out1 = model.lm_head.output[:, -1].save()
    
    with tracer.invoke(["World", "Test"]):
        out2 = model.lm_head.output[:, -1].save()
    
    # No-arg invoke: operates on ALL 3 inputs
    with tracer.invoke():
        out_all = model.lm_head.output[:, -1].save()  # Shape: [3, vocab]
```

---

## Multi-Token Generation

Use `.generate()` for autoregressive generation:

```python
with model.generate("The Eiffel Tower is in", max_new_tokens=3) as tracer:
    output = model.generator.output.save()

print(model.tokenizer.decode(output[0]))
# "The Eiffel Tower is in the city of Paris"
```

### Iterating Over Generation Steps

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    logits = list().save()
    
    # Iterate over all generation steps
    with tracer.iter[:]:
        logits.append(model.lm_head.output[0][-1].argmax(dim=-1))

print(model.tokenizer.batch_decode(logits))
```

### Conditional Interventions Per Step

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    outputs = list().save()
    
    with tracer.iter[:] as step_idx:
        if step_idx == 2:
            model.transformer.h[0].output[0][:] = 0  # Only on step 2
        
        outputs.append(model.transformer.h[-1].output[0])
```

---

## Gradients

Gradients are accessed on **tensors** (not modules), only inside a `with tensor.backward():` context:

```python
with model.trace("Hello"):
    hs = model.transformer.h[-1].output[0]
    hs.requires_grad_(True)
    
    logits = model.lm_head.output
    loss = logits.sum()
    
    with loss.backward():
        grad = hs.grad.save()

print(grad.shape)
```

---

## Model Editing

Create persistent model modifications:

```python
# Create edited model (non-destructive)
with model.edit() as model_edited:
    model.transformer.h[0].output[0][:] = 0

# Original model unchanged
with model.trace("Hello"):
    out1 = model.transformer.h[0].output[0].save()

# Edited model has modification
with model_edited.trace("Hello"):
    out2 = model_edited.transformer.h[0].output[0].save()

assert not torch.all(out1 == 0)
assert torch.all(out2 == 0)
```

---

## Scanning (Shape Inference)

Get shapes without running the full model:

```python
with model.scan("Hello"):
    dim = model.transformer.h[0].output[0].shape[-1]

print(dim)  # 768
```

---

## Caching Activations

Automatically cache outputs from modules:

```python
with model.trace("Hello") as tracer:
    cache = tracer.cache()

# Access cached values
layer0_out = cache['model.transformer.h.0'].output
print(cache.model.transformer.h[0].output[0].shape)
```

---

## Sessions

Group multiple traces for efficiency:

```python
with model.session() as session:
    with model.trace("Hello"):
        hs1 = model.transformer.h[0].output[0].save()
    
    with model.trace("World"):
        model.transformer.h[0].output[0][:] = hs1  # Use value from first trace
        hs2 = model.transformer.h[0].output[0].save()
```

---

## Remote Execution (NDIF)

Run on NDIF's remote infrastructure:

```python
from nnsight import CONFIG
CONFIG.set_default_api_key("YOUR_API_KEY")

model = LanguageModel("meta-llama/Meta-Llama-3.1-8B")

with model.trace("Hello", remote=True):
    hidden_states = model.model.layers[-1].output.save()
```

Check available models at [nnsight.net/status](https://nnsight.net/status/)

---

## vLLM Integration

High-performance inference with vLLM:

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("gpt2", tensor_parallel_size=1, dispatch=True)

with model.trace("Hello", temperature=0.0, max_tokens=5) as tracer:
    logits = list().save()
    
    with tracer.iter[:]:
        logits.append(model.logits.output)
```

---

## NNsight for Any PyTorch Model

Use `NNsight` for arbitrary PyTorch models:

```python
from nnsight import NNsight
import torch

net = torch.nn.Sequential(
    torch.nn.Linear(5, 10),
    torch.nn.Linear(10, 2)
)

model = NNsight(net)

with model.trace(torch.rand(1, 5)):
    layer1_out = model[0].output.save()
    output = model.output.save()
```

---

## Source Tracing

Access intermediate operations inside a module's forward pass. `.source` rewrites the forward method to hook into all operations:

```python
# Discover available operations
print(model.transformer.h[0].attn.source)
# Shows forward method with operation names like:
#   attention_interface_0 -> 66  attn_output, attn_weights = attention_interface(...)
#   self_c_proj_0         -> 79  attn_output = self.c_proj(attn_output)

# Access operation values
with model.trace("Hello"):
    attn_out = model.transformer.h[0].attn.source.attention_interface_0.output.save()
```

---

## Ad-hoc Module Application

Apply modules out of their normal execution order:

```python
with model.trace("The Eiffel Tower is in the city of"):
    # Get intermediate hidden states
    hidden_states = model.transformer.h[-1].output[0]
    
    # Apply lm_head to get "logit lens" view
    logits = model.lm_head(model.transformer.ln_f(hidden_states))
    tokens = logits.argmax(dim=-1).save()

print(model.tokenizer.decode(tokens[0]))
```

---

## Core Concepts

### Deferred Execution with Thread-Based Synchronization

NNsight uses **deferred execution** with **thread-based synchronization**:

1. **Code extraction**: When you enter a `with model.trace(...)` block, nnsight captures your code (via AST) and immediately exits the block
2. **Thread execution**: Your code runs in a separate worker thread
3. **Value waiting**: When you access `.output`, the thread **waits** until the model provides that value
4. **Hook-based injection**: The model uses PyTorch hooks to provide values to waiting threads

```python
with model.trace("Hello"):
    # Code runs in a worker thread
    # Thread WAITS here until layer output is available
    hs = model.transformer.h[-1].output[0]
    
    # .save() marks the value to persist after the context exits
    hs = hs.save()

# After exiting, hs contains the actual tensor
print(hs.shape)  # torch.Size([1, 2, 768])
```

**Key insight:** Your code runs directly. When you access `.output`, you get the **real tensor** - your thread just waits for it to be available.

**Important:** Within an invoke, you must access modules in execution order. Accessing layer 5's output before layer 2's output will cause a deadlock (layer 2 has already been executed).

### Key Properties

Every module has these special properties. Accessing them causes the worker thread to **wait** for the value:

| Property | Description |
|----------|-------------|
| `.output` | Module's forward pass output (thread waits) |
| `.input` | First positional argument to the module |
| `.inputs` | All inputs as `(args_tuple, kwargs_dict)` |

**Note:** `.grad` is accessed on **tensors** (not modules), only inside a `with tensor.backward():` context.

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

---

## More Examples

Find more examples and tutorials at [nnsight.net](https://www.nnsight.net)

---

## Citation

If you use `nnsight` in your research, please cite:

```bibtex
@article{fiottokaufman2024nnsightndifdemocratizingaccess,
      title={NNsight and NDIF: Democratizing Access to Foundation Model Internals}, 
      author={Jaden Fiotto-Kaufman and Alexander R Loftus and Eric Todd and Jannik Brinkmann and Caden Juang and Koyena Pal and Can Rager and Aaron Mueller and Samuel Marks and Arnab Sen Sharma and Francesca Lucchetti and Michael Ripa and Adam Belfki and Nikhil Prakash and Sumeet Multani and Carla Brodley and Arjun Guha and Jonathan Bell and Byron Wallace and David Bau},
      year={2024},
      eprint={2407.14561},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.14561}, 
}
```
