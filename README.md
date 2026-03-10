<p align="center">
  <img src="./nnsight_logo.svg" alt="nnsight" width="300">
</p>

<h3 align="center">
Interpret and manipulate the internals of deep learning models
</h3>

<p align="center">
<a href="https://www.nnsight.net"><b>Documentation</b></a> | <a href="https://github.com/ndif-team/nnsight"><b>GitHub</b></a> | <a href="https://discord.gg/6uFJmCSwW7"><b>Discord</b></a> | <a href="https://discuss.ndif.us/"><b>Forum</b></a> | <a href="https://x.com/ndif_team"><b>Twitter</b></a> | <a href="https://arxiv.org/abs/2407.14561"><b>Paper</b></a>
</p>

<p align="center">
<a href="https://colab.research.google.com/github/ndif-team/nnsight/blob/main/NNsight_Walkthrough.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg"></img></a>
<a href="https://deepwiki.com/ndif-team/nnsight"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></img></a>
</p>

---

## About

**nnsight** is a Python library that enables interpreting and intervening on the internals of deep learning models. It provides a clean, Pythonic interface for:

- **Accessing activations** at any layer during forward passes
- **Modifying activations** to study causal effects
- **Computing gradients** with respect to intermediate values
- **Batching interventions** across multiple inputs efficiently

Originally developed in the [NDIF team](https://ndif.us/) at Northeastern University, nnsight supports local execution on any PyTorch model and remote execution on large models via the NDIF infrastructure.

> 📖 For a deeper technical understanding of nnsight's internals (tracing, interleaving, the Envoy system, etc.), see **[NNsight.md](./NNsight.md)**.

---

## Installation

```bash
pip install nnsight
```

---

## Agents

Inform LLM agents how to use nnsight using one of these methods:

### Skills Repository

**Claude Code**

```bash
# Open Claude Code terminal
claude

# Add the marketplace (one time)
/plugin marketplace add https://github.com/ndif-team/skills.git

# Install all skills
/plugin install nnsight@skills
```

**OpenAI Codex**

```bash
# Open OpenAI Codex terminal
codex

# Install skills
skill-installer install https://github.com/ndif-team/skills.git
```

### Context7 MCP

Alternatively, use [Context7](https://github.com/upstash/context7) to provide up-to-date nnsight documentation directly to your LLM. Add `use context7` to your prompts or configure it in your MCP client:

```json
{
  "mcpServers": {
    "context7": {
      "url": "https://mcp.context7.com/mcp"
    }
  }
}
```

See the [Context7 README](https://github.com/upstash/context7/blob/master/README.md) for full installation instructions across different IDEs.

### Documentation Files

You can also add our documentation files directly to your agent's context:

- **[CLAUDE.md](./CLAUDE.md)** — Comprehensive guide for AI agents working with nnsight
- **[NNsight.md](./NNsight.md)** — Deep technical documentation on nnsight's internals

---

## Quick Start

```python
from nnsight import LanguageModel

model = LanguageModel('openai-community/gpt2', device_map='auto', dispatch=True)

with model.trace('The Eiffel Tower is in the city of'):
    # Intervene on activations (must access in execution order!)
    model.transformer.h[0].output[0][:] = 0
    
    # Access and save hidden states from a later layer
    hidden_states = model.transformer.h[-1].output[0].save()
    
    # Get model output
    output = model.output.save()

print(model.tokenizer.decode(output.logits.argmax(dim=-1)[0]))
```

> **💡 Tip:** Always call `.save()` on values you want to access after the trace exits. Without `.save()`, values are garbage collected. You can also use `nnsight.save(value)` as an alternative.

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
    for step in tracer.iter[:]:
        logits.append(model.lm_head.output[0][-1].argmax(dim=-1))

print(model.tokenizer.batch_decode(logits))
```

### Conditional Interventions Per Step

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    outputs = list().save()
    
    for step_idx in tracer.iter[:]:
        if step_idx == 2:
            model.transformer.h[0].output[0][:] = 0  # Only on step 2

        outputs.append(model.transformer.h[-1].output[0])
```

> **⚠️ Warning:** Code after `tracer.iter[:]` never executes! The unbounded iterator waits forever for more steps. Put post-iteration code in a separate `tracer.invoke()`. When using multiple invokes, do not pass input to `generate()` — pass it to the first invoke:
> ```python
> with model.generate(max_new_tokens=3) as tracer:
>     with tracer.invoke("Hello"):  # First invoker — pass input here
>         for step in tracer.iter[:]:
>             hidden = model.transformer.h[-1].output.save()
>     with tracer.invoke():  # Second invoker — runs after generation
>         final = model.output.save()  # Now works!
> ```


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


## Scanning (Shape Inference)

Get shapes without running the full model. Like all tracing contexts, `.save()` is required to persist values outside the block:

```python
import nnsight

with model.scan("Hello"):
    dim = nnsight.save(model.transformer.h[0].output[0].shape[-1])

print(dim)  # 768
```


## Caching Activations

Automatically cache outputs from modules:

```python
with model.trace("Hello") as tracer:
    cache = tracer.cache()

# Access cached values
layer0_out = cache['model.transformer.h.0'].output
print(cache.model.transformer.h[0].output[0].shape)
```


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


## vLLM Integration

High-performance inference with vLLM:

```python
from nnsight.modeling.vllm import VLLM

model = VLLM("gpt2", tensor_parallel_size=1, dispatch=True)

with model.trace("Hello", temperature=0.0, max_tokens=5) as tracer:
    logits = list().save()
    
    for step in tracer.iter[:]:
        logits.append(model.logits.output)
```


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
    # Alternative: hs = nnsight.save(hs)

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

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `OutOfOrderError: Value was missed...` | Accessed modules in wrong order | Access modules in forward-pass execution order |
| `NameError` after `tracer.iter[:]` | Code after unbounded iter doesn't run | Use separate `tracer.invoke()` for post-iteration code; pass input to first invoke, not `generate()` |
| `ValueError: Cannot invoke during an active model execution` | Passed input to `generate()` while using multiple invokes | Use `model.generate(max_new_tokens=N)` with no input; pass prompt to first `tracer.invoke("Hello")` |
| `ValueError: Cannot return output of Envoy...` | No input provided to trace | Provide input: `model.trace(input)` or use `tracer.invoke(input)` |

For more debugging tips, see the [documentation](https://www.nnsight.net).

---

## More Resources

- **[Documentation](https://www.nnsight.net)** — Tutorials, guides, and API reference
- **[NNsight.md](./NNsight.md)** — Deep technical documentation on nnsight's internals
- **[CLAUDE.md](./CLAUDE.md)** — Comprehensive guide for AI agents working with nnsight
- **[Performance Report](./tests/performance/profile/results/performance_report.md)** — Detailed performance analysis and benchmarks

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
